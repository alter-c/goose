import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from abc import ABC, abstractmethod
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU
from torch import Tensor
from typing import Optional, List

torch.set_num_threads(1)

def construct_mlp(in_features: int, out_features: int, n_hid: int) -> torch.nn.Module:
    return Sequential(
        Linear(in_features, n_hid),
        ReLU(),
        Linear(n_hid, out_features),
    )


class RGNNLayer(nn.Module):
    """Single RGNN layer with Transformer-based root update"""
    def __init__(self, in_feat_size: int, out_feat_size: int, n_edge_labels: int, aggr: str):
        super(RGNNLayer, self).__init__()
        # 此处的feat_size是单个特征维度
        # out_feat_size是Transformer的隐藏维度也是输出的单个特征维度
        # 共n个特征，则输入 = n*in_feat_size，输出 = n*out_feat_size
        self.root = FeatureAttention(in_feat_size, out_feat_size)

        self.convs = torch.nn.ModuleList()
        for _ in range(n_edge_labels):
            self.convs.append(
                LinearConv(in_feat_size, out_feat_size, aggr=aggr
            ).jittable())  # jittable for acceleration

    def forward(self, x: torch.Tensor, edge_indices: List[torch.Tensor]) -> torch.Tensor:
        """Update node features.
        
        Args:
            x (Tensor): Node features.
            edge_indices (List[Tensor]): List of edge index.
        """
        x_out = self.root(x)  # Transformer-encoded

        for i, conv in enumerate(self.convs):
            x_out += conv(x, edge_indices[i])

        return x_out


class LinearConv(MessagePassing):
    """Linear graph convolution layer.
    For aggregate features from neighboring nodes.
    """
    propagate_type = {"x": torch.Tensor}

    def __init__(self, in_feat_size: int, out_feat_size: int, aggr: str) -> None:
        super().__init__(aggr=aggr)
        self.f = FeatureAttention(feat_size=in_feat_size, hidden_size=out_feat_size)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        return self.propagate(edge_index=edge_index, x=x, size=None)

    
class FeatureAttention(nn.Module):
    """Simple NodeAttention: Just attention model."""
    def __init__(self, feat_size, hidden_size):
        super().__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size

        # Q, K, V projections
        self.qkv_proj = nn.Linear(feat_size, hidden_size * 3, bias=False)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        """
        x: [N, seq_len * feat_dim]
        """
        N, D = x.shape
        assert D % self.feat_size == 0, "Feature dimension not divisible by feat_size"
        seq_len = D // self.feat_size

        # reshape to sequence
        x = x.view(N, seq_len, self.feat_size)

        qkv = self.qkv_proj(x)   # [N, seq_len, 3*H]
        Q, K, V = qkv.chunk(3, dim=-1)  

        att = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_size)   
        att = torch.softmax(att, dim=-1)

        # attention output
        x = torch.matmul(att, V)      

        # FFN residual block
        x = x + self.ffn(x)

        return x.reshape(N, -1)

class FeatureLinear(nn.Module):
    def __init__(self, feat_size, nhid):
        super().__init__()
        self.feat_trans = nn.Linear(feat_size, nhid, bias=False)
        self.feat_size = feat_size
        self.nhid = nhid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        assert D % self.feat_size == 0, "Feature dimension not divisible by feat_size"
        seq_len = D // self.feat_size
        x = x.view(N, seq_len, self.feat_size)
        x = self.feat_trans(x)
        x = x.reshape(N, -1)
        return x

class RGNN(nn.Module):
    """RGNN model, for graph encoding, processing and decoding to heuristic value.
    The class can be compiled with jit or the new pytorch-2. However, pytorch-geometric
    has yet to provide compiling for GNNs with variable sized graph inputs.
    """

    def __init__(self, params) -> None:
        super().__init__()
        # RGNN params
        self.in_feat = params["in_feat"]                # Input feature dimension, depends on representation
        self.out_feat = params["out_feat"]              # Output feature dimension, usually 1 (heuristic value)
        self.nhid = params["nhid"]                      # Hidden layer dimension (this code for Transformer)
        self.aggr = params["aggr"]                      # aggregate: max, mean, sum
        self.n_edge_labels = params["n_edge_labels"]    # Num of edge labels
        self.nlayers = params["nlayers"]                # Num of RGNN Layers (L)
        self.rep_type = params["rep"]                   # Rep string
        self.rep = None                                 # Problem's representation          
        self.device = None
        self.batch = False

        # self.feat_size = 6 # TODO now just for test, if needed, change train_gnn.py
        self.feat_size_list = params["feat_size_list"]
        self.feat_size = max(self.feat_size_list)

        # global pooling method
        if params["pool"] == "max":
            self.pool = global_max_pool
        elif params["pool"] == "mean":
            self.pool = global_mean_pool
        elif params["pool"] == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError

        self.initialise_layers()    # Initialize layers which are in RGNN.

        return

    @abstractmethod
    def create_layer(self) -> None:
        raise NotImplementedError

    def initialise_layers(self) -> None:
        """RGNN layers
        emb:    Encoder, input_frature -> hidden_dim
        layers: RGNN message passing layers
        mlp_h:  Heuristic decoder, hiden_dim -> output_dim
        """
        self.pos_emb = self._build_pos_enc(max_len=10, dim=self.feat_size)
        self.emb = FeatureLinear(feat_size=self.feat_size, nhid=self.nhid)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(self.create_layer())

        self.final_size = len(self.feat_size_list) * self.nhid
        self.mlp_h = construct_mlp(in_features=self.final_size, n_hid=self.final_size, out_features=self.out_feat)
        return

    def _build_pos_enc(self, max_len, dim):
        """sin cos 位置编码"""
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, dim]
    
    def create_layer(self):
        return RGNNLayer(
            in_feat_size=self.nhid, 
            out_feat_size=self.nhid, 
            n_edge_labels=self.n_edge_labels, 
            aggr=self.aggr
        )
    
    def padding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """特征补零"""
        assert sum(self.feat_size_list) == self.in_feat, "Feature size list does not match input feature size"
        feat_list = torch.split(x, self.feat_size_list, dim=1)
        padded = [torch.nn.functional.pad(f, (0, self.feat_size - f.size(1))) for f in feat_list]
        x = torch.cat(padded, dim=1)
        return x

    def pos_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        # add positional encoding
        N, D = x.shape
        assert D % self.feat_size == 0, "Feature dimension not divisible by feat_size"
        seq_len = D // self.feat_size
        x = x.view(N, seq_len, self.feat_size)
        x = x + self.pos_emb[:, :seq_len, :].to(x.device)
        x = x.reshape(N, -1)
        return x
        
    def node_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        x = self.padding(x, list_of_edge_index, batch)
        x = self.pos_embedding(x, list_of_edge_index, batch)
        # Encode and update node features.
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, list_of_edge_index)
            x = F.relu(x)
        return x

    def graph_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        # Encode and update node features, then globally pool to single graph feature.
        x = self.node_embedding(x, list_of_edge_index, batch)
        x = self.pool(x, batch)  # dim: nhid
        return x

    def forward(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        # encode -> update -> pool -> decode
        x = self.graph_embedding(x, list_of_edge_index, batch) 
        h = self.mlp_h(x)
        h = h.squeeze(1)    # delete extra dim
        return h

    def name(self) -> str:
        return type(self).__name__