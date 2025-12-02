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
    """Single RGNN layer"""
    def __init__(self, in_features: int, out_features: int, n_edge_labels: int, aggr: str):
        super(RGNNLayer, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(n_edge_labels):  # For each edge type: create a linear convolution layer.
            self.convs.append(LinearConv(in_features, out_features, aggr=aggr).jittable()) # jittable for acceleration
        self.root = Linear(in_features, out_features, bias=True)
        return

    def forward(self, x: Tensor, edge_indices: List[Tensor]) -> Tensor:
        """Update node features.
        
        Args:
            x (Tensor): Node features.
            edge_indices (List[Tensor]): List of edge index.
        """
        x_out = self.root(x)
        for i, conv in enumerate(self.convs):  # bottleneck; difficult to parallelise efficiently
            x_out += conv(x, edge_indices[i])  # According to edge type, aggregate features from neighboring nodes.
        return x_out


class LinearConv(MessagePassing):
    """Linear graph convolution layer.
    For aggregate features from neighboring nodes.
    """
    propagate_type = {"x": Tensor}

    def __init__(self, in_features: int, out_features: int, aggr: str) -> None:
        super().__init__(aggr=aggr)
        self.f = Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.f(x)
        x = self.propagate(edge_index=edge_index, x=x, size=None)  # propagate = message + aggregate + update
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
        self.nhid = params["nhid"]                      # Hidden layer dimension
        self.aggr = params["aggr"]                      # aggregate: max, mean, sum
        self.n_edge_labels = params["n_edge_labels"]    # Num of edge labels
        self.nlayers = params["nlayers"]                # Num of RGNN Layers (L)
        self.rep_type = params["rep"]                   # Rep string
        self.rep = None                                 # Problem's representation          
        self.device = None
        self.batch = False

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
        self.emb = torch.nn.Linear(self.in_feat, self.nhid)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(self.create_layer())
        self.mlp_h = construct_mlp(in_features=self.nhid, n_hid=self.nhid, out_features=self.out_feat)
        return

    def create_layer(self):
        return RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)

    def node_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
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

    
