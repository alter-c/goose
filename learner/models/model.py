import numpy as np
import torch
import torch.nn as nn
import time
import os
import random
from planning import Proposition, State
from representation import REPRESENTATIONS, Representation
from typing import List
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from models.module import RGNN, construct_mlp
# from models.rgcn import RGNN, construct_mlp


torch.set_num_threads(1)

class Model(nn.Module):
    """A wrapper for a GNN which contains the GNN, additional informations beyond hyperparameters,
    and helpful methods such as I/O and providing an interface for planners to call as a heuristic
    evaluator.
    """

    def __init__(self, params=None, jit=False) -> None:
        super().__init__()
        if params is not None:
            self.model = None
            self.jit = jit
            self.rep_type = params["rep"]
            self.rep = None
            self.device = None
            self.batch = False
            self.create_model(params)   # Create RGNN model.
        if self.jit:
            self.model = torch.jit.script(self.model)   # Compile with jit, accelerate inference.
        return
    
    def set_eval(self) -> None:
        self.model.eval()
        return

    def lifted_state_input(self) -> bool:
        return self.rep.lifted

    def dump_model_stats(self) -> None:
        print(f"Model name: RGNN")
        print(f"Device:", self.device)
        print(f"Number of parameters:", self.get_num_parameters())
        print(f"Number of layers:", self.model.nlayers)
        print(f"Number of hidden units:", self.model.nhid)
        return

    def load_state_dict_into_gnn(self, model_state_dict) -> None:
        """Load saved weights"""
        self.model.load_state_dict(model_state_dict)

    def forward(self, data):
        return self.model.forward(data.x, data.edge_index, data.batch)

    def embeddings(self, data):
        return self.model.graph_embedding(data.x, data.edge_index, data.batch) 

    def forward_from_embeddings(self, embeddings):
        x = self.model.mlp_h(embeddings)
        x = x.squeeze(1)  # delete extra dim
        return x

    def initialise_readout(self):
        if self.jit:
            self.model.mlp = torch.jit.script(
                construct_mlp(
                    in_features=self.model.nhid,
                    n_hid=self.model.nhid,
                    out_features=self.model.out_feat,
                )
            )
        else:
            self.model.mlp = construct_mlp(
                in_features=self.model.nhid,
                n_hid=self.model.nhid,
                out_features=self.model.out_feat,
            )
        return

    def update_representation(self, domain_pddl: str, problem_pddl: str, args, device):
        self.rep: Representation = REPRESENTATIONS[self.rep_type](domain_pddl, problem_pddl)
        self.rep.convert_to_pyg()
        self.device = device
        return

    def update_device(self, device):
        self.device = device
        return

    def batch_search(self, batch: bool):
        self.batch = batch
        return

    def print_weights(self) -> None:
        weights = self.state_dict()
        for weight_group in weights:
            print(weight_group)
            print(weights[weight_group])
        return

    def get_num_parameters(self) -> int:
        """Count number of weight parameters"""
        # https://stackoverflow.com/a/62764464/13531424
        # e.g. to deal with case of sharing layers
        params = sum(
            dict((p.data_ptr(), p.numel()) for p in self.parameters() if p.requires_grad).values()
        )
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return params

    def get_num_zero_parameters(self) -> int:
        """Count number of parameters that are zero after training"""
        zero_weights = 0
        for p in self.parameters():
            if p.requires_grad:
                zero_weights += torch.sum(torch.isclose(p.data, torch.zeros_like(p.data)))
        return zero_weights

    def print_num_parameters(self) -> None:
        print(f"number of parameters: {self.get_num_parameters()}")
        return

    def set_zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def create_model(self, params):
        self.model = RGNN(params)

    def h(self, state: State) -> float:
        """Compute heuristic value for single state."""
        # state -> graph rep -> tensor -> h
        with torch.no_grad():
            x, edge_index = self.rep.state_to_tensor(state)
            x = x.to(self.device)
            for i in range(len(edge_index)):
                edge_index[i] = edge_index[i].to(self.device)
            h = self.model.forward(x, edge_index, None)
            h = round(h.item())  # h should be integer
            
            return h

    def h_batch(self, states: List[State]) -> List[float]:
        """Parallelly compute heuristic values for batch states."""
        with torch.no_grad():
            # Trans data to batch.
            data_list = []
            for state in states:
                x, edge_index = self.rep.state_to_tensor(state)
                data_list.append(Data(x=x, edge_index=edge_index))
            loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 64))
            # Parallelly compute h values.
            hs_all = []
            for data in loader:
                data = data.to(self.device)
                hs = self.model.forward(data.x, data.edge_index, data.batch)
                hs = hs.detach().cpu().numpy()  # annoying error with jit
                hs_all.append(hs)
            hs_all = np.concatenate(hs_all)
            hs_all = np.rint(hs_all)
            hs_all = hs_all.astype(int).tolist()

            return hs_all

    def __call__(self, node_or_list_nodes):
        """For Pyperplan search."""
        if self.batch:
            states = [n.state for n in node_or_list_nodes]
            h = self.h_batch(states)  # list of states
        else:
            state = node_or_list_nodes.state
            h = self.h(state)  # single state
        return h

    def name(self) -> str:
        return self.model.name()
    
