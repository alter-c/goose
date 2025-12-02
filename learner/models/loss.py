import torch
import torch.nn.functional as F
from torch import Tensor


class MSELoss:
    def __init__(self,) -> None:
        pass

    def forward(self, h_input: Tensor, h_target: Tensor):
        h_loss = F.mse_loss(h_input, h_target)
        return h_loss

class BCELoss:
    def __init__(self,) -> None:
        pass

    def forward(self, h_input: Tensor, h_target: Tensor):
        h_loss = F.binary_cross_entropy_with_logits(h_input, h_target)
        return h_loss