# https://github.com/gastruc/osv5m/blob/main/models/losses.py
import torch
from torch import nn, Tensor

class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor Bx2
            y: torch.Tensor Bx2
        Returns:
            torch.Tensor: Haversine loss between x and y: torch.Tensor([B])
        Note:
            Haversine distance doesn't contain the 2 * 6371 constant.
        """
        lhs = torch.sin((x[:, 0] - y[:, 0]) / 2) ** 2
        rhs = (
            torch.cos(x[:, 0])
            * torch.cos(y[:, 0])
            * torch.sin((x[:, 1] - y[:, 1]) / 2) ** 2
        )
        a = lhs + rhs
        return torch.arctan2(torch.sqrt(a), torch.sqrt(1 - a))