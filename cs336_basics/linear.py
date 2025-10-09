from torch import nn
import torch
from einops import einsum
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weights = self._init_weights(in_features, out_features)

    def _init_weights(self, in_features, out_features):
        std = np.sqrt(2 / (in_features + out_features))
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
