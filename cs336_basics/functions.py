import torch
from einops import einsum
import numpy as np
import logging

logger = logging.getLogger(__name__)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_adjusted = x - x_max
    x_exp = torch.exp(x_adjusted)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    #  Q (Float[Tensor, " ... n d_k"]): Query tensor
    #  K (Float[Tensor, " ... m d_k"]): Key tensor
    #  V (Float[Tensor, " ... m d_v"]): Values tensor
    #  mask (Bool[Tensor, " ... n m"] | None): Mask tensor
    dk = query.shape[-1]
    score = einsum(query, key, "... n d_k, ... m d_k -> ... n m") / np.sqrt(dk)
    mask_val = torch.zeros(mask.shape, dtype=torch.float)
    mask_val.masked_fill_(~mask, float("-inf"))
    score = softmax(score + mask_val, dim=-1)
    attention = einsum(score, value, "... n m, ... m d_v -> ... n d_v")
    return attention
