import torch
from torch import nn, optim
from einops import einsum
import numpy as np
import numpy.typing as npt
import logging
import math
import random

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


def cross_entropy(o: torch.Tensor, t: torch.Tensor):
    # p's shape: [... seq_len vocab_size]
    #  p = -torch.log(softmax(o, dim=-1))
    # logger.warning(f"---------- {o.shape} -> {t.shape}")
    o_max = o.max(dim=-1, keepdim=True).values
    o_adjusted = o - o_max
    p = -(o_adjusted - torch.log(torch.exp(o_adjusted).sum(dim=-1, keepdim=True)))
    pxi = p.gather(dim=-1, index=t.unsqueeze(dim=-1)).squeeze(-1)
    return pxi.mean()


def learning_rate_schedule(t: int, a_max: float, a_min: float, t_w: int, t_c: int):
    if t < t_w:
        return t / t_w * a_max
    elif t <= t_c and t >= t_w:
        return a_min + 0.5 * (1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) * (
            a_max - a_min
        )
    else:
        return a_min


def gradient_clipping(params: list[torch.nn.Parameter], m: float):
    l2_norm = 0.0
    for param in params:
        if param.grad is None:
            continue
        l2_norm += param.grad.data.pow(2).sum().item()
    l2_norm = math.sqrt(l2_norm)

    for param in params:
        if param.grad is None:
            continue
        if l2_norm > m:
            factor = m / (l2_norm + 1e-6)
            param.grad.data.mul_(factor)


def data_loading(
    x: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = 0
    output1 = []
    output2 = []

    used = set()

    while len(used) < batch_size:
        idx = random.randint(0, len(x) - context_length - 1)
        if idx in used:
            continue
        output1.append(torch.from_numpy(x[idx : idx + context_length]))
        output2.append(torch.from_numpy(x[idx + 1 : idx + context_length + 1]))
        used.add(idx)
    x1 = torch.cat(output1).reshape(batch_size, context_length)
    x2 = torch.cat(output2).reshape(batch_size, context_length)
    x1.to(device)
    x2.to(device)
    return x1, x2


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    state = {"m_state": model_state, "o_state": optimizer_state, "i": iteration}
    torch.save(state, out)


def load_checkpoint(src, model: nn.Module, optimizer: optim.Optimizer):
    state = torch.load(src)
    model.load_state_dict(state.get("m_state"))
    optimizer.load_state_dict(state.get("o_state"))
    return state.get("i")
