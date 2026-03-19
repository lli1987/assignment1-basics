import torch
from torch import nn, optim
from einops import einsum
import numpy as np
import numpy.typing as npt
import logging
import math
import random
import os

logger = logging.getLogger(__name__)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_adjusted = x - x_max
    x_exp = torch.exp(x_adjusted)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    mask: torch.Tensor,
    device=None,
) -> torch.Tensor:
    #  Q (Float[Tensor, " ... n d_k"]): Query tensor
    #  K (Float[Tensor, " ... m d_k"]): Key tensor
    #  V (Float[Tensor, " ... m d_v"]): Values tensor
    #  mask (Bool[Tensor, " ... n m"] | None): Mask tensor
    dk = query.shape[-1]
    score = einsum(query, key, "... n d_k, ... m d_k -> ... n m") / np.sqrt(dk)
    mask_val = torch.zeros(mask.shape, dtype=torch.float, device=device)
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
    x: torch.Tensor, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # idx = 0
    # output1 = []
    # output2 = []

    # for _ in range(batch_size):
    #     idx = random.randint(0, len(x) - context_length - 1)
    #     output1.append(torch.from_numpy(x[idx : idx + context_length].copy()))
    #     output2.append(torch.from_numpy(x[idx + 1 : idx + context_length + 1].copy()))
    # x1 = torch.cat(output1).reshape(batch_size, context_length).to(device=device)
    # x2 = torch.cat(output2).reshape(batch_size, context_length).to(device=device)

    # return x1, x2

    # Convert once if needed

    # Sample all start indices at once
    max_idx = x.shape[0] - context_length
    idx = torch.randint(0, max_idx, (batch_size,))

    # Create offsets [0, 1, ..., context_length-1]
    offsets = torch.arange(context_length)

    # Shape: (batch_size, context_length)
    x1 = x[idx[:, None] + offsets[None, :]]
    x2 = x[idx[:, None] + offsets[None, :] + 1]

    return x1.to(device), x2.to(device)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    state = {"m_state": model_state, "o_state": optimizer_state, "i": iteration}
    torch.save(state, out)


def load_checkpoint(src, model: nn.Module, optimizer: optim.Optimizer = None):
    state = torch.load(src)
    model.load_state_dict(state.get("m_state"))
    if optimizer:
        optimizer.load_state_dict(state.get("o_state"))
    return state.get("i")


def delete_checkpoint(checkpoint_path):
    try:
        os.remove(checkpoint_path)
        logger.info(f"File '{checkpoint_path}' deleted successfully.")
    except FileNotFoundError:
        logger.error(f"Error: File '{checkpoint_path}' not found.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def checkpoint_exist(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return True
    else:
        return False