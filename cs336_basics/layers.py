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


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weights = self._init_weights(num_embeddings, embedding_dim)

    def _init_weights(self, num_embeddings, embedding_dim):
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim), mean=0, std=1, a=-3, b=-3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(nn.init.ones_(torch.empty(d_model)))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = torch.sqrt((x**2).sum(dim=-1, keepdim=True) / self.d_model + self.eps)
        result = x / rms_a * self.weights
        return result.to(in_dtype)


'''
   Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight '''


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_ff = int((8 / 3 * d_model // 64) * 64)
        self.weights1 = self._init_weights(d_model, self.d_ff)
        self.weights2 = self._init_weights(self.d_ff, d_model)
        self.weights3 = self._init_weights(d_model, self.d_ff)

    def _init_weights(self, d_in, d_out):
        std = np.sqrt(2 / (d_in + d_out))
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = [... d_model]
        # w1 = [d_ff d_model]
        # w2 = [d_model d_ff]
        # w3 = [d_ff d_model]
        w1x = einsum(self.weights1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = torch.sigmoid(w1x) * w1x
        w3x = einsum(self.weights3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(
            self.weights2, silu * w3x, "d_model d_ff, ... d_ff -> ... d_model"
        )


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer("invert_freq", self._gen_invert_freq(), persistent=False)

    def _gen_invert_freq(self):
        half = self.d_k // 2
        # pos = torch.arange(0, self.max_seq_len).float()
        k = torch.arange(1, half + 1)
        invert_freq = 1.0 / (self.theta ** ((2 * k - 2) / self.d_k))  # [d_k // 2]
        return invert_freq

    def _gen_cos_sin(self, token_positions):
        invert_freq = self.invert_freq
        angles = einsum(
            token_positions,
            invert_freq,
            "... max_seq_len, half -> ... max_seq_len half",
        )

        cos, sin = torch.cos(angles), torch.sin(angles)
        return cos, sin

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x1, x2 = [... max_seq_len, d_k // 2]
        cos, sin = self._gen_cos_sin(token_positions)  # [..., max_seq_len, d_k // 2]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x1_rot
        x_rotated[..., 1::2] = x2_rot
        return x_rotated
