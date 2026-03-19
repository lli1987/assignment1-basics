from torch import nn
import torch
from einops import einsum
import numpy as np

from cs336_basics.functions import scaled_dot_product_attention
import logging

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weights = self._init_weights(in_features, out_features, device)

    def _init_weights(self, in_features, out_features, device):
        std = np.sqrt(2 / (in_features + out_features))
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device),
                mean=0,
                std=std,
                a=-2 * std,
                b=2 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weights = self._init_weights(num_embeddings, embedding_dim, device)

    def _init_weights(self, num_embeddings, embedding_dim, device):
        std = 0.02
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device),
                mean=0,
                std=std,
                a=-2 * std,
                b=2 * std,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(nn.init.ones_(torch.empty(d_model, device=device)))
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
    def __init__(self, d_model, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = int((8 / 3 * d_model // 64) * 64)
        self.weights1 = self._init_weights(d_model, self.d_ff, device)
        self.weights2 = self._init_weights(self.d_ff, d_model, device)
        self.weights3 = self._init_weights(d_model, self.d_ff, device)

    def _init_weights(self, d_in, d_out, device):
        std = np.sqrt(2 / (d_in + d_out))
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device),
                mean=0,
                std=std,
                a=-2 * std,
                b=2 * std,
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


class SiLU(nn.Module):
    def __init__(self, d_model, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model
        self.weights1 = self._init_weights(d_model, self.d_ff, device)
        self.weights2 = self._init_weights(self.d_ff, d_model, device)

    def _init_weights(self, d_in, d_out, device):
        std = np.sqrt(2 / (d_in + d_out))
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device),
                mean=0,
                std=std,
                a=-2 * std,
                b=2 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = [... d_model]
        # w1 = [d_ff d_model]
        # w2 = [d_model d_ff]
        w1x = einsum(self.weights1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = torch.sigmoid(w1x) * w1x
        return einsum(self.weights2, silu, "d_model d_ff, ... d_ff -> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer(
            "invert_freq", self._gen_invert_freq(device), persistent=False
        )

    def _gen_invert_freq(self, device):
        half = self.d_k // 2
        # pos = torch.arange(0, self.max_seq_len).float()
        k = torch.arange(1, half + 1, device=device)
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


class MultiHeadSelfAttention(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens
    """

    def __init__(
        self, d_model: int, num_heads: int, theta: float = None, device=None
    ) -> torch.Tensor:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.device = device
        self.q_proj_weight = self._init_weights(d_model, d_model, device)
        self.k_proj_weight = self._init_weights(d_model, d_model, device)
        self.v_proj_weight = self._init_weights(d_model, d_model, device)
        self.o_proj_weight = self._init_weights(d_model, d_model, device)

    def _init_weights(self, d_in, d_out, device):
        std = np.sqrt(2 / (d_in + d_out))
        return nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device),
                mean=0,
                std=std,
                a=-2 * std,
                b=2 * std,
            )
        )

    def reshape_expand_contract(self, t: torch.Tensor):
        batch, seq_len, d_k = t.shape
        hd_k = d_k // self.num_heads
        return t.view(batch, seq_len, self.num_heads, hd_k).transpose(1, 2)

    def forward(
        self,
        in_features: torch.Tensor,
        token_positions: torch.Tensor = None,
        max_seq_len: int = -1,
    ):
        q = einsum(
            in_features,
            self.q_proj_weight,
            "... sequence_length d_in, d_k d_in -> ... sequence_length d_k",
        )
        k = einsum(
            in_features,
            self.k_proj_weight,
            "... sequence_length d_in, d_k d_in -> ... sequence_length d_k",
        )

        v = einsum(
            in_features,
            self.v_proj_weight,
            "... sequence_length d_in, d_k d_in -> ... sequence_length d_k",
        )

        qh = self.reshape_expand_contract(q)
        kh = self.reshape_expand_contract(k)
        vh = self.reshape_expand_contract(v)

        if self.theta:
            rope = RotaryPositionalEmbedding(
                self.theta,
                self.d_model // self.num_heads,
                max_seq_len,
                device=self.device,
            )
            qh = rope.forward(qh, token_positions)
            kh = rope.forward(kh, token_positions)

        mask_shape = qh.shape[:-1]
        mask_shape = mask_shape + (mask_shape[-1],)

        mask = torch.tril(
            torch.ones(mask_shape, dtype=torch.bool, device=self.device), diagonal=0
        )
        attn = scaled_dot_product_attention(kh, vh, qh, mask, self.device)
        attn = attn.transpose(1, 2)
        batch, seq_len, heads, d_k = attn.shape
        attn = attn.contiguous().view(batch, seq_len, heads * d_k)

        multiheads_attention = einsum(
            attn,
            self.o_proj_weight,
            "... sequence_length dv, d_model dv  -> ... sequence_length d_model",
        )
        return multiheads_attention
