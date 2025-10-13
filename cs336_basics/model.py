import torch
from torch import nn
from cs336_basics.layers import (
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
    RotaryPositionalEmbedding,
)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, theta)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor):
        x_orig = x
        x = self.rms_norm1.forward(x)
        token_positions = torch.arange(0, x.shape[-2], 1)
        x = x_orig + self.mha.forward(x, token_positions)
        x_orig = x
        x = self.rms_norm2.forward(x)
        return x_orig + self.ffn.forward(x)
