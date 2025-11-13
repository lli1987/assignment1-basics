import torch
from torch import nn
from cs336_basics.layers import (
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
    Embedding,
    Linear,
)
import logging

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        self.max_seq_len = max_seq_len
        self.rms_norm1 = RMSNorm(d_model, device=device)
        self.rms_norm2 = RMSNorm(d_model, device=device)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, theta, device=device)
        self.ffn = SwiGLU(d_model, device=device)

    def forward(self, x: torch.Tensor):
        x_orig = x
        x = self.rms_norm1.forward(x)
        token_positions = torch.arange(0, x.shape[-2], 1, device=self.device)
        x = x_orig + self.mha.forward(x, token_positions)
        x_orig = x
        x = self.rms_norm2.forward(x)
        return x_orig + self.ffn.forward(x)

        # post norm
        # token_positions = torch.arange(0, x.shape[-2], 1, device=self.device)
        # x = x + self.mha.forward(x, token_positions)
        # x = self.rms_norm1.forward(x)
        # return self.rms_norm2.forward(x + self.ffn.forward(x))


class LLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        theta: int,
        device=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, theta, context_length, device=device
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model, device=device)
        self.ln = Linear(d_model, vocab_size, device=device)

    def forward(self, x: torch.Tensor):
        x = self.embedding.forward(x)
        for idx in range(self.num_layers):
            x = self.transformer_blocks[idx].forward(x)
        x = self.rms_norm.forward(x)
        return self.ln.forward(x)
