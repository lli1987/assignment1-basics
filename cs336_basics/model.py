import torch
from torch import nn
from cs336_basics.layers import (
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
    Embedding,
    Linear,
)
from cs336_basics.functions import softmax
import logging

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
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
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model)
        self.ln = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding.forward(x)
        for idx in range(self.num_layers):
            x = self.transformer_blocks[idx].forward(x)
        x = self.rms_norm.forward(x)
        return self.ln.forward(x)
