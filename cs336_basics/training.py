from torch.optim import Optimizer
from typing import Optional, Callable
import math
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta": betas, "epsilon": eps, "lambda": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            epsilon = group["epsilon"]
            h_lambda = group["lambda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = beta[0] * m + (1 - beta[0]) * grad
                v = beta[1] * v + (1 - beta[1]) * torch.pow(grad, 2)
                lr_adjusted = (
                    lr
                    * math.sqrt(1 - math.pow(beta[1], t))
                    / (1 - math.pow(beta[0], t))
                )

                p.data -= lr_adjusted * m / (torch.sqrt(v) + epsilon)
                p.data -= lr * h_lambda * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v


def train(
    training_file,
    vocab_size,
    special_tokens,
    context_length,
    batch_size,
    device,
    num_layers,
    num_heads,
    d_model,
    d_ff,
    theta,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    iterations=1000,
):
    from model import LLM
    from functions import cross_entropy, data_loading
    from bpe_training import train_bpe
    from bpe_encoding import Tokenizer

    vocab, merges = train_bpe(
        input_path=training_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    with open(training_file) as f:
        contents = f.read()
    ids = tokenizer.encode(contents)
    ids = np.array(ids)
    x1, x2 = data_loading(
        x=ids, batch_size=batch_size, context_length=context_length, device=device
    )

    model = LLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        theta=theta,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    for _ in range(iterations):
        optimizer.zero_grad()
        o = model.forward(x=x1)
        loss = cross_entropy(o, x2)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train(
        training_file="/Users/luyaoli/code/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        context_length=48,
        batch_size=5,
        device="cpu",
        num_layers=5,
        num_heads=3,
        d_model=12,
        d_ff=20,
        theta=10000,
        iterations=12000
    )
