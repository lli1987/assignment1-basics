from torch.optim import Optimizer
from typing import Optional, Callable
import math
import torch


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
