from typing import Optional, Callable
from torch.optim import Optimizer
import torch
import math
from cs336_basics.functions import (
    learning_rate_schedule,
)


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        lr_scheduler=None,
        warmup_iters=500,
        cosine_cycle_iters=10000,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta": betas, "epsilon": eps, "lambda": weight_decay}
        super().__init__(params, defaults)
        self.lr_scheduler = lr_scheduler
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

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

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0

                m = state.get("m")
                v = state.get("v")
                t = state.get("t") + 1
                state["t"] = t

                if self.lr_scheduler:
                    max_learning_rate = lr
                    min_learning_rate = max_learning_rate * 0.1
                    lr = self.lr_scheduler(
                        t,
                        max_learning_rate,
                        min_learning_rate,
                        self.warmup_iters,
                        self.cosine_cycle_iters,
                    )
                # run.log({"lr": lr})
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m.mul_(beta[0]).add_(grad, alpha=1 - beta[0])
                v.mul_(beta[1]).addcmul_(grad, grad, value=1 - beta[1])

                lr_adjusted = lr * (1 - beta[1] ** t) ** 0.5 / (1 - beta[0] ** t)

                p.data -= lr_adjusted * m / (v.sqrt() + epsilon)
                p.data -= lr * h_lambda * p.data
