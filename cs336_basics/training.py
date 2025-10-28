from torch.optim import Optimizer
from typing import Optional, Callable
import math
import torch
import numpy as np
import logging
import wandb
from config import tinystories_default_config

config = tinystories_default_config

logger = logging.getLogger(__name__)

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="luyaoiosapp-personal",
    # Set the wandb project where this run will be logged.
    project="cs336_llm",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-3,
        "architecture": "LLM",
        "dataset": config["name"],
        "epochs": 2560,
    },
)


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
    model_output,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    iterations=1000,
):
    from model import LLM
    from functions import cross_entropy, data_loading, save_checkpoint
    from bpe_training import train_bpe
    from bpe_encoding import Tokenizer

    vocab, merges = train_bpe(
        input_path=training_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    logger.warning("++++++ BPE training finished ++++++")

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
    for it in range(iterations):
        logger.warning(f"++++ iteration {it} started ++++")
        optimizer.zero_grad()
        o = model.forward(x=x1)
        loss = cross_entropy(o, x2)
        run.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()
        logger.warning(f"---- iteration {it}: loss {loss} ----")
    save_checkpoint(model, optimizer, iterations, model_output)


if __name__ == "__main__":

    train(
        training_file=config["training_file"],
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"],
        context_length=config["context_length"],
        batch_size=config["batch_size"],
        device=config["device"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        theta=config["theta"],
        iterations=config["iterations"],
        model_output=config["model_output"],
    )
