import torch
import numpy as np
import logging

import wandb
import regex as re
from cs336_basics.model import LLM
from cs336_basics.functions import cross_entropy, data_loading, save_checkpoint
from cs336_basics.bpe_encoding_3 import Tokenizer
# from cs336_basics.optimizers import AdamW
from torch.optim import AdamW

from cs336_basics.config import openwebtext_tiktoken_config


from cs336_basics.utils import find_chunk_boundaries

from cs336_basics.functions import (
    gradient_clipping,
    checkpoint_exist,
    delete_checkpoint,
    load_checkpoint,
)

# from config import (
#     tinystories_post_norm_config,
#     tinystories_default_config,
#     tinystories_layer_ablation_config,
#     tinystories_no_rope_config,
#     tinystories_silu_config,
#     openwebtext_config,
# )
# from utils import find_chunk_boundaries
# from functions import (
#     learning_rate_schedule,
#     gradient_clipping,
#     checkpoint_exist,
#     delete_checkpoint,
#     load_checkpoint,
# )


config = openwebtext_tiktoken_config

logger = logging.getLogger(__name__)


def train(
    training_files,
    context_length,
    batch_size,
    device,
    num_layers,
    num_heads,
    d_model,
    d_ff,
    theta,
    model_output,
    memmap_output,
    checkpoint_path,
    checkpoint_freq,
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    iterations=1000,
):
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="luyaoiosapp-personal",
        # Set the wandb project where this run will be logged.
        project="cs336_llm_tiktoken",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 1e-3,
            "architecture": "LLM",
            "dataset": config["name"],
            "epochs": 2560,
        },
    )

    n_tokens_estimate = 10_000_000_000
    tokenizer = Tokenizer.get_tokenizer(vocab_size)
    ids = np.memmap(memmap_output, dtype=int, mode="w+", shape=(n_tokens_estimate,))
    idx = 0

    def remove_special_tokens(chunk):
        docs = re.split(
            "|".join([re.escape(special_token) for special_token in ["<|endoftext|>"]]),
            chunk,
        )
        for doc in docs:
            nonlocal idx
            out = []
            for id in tokenizer.encode(doc):
                out.append(id)
            n = len(out)
            ids[idx : idx + n] = out
            idx += n

    for training_file in training_files:
        with open(training_file, "rb") as f:
            num_processes = 12
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)

            for c_id, chunk in enumerate(chunks):
                logger.warning(f"processing chunk: {c_id}")
                remove_special_tokens(chunk)

            logger.warning(f"finished processing {training_file}")
    ids.flush()
    del ids

    ids = np.memmap(memmap_output, dtype=int, mode="r")[:idx]

    logger.warning(f"Number of tokens: {len(ids)}")

    model = LLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        theta=theta,
        device=device,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    it = 0
    if checkpoint_exist(checkpoint_path):
        it = load_checkpoint(checkpoint_path, model, optimizer)
    x = torch.from_numpy(ids.copy()).to(device)
    import time
    params = list(model.parameters())
    while it < iterations:
        logger.warning(f"++++ iteration {it} started ++++")
        
        x1, x2 = data_loading(
            x=x, batch_size=batch_size, context_length=context_length, device=device
        )

        # torch.empty((), device="mps").cpu()
        # t0 = time.perf_counter()    

        optimizer.zero_grad()
        o = model.forward(x=x1)
        loss = cross_entropy(o, x2)
        run.log({"loss": loss.item()})
        loss.backward()
        
        gradient_clipping(params, 1.0)
        optimizer.step()
        logger.warning(f"---- iteration {it}: loss {loss.item()} ----")
        
        # torch.empty((), device="mps").cpu()
        # t1 = time.perf_counter()
        # logger.warning(f"step time: {t1 - t0:.4f}s")

        if it % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, it, checkpoint_path)
        it += 1
    save_checkpoint(model, optimizer, iterations, model_output)


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.warning("using Apple GPU")
    else:
        device = torch.device("cpu")
        logger.warning("using Apple CPU")
    # pre_process(
    #     "/Users/luyaoli/code/cs336/assignment1-basics/openwebtext_extracted/openwebtext",
    #     "/Users/luyaoli/code/cs336/assignment1-basics/openwebtext_processed",
    # )

    # pre_process2(
    #     "/Users/luyaoli/code/cs336/assignment1-basics/openwebtext",
    #     "/Users/luyaoli/code/cs336/assignment1-basics/owt_text",
    # )

    files = []
    input_file_or_dir = config.get("training_file")
    if config.get("is_input_dir", False):
        import os

        fn_list = os.listdir(input_file_or_dir)
        for fn in fn_list:
            full_path = input_file_or_dir + "/" + fn
            files.append(full_path)
    else:
        files = [input_file_or_dir]
    train(
        training_files=files,
        context_length=config["context_length"],
        batch_size=config["batch_size"],
        device=torch.device("mps"),
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        theta=config["theta"],
        iterations=config["iterations"],
        model_output=config["model_output"],
        memmap_output=config["memmap_output"],
        checkpoint_path=config["checkpoint_path"],
        checkpoint_freq=config["checkpoint_freq"],
    )
