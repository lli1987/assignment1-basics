import torch
from model import LLM
from bpe_encoding_3 import Tokenizer
from functions import softmax, load_checkpoint
import logging
import os

logger = logging.getLogger(__name__)


def _get_full_path(dir_path, file_name):
    return os.path.join(dir_path, file_name)


def generate_tokens(
    prompt: str,
    max_tokens: int,
    temperature: int,
    top_p: int,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    context_length: int,
) -> torch.Tensor:
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.int).reshape(1, len(ids))
    ids = ids[:, -context_length:]
    p_list = [id for id in ids[0].tolist()]
    text_list = [word for word in tokenizer.decode(p_list)]

    for _ in range(max_tokens):
        ids = ids[:, -context_length:]  # get max of context_length tokens
        v = model.forward(ids)  # calculate logits from LLM
        v = v[:, -1, :]  # get last token only
        p = softmax(v / temperature, -1)
        p_v, p_idx = torch.sort(p, dim=-1, descending=True)  # sort probability

        # calculate the mask to decide which elements should keep
        cumsum = torch.cumsum(p_v, dim=-1)
        mask = cumsum <= top_p
        first_exceed = torch.argmax((cumsum >= top_p).int(), dim=-1, keepdim=True)
        mask.scatter_(dim=-1, index=first_exceed, value=True)

        # set abandoned elements' values to zero
        kept = torch.zeros_like(p)
        kept_vals = mask * p_v
        kept.scatter_(dim=-1, index=p_idx, src=kept_vals)
        kept_sum = kept.sum(dim=-1, keepdim=True)

        # recalculate probability
        kept_p = kept / kept_sum

        # pick the most probable token
        next_id = torch.argmax(kept_p, dim=-1, keepdim=True)
        word = tokenizer.decode([next_id[0][0].item()])
        if word == "<|endoftext|>":
            break
        text_list.append(word)

        # add new token to the token list's end
        ids = torch.cat((ids, next_id), dim=-1)
    logger.warning("".join(text_list))
    return ids


if __name__ == "__main__":
    from config import openwebtext_config

    config = openwebtext_config
    vocab_size = config["vocab_size"]
    tokenizer = Tokenizer.get_tokenizer(config["name"], vocab_size)
    model = LLM(
        vocab_size=vocab_size,
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        theta=config["theta"],
    )
    model_path = _get_full_path(config["output_dir"], config["model_output"])

    load_checkpoint(
        src=model_path,
        model=model,
    )
    generate_tokens(
        prompt="How many people in US?",
        max_tokens=100,
        temperature=1.0,
        top_p=0.95,
        model=model,
        tokenizer=tokenizer,
        context_length=config["context_length"],
    )
