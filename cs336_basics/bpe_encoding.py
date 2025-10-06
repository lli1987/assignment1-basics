import regex as re
from _collections_abc import Iterable, Iterator

import cs336_basics.constants as constants
from cs336_basics.utils import get_pre_token_bytes

# import constants as constants
# from utils import get_pre_token_bytes
import os
import json


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [] if not special_tokens else special_tokens
        self.special_tokens = sorted(self.special_tokens, key=lambda x: (-len(x), x))

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}
        merges = []
        with open(vocab_filepath) as f:
            vocab_str = f.read()
            vocab = json.loads(vocab_str)
        with open(merges_filepath) as f:
            merges_str = f.read()
            merges = json.loads(merges_str)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
        return tokenizer

    def encode(self, text: str) -> list[int]:
        pre_tokens = self._pre_tokenize(text)
        ivocab = self._build_inverted_vocab()
        tokens = []
        for pre_token in pre_tokens:
            token = self._merge(pre_token, ivocab)
            tokens = tokens + token
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens = self.encode(text)
            for token in tokens:
                yield token

    def decode(self, ids: list[int]) -> str:
        code_list = bytes([])
        for id in ids:
            code_bytes = self.vocab[id]
            code_list = code_list + code_bytes
        return code_list.decode("utf-8", errors="replace")

    def _pre_tokenize(self, text):
        docs = [text]
        if len(self.special_tokens) > 0:
            delimiters = "|".join(
                [re.escape(special_token) for special_token in self.special_tokens]
            )

            docs = re.split(
                rf"({delimiters})",
                text,
            )
        pre_tokens = []
        for doc in docs:
            if not doc:
                continue
            if doc in self.special_tokens:
                pre_tokens.append([doc.encode("utf-8")])
                continue
            for pre_token_group in re.finditer(constants.PAT, doc):
                pre_token = get_pre_token_bytes(pre_token_group)
                pre_token = [bytes([b]) for b in pre_token]
                pre_tokens.append(pre_token)
        return pre_tokens

    def _merge(self, pre_token, ivocab):
        token = []
        bytes_list = []
        for b in pre_token:
            bytes_list.append(b)
        new_bytes_list = self._merge_based_on_first_match(bytes_list)
        for bytes in new_bytes_list:
            token.append(ivocab[bytes])
        return token

    def _merge_based_on_first_match(self, bytes_list):
        while True:
            start = -1
            for merge in self.merges:
                idx = 0
                while idx < len(bytes_list):
                    if idx < len(bytes_list) - 1:
                        if merge == (bytes_list[idx], bytes_list[idx + 1]):
                            start = idx
                            break
                    idx += 1
                if start != -1:
                    break
            new_bytes_list = []
            if start != -1:
                for b_id, bs in enumerate(bytes_list):
                    if b_id < start or b_id > start + 1:
                        new_bytes_list.append(bs)
                    elif b_id == start:
                        new_bytes_list.append(bytes_list[start] + bytes_list[start + 1])

                bytes_list = new_bytes_list
                start = -1
            else:
                break
        return bytes_list

    def _build_inverted_vocab(self):
        ivocab: dict[bytes, int] = {}
        for idx, b in self.vocab.items():
            ivocab[b] = idx
        return ivocab


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer


if __name__ == "__main__":
    # from bpe_training import train_bpe

    # vocab, merges = train_bpe(
    #     "/Users/luyaoli/code/cs336/assignment1-basics/cs336_basics/test_file",
    #     10000,
    #     ["<|endoftext|>"],
    # )
    # print(vocab)
    # tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="/Users/luyaoli/code/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json",
        merges_path="/Users/luyaoli/code/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    print(tokenizer.vocab)
    print(tokenizer.merges)
    corpus_path = (
        "/Users/luyaoli/code/cs336/assignment1-basics/tests/fixtures/address.txt"
    )
    with open(corpus_path) as f:
        corpus_contents = f.read()
    ids = tokenizer.encode(corpus_contents)
    # print(ids)
    import tiktoken

    reference_tokenizer = tiktoken.get_encoding("gpt2")
    ref_ids = reference_tokenizer.encode(corpus_contents)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    ref_tokenized_string = [reference_tokenizer.decode([x]) for x in ref_ids]
    ret = tokenizer.decode(ids)
    # print(tokenized_string)
    # print(ret)
