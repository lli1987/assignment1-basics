import regex as re
import os
from collections import defaultdict
from typing import BinaryIO
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        g_pre_tokens_count: dict[tuple[bytes], int] = {}
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        # build pre-tokens count map concurrently
        with Pool(num_processes) as p:
            pre_tokens_count_list = [
                p.apply_async(
                    remove_special_tokens_and_pre_tokenize, (chunk, special_tokens)
                )
                for chunk in chunks
            ]
            [
                merge_pre_tokens_count(g_pre_tokens_count, pre_tokens_count.get())
                for pre_tokens_count in pre_tokens_count_list
            ]
        return merge(g_pre_tokens_count, vocab_size, special_tokens)


def pre_tokenize_doc(doc) -> dict[tuple[bytes], int]:
    pre_tokens_count: dict[tuple[bytes], int] = defaultdict(int)

    # counts track pre-token occurrence, key is bytes tuple
    for pre_token_pat in re.finditer(PAT, doc):
        pre_token_str = pre_token_pat.group()
        pre_token = pre_token_str.encode("utf-8")
        pre_token = [bytes([b]) for b in pre_token]
        key = tuple(pre_token)
        if key in pre_tokens_count:
            pre_tokens_count[key] += 1
        else:
            pre_tokens_count[key] = 1
    return pre_tokens_count


def merge(
    pre_tokens_count: dict[tuple[bytes], int], vocab_size, special_tokens
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {}
    idx = 0
    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1
    for i in range(256):
        vocab[idx + i] = bytes([i])

    # counts track pre-token occurrence, key is bytes tuple, each element
    # starts from one byte, then gets merged
    counts: dict[tuple[bytes], int] = dict()
    for _ in range(vocab_size - 256 - len(special_tokens)):
        for pre_token, cnt in pre_tokens_count.items():
            # count pair occurrence across all pre-tokens
            for t1, t2 in zip(pre_token, pre_token[1:]):
                if (t1, t2) in counts:
                    counts[(t1, t2)] += cnt
                else:
                    counts[(t1, t2)] = cnt
        # if there is at least one pair
        if len(counts) > 0:
            # find the most occurred pair, if tie, pick lexicographically largest
            pair = max(counts, key=lambda k: (counts[k], k))
            new_index = len(vocab)
            # update vocabuary: new index -> new pair
            vocab[new_index] = pair[0] + pair[1]
            # update merges: new pair -> new index
            merges.append(pair)
            # update pre-token pool
            new_pre_tokens_count: defaultdict[tuple[bytes], int] = defaultdict(int)
            for pre_token, cnt in pre_tokens_count.items():
                new_pre_token = []
                idx = 0
                while idx < len(pre_token):
                    if (
                        idx < len(pre_token) - 1
                        and pre_token[idx] == pair[0]
                        and pre_token[idx + 1] == pair[1]
                    ):
                        new_bytes = b"".join(pair)
                        new_pre_token.append(new_bytes)
                        idx += 2
                    else:
                        new_pre_token.append(pre_token[idx])
                        idx += 1
                new_pre_tokens_count[tuple(new_pre_token)] = cnt
            pre_tokens_count = new_pre_tokens_count
            counts.clear()
    return vocab, merges


def merge_pre_tokens_count(count1, count2):
    for key, c2_value in count2.items():
        if key in count1:
            c1_value = count1[key]
            count1[key] = c1_value + c2_value
        else:
            count1[key] = c2_value


def remove_special_tokens_and_pre_tokenize(
    chunk: str, special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    docs = re.split(
        "|".join([re.escape(special_token) for special_token in special_tokens]), chunk
    )
    g_pre_tokens_count: dict[tuple[bytes], int] = {}
    for doc in docs:
        if not doc:
            continue
        pre_tokens_count = pre_tokenize_doc(doc)
        merge_pre_tokens_count(g_pre_tokens_count, pre_tokens_count)
    return g_pre_tokens_count


if __name__ == "__main__":
    vocab, merges = train_bpe(
        "/Users/luyaoli/code/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",
        10000,
        ["<|endoftext|>"],
    )

    print(vocab)
    print(merges)
