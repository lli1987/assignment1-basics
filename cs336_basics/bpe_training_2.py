import regex as re
from collections import defaultdict
from multiprocessing import Pool

from cs336_basics.utils import (
    find_chunk_boundaries,
    get_pre_token_bytes,
)
import cs336_basics.constants as constants

# from utils import (
#     find_chunk_boundaries,
#     get_pre_token_bytes,
# )
# import constants as constants
import logging
import heapq

logger = logging.getLogger(__name__)

# from utils import (
#     find_chunk_boundaries,
#     get_pre_token_bytes,
# )
# import constants as constants

SERIALIZE_FILE_PATH = "/Users/luyaoli/code/cs336/assignment1-basics/cs336_basics/tmp"
G_PRE_TOKEN_COUNT_PKL = "test_g_pre_tokens_count.pkl"
VOCAB_PKL = "test_vocab.pkl"
MERGES_PKL = "test_merges.pkl"


class PairKey:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        # compare first element

        for key in [0, 1]:
            min_length = min(len(self.pair[key]), len(other.pair[key]))
            for idx in range(min_length):
                if self.pair[key][idx] < other.pair[key][idx]:
                    return False
                elif self.pair[key][idx] > other.pair[key][idx]:
                    return True
            if len(self.pair[key]) > len(other.pair[key]):
                return True
            elif len(self.pair[key]) < len(other.pair[key]):
                return False
        return True


def prefix_with_name(name, pkl):
    return f"{name}_{pkl}"


def serialize_bpe(obj, file):
    import pickle

    with open(SERIALIZE_FILE_PATH + "/" + file, "wb") as f:
        pickle.dump(obj, f)


def deserialize_bpe(file):
    import pickle

    try:
        with open(SERIALIZE_FILE_PATH + "/" + file, "rb") as f:
            loaded_obj = pickle.load(f)
        return loaded_obj
    except Exception as e:
        logger.warning(f"cannot deserialize: {e}")
        return None


def func1(chunk):
    return dict()


def train_bpe(
    name,
    input_paths: list[str],
    vocab_size: int,
    special_tokens: list[str],
    enable_cache=False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    g_pre_tokens_count: dict[tuple[bytes], int] = {}
    loaded_g_pre_tokens_count = None

    if enable_cache:
        loaded_vocab = deserialize_bpe(prefix_with_name(name, VOCAB_PKL))
        loaded_merges = deserialize_bpe(prefix_with_name(name, MERGES_PKL))

        if loaded_vocab and loaded_merges:
            return loaded_vocab, loaded_merges
        loaded_g_pre_tokens_count = deserialize_bpe(
            prefix_with_name(name, G_PRE_TOKEN_COUNT_PKL)
        )

    if not loaded_g_pre_tokens_count:
        idx = -1
        for input_path in input_paths:
            idx += 1
            with open(input_path, "rb") as f:
                num_processes = 2
                boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

                chunks = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    chunks.append(chunk)
                # build pre-tokens count map concurrently
                with Pool(num_processes) as p:
                    pre_tokens_count_list = [
                        p.apply_async(
                            remove_special_tokens_and_pre_tokenize,
                            (chunk, special_tokens),
                        )
                        for chunk in chunks
                    ]
                    [
                        merge_pre_tokens_count(
                            g_pre_tokens_count, pre_tokens_count.get()
                        )
                        for pre_tokens_count in pre_tokens_count_list
                    ]
            logger.warning(f"Finished processing file {idx}: {input_path}")
        serialize_bpe(g_pre_tokens_count, prefix_with_name(name, G_PRE_TOKEN_COUNT_PKL))
    else:
        g_pre_tokens_count = loaded_g_pre_tokens_count
    return merge(name, g_pre_tokens_count, vocab_size, special_tokens)


def pre_tokenize_doc(doc) -> dict[tuple[bytes], int]:
    pre_tokens_count: dict[tuple[bytes], int] = defaultdict(int)

    # counts track pre-token occurrence, key is bytes tuple
    for pre_token_group in re.finditer(constants.PAT, doc):
        pre_token = get_pre_token_bytes(pre_token_group)
        pre_token = [bytes([b]) for b in pre_token]
        key = tuple(pre_token)
        if key in pre_tokens_count:
            pre_tokens_count[key] += 1
        else:
            pre_tokens_count[key] = 1
    return pre_tokens_count


def build_pair_counts(pre_tokens_count):
    counts: dict[tuple[bytes], int] = defaultdict(int)
    for pre_token, cnt in pre_tokens_count.items():
        # count pair occurrence across all pre-tokens
        for t1, t2 in zip(pre_token, pre_token[1:]):
            counts[(t1, t2)] += cnt
    return counts


def build_heap(pair_counts):
    heap = []
    for pair, cnt in pair_counts.items():
        heap.append((-cnt, PairKey(pair), pair))
    heapq.heapify(heap)
    return heap


def print_diffs(pair_counts_1, pair_counts_2):
    for pair, cnt in pair_counts_1.items():
        if pair not in pair_counts_2:
            print(f"{pair}: pair missing in latter, the count is {cnt}")
        else:
            cnt2 = pair_counts_2[pair]
            if cnt2 != cnt:
                print(f"{pair}: count different - {cnt} VS {cnt2}")
    for pair, cnt in pair_counts_2.items():
        if pair not in pair_counts_1:
            print(f"{pair}: pair missing in former")


def merge(
    name, pre_tokens_count: dict[tuple[bytes], int], vocab_size, special_tokens
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
    pair_counts: dict[tuple[bytes], int] = build_pair_counts(pre_tokens_count)
    heap = build_heap(pair_counts)

    def update_pair(pair, delta):
        pair_counts[pair] += delta
        if pair_counts[pair] > 0:
            heapq.heappush(
                heap,
                (
                    -pair_counts[pair],
                    PairKey(pair),
                    pair,
                ),
            )
        else:
            pair_counts.pop(pair, 0)

    for i in range(vocab_size - 256 - len(special_tokens)):
        logger.warning(f"Iteration {i}...")
        # n_pair_counts = build_pair_counts(pre_tokens_count)
        # print_diffs(pair_counts, n_pair_counts)
        if not heap:
            break

        pair = None
        # if there is at least one pair
        while heap:
            # find the most occurred pair, if tie, pick lexicographically largest
            neg_cnt, key, pair = heapq.heappop(heap)
            exist_cnt = pair_counts.get(pair, -123456789)
            if exist_cnt == -neg_cnt:
                break
        if not pair:
            break
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
                    # has left neighbor
                    if new_pre_token:
                        left = new_pre_token[-1]
                        new_pair = (left, new_bytes)
                        update_pair(new_pair, cnt)
                        old_pair = (left, pair[0])
                        update_pair(old_pair, -cnt)
                    # has right neighbor
                    if idx + 2 < len(pre_token):
                        right = pre_token[idx + 2]
                        new_pair = (new_bytes, right)
                        update_pair(new_pair, cnt)
                        old_pair = (pair[1], right)
                        update_pair(old_pair, -cnt)
                    pair_counts.pop(pair, 0)
                    new_pre_token.append(new_bytes)
                    idx += 2
                else:
                    new_pre_token.append(pre_token[idx])
                    idx += 1
            new_pre_tokens_count[tuple(new_pre_token)] += cnt
        pre_tokens_count = new_pre_tokens_count
    serialize_bpe(vocab, prefix_with_name(name, VOCAB_PKL))
    serialize_bpe(merges, prefix_with_name(name, MERGES_PKL))
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
        input_paths=[
            "/Users/luyaoli/code/cs336/assignment1-basics/tests/fixtures/corpus.en"
        ],
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        name="local",
    )

    print(vocab)
    print(merges)
