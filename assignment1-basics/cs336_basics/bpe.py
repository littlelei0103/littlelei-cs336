"""BPE tokenizer 实现。

这是一个【骨架】：只搭了结构和签名，所有算法逻辑留给你自己填写。
每个 TODO 对应你自己设计过的决策 1-5，照着你的思路填进去即可。
"""

from __future__ import annotations
import json
import os
from collections.abc import Iterable, Iterator
from collections import defaultdict
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
import heapq

# GPT-2 预分词正则（handout 第 6 页直接给定，可原样使用）
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

num_processes = min(4, os.cpu_count())

class HeapNode:
    def __init__(self, pair: tuple[bytes, bytes], count: int):
        self.pair = pair
        self.count = count

    def __lt__(self, other: HeapNode) -> bool:
        # 频次高的优先级高，平局按字典序更大者优先
        return (self.count, self.pair) > (other.count, other.pair)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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

def process_chunk(args) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args
    counts = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start)
        text = text.decode("utf-8", errors="ignore")
        pattern = "|".join(re.escape(token) for token in special_tokens)
        segments = re.split(pattern, text) if pattern else [text]
        for segment in segments:
            for m in re.finditer(PAT, segment):
                b = m.group()
                bs = b.encode("utf-8")
                t = tuple(bytes([b]) for b in bs)
                counts[t] += 1
    return counts

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 byte-level BPE，返回 (vocab, merges)。"""
    # TODO(决策1): 读语料 -> 预分词 -> 建频次表 dict[tuple[bytes, ...], int]
    counts = defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes, split_special_token=b"<|endoftext|>")
        tasks = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_chunk, tasks)
            for result in results:
                for key, value in result.items():
                    counts[key] += value
    #print(counts)
    # TODO(决策5): 记录 vocab 与 merges，达到 vocab_size 时停止
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        b = token.encode("utf-8")
        vocab[len(vocab)] = b
    merges = []
    # TODO(决策2): 统计所有相邻 pair 的加权频次
    pair_counts = defaultdict(int)
    backtrack = defaultdict(set)
    for key, value in counts.items():
        for i in range(len(key) - 1):
            pair = (key[i], key[i + 1])
            backtrack[pair].add(key)
            pair_counts[pair] += value
    #print(pair_counts)
    # 使用堆来优化选出频次最高的 pair
    heap = [HeapNode(pair, count) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    cnt = 0
    while len(vocab) < vocab_size:
        # TODO(决策3): 选出频次最高的 pair（平局取字典序更大者）
        if cnt % 200 == 0:
            heap = [HeapNode(pair, count) for pair, count in pair_counts.items()]
            heapq.heapify(heap)
        while heap:
            best_node = heapq.heappop(heap)
            best = best_node.pair
            if best in pair_counts and pair_counts[best] == best_node.count:
                break
        else:
            break  # No more pairs to merge
        #print(best)
        # TODO(决策4): 合并该 pair，构造新 tuple / 新表
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        changed = set()
        for key in list(backtrack[best]):
            new_key = []
            i = 0
            while i < len(key):
                if i < len(key) - 1 and (key[i], key[i + 1]) == best:
                    new_key.append(best[0] + best[1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_key = tuple(new_key)
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                pair_counts[pair] -= counts[key]
                backtrack[pair].discard(key)
                changed.add(pair)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                    del backtrack[pair]
            for i in range(len(new_key) - 1):
                pair = (new_key[i], new_key[i + 1])
                pair_counts[pair] += counts[key]
                changed.add(pair)
                backtrack[pair].add(new_key)
            backtrack[best].discard(key)
            counts[new_key] += counts[key]
            del counts[key]
        
        for pair in changed:
            if pair in pair_counts:
                heapq.heappush(heap, HeapNode(pair, pair_counts[pair]))
                
        cnt += 1
    #print(merges)
    return vocab, merges


class Tokenizer:
    """加载 vocab + merges，做编码 / 解码。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # TODO: 保存 vocab / merges；special_tokens 若不在词表则补入
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        for token in self.special_tokens:
            b_token = token.encode("utf-8")
            if b_token not in self.reverse_vocab:
                self.vocab[len(self.vocab)] = b_token
                self.reverse_vocab[b_token] = len(self.vocab) - 1
        self.rank = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        # TODO: 从序列化文件读回 vocab / merges，再构造 cls(...)
        with open(vocab_filepath) as f:
            raw_vocab = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}
        with open(merges_filepath) as f:
            raw_merges = json.load(f)
        merges = [tuple(x.encode("latin-1") for x in pair) for pair in raw_merges]
        return cls(vocab, merges, special_tokens)

    def save(self, 
             vocab_filepath: str,
             merges_filepath: str
    ) -> None:
        with open(vocab_filepath, "w") as f:
            json.dump({str(k) : v.decode("latin-1") for k, v in self.vocab.items()}, f)
        with open(merges_filepath, "w") as f:
            json.dump([[a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges], f)

    def encode(self, text: str) -> list[int]:
        # TODO: 预分词 -> 按 merges 生成顺序合并 -> 映射成 ID
        #       特殊 token 必须整体保留为单个 ID
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        patterns = [re.escape(token) for token in self.special_tokens]
        segments = re.split("(" + "|".join(patterns) + ")", text) if patterns else [text]
        ids = []
        for segment in segments:
            if segment in self.special_tokens:
                ids.append(self.reverse_vocab[segment.encode("utf-8")])
            else:
                for m in re.finditer(PAT, segment):
                    b = m.group().encode("utf-8")
                    tokens = [bytes([b]) for b in b]
                    while True:
                        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                        best_pair = None
                        best_rank = float("inf")
                        for pair in pairs:
                            if pair in self.rank and self.rank[pair] < best_rank:
                                best_rank = self.rank[pair]
                                best_pair = pair
                        if best_pair is None:
                            break
                        new_tokens = []
                        i = 0
                        while i < len(tokens):
                            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                                new_tokens.append(best_pair[0] + best_pair[1])
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1
                        tokens = new_tokens
                    ids.extend(self.reverse_vocab[token] for token in tokens)
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # TODO: 惰性逐块 yield ID；注意 token 不能跨块边界
        buffer = []
        for chunk in iterable:
            buffer.append(chunk)
            text = "".join(buffer)
            cut = 0
            for m in re.finditer(PAT, text):
                cut = m.start()
            yield from self.encode(text[:cut])
            buffer = [text[cut:]]
        yield from self.encode("".join(buffer))
                
                
        

    def decode(self, ids: list[int]) -> str:
        # TODO: 查 vocab 拼出 bytes，再 decode("utf-8", errors="replace")
        bytes_list = []
        for token_id in ids:
            bytes_list.append(self.vocab[token_id])
        return b"".join(bytes_list).decode("utf-8", errors="replace")


if __name__ == "__main__":
    # TODO(Step A): 先用 bpe_example 的频次表硬编码测试你的合并循环，
    #   对照 handout 第 7 页的合并序列 ['s t', 'e st', 'o w', 'l ow', ...]
    train_bpe("bpe_example.txt", vocab_size=264, special_tokens=["<pad>", "<unk>"])
    pass