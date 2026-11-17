import random
import time
import numpy as np
from cs336_basics.bpe import Tokenizer

tiny_vocab = Tokenizer.from_files("experiments/vocab.json", "experiments/merges.json", special_tokens=["<|endoftext|>"])
owt_vocab = Tokenizer.from_files("experiments/owt_vocab_2G.json", "experiments/owt_merges_2G.json", special_tokens=["<|endoftext|>"])
random.seed(0)

with open("data/TinyStoriesV2-GPT4-train.txt", encoding="utf-8") as f:
    text = f.read(5_000_000)
    parts = text.split("<|endoftext|>")
    parts = [p for p in parts if len(p) > 0]
    sampled_parts_1 = random.sample(parts[:-1], 10)
    tot_bytes = 0
    tot_tokens = 0
    for doc in sampled_parts_1:
        new_doc = tiny_vocab.encode(doc)
        tot_bytes += len(doc.encode("utf-8"))
        tot_tokens += len(new_doc)
    
    print(f"Total bytes: {tot_bytes}, Total tokens: {tot_tokens}, Compression ratio: {tot_bytes / tot_tokens:.2f}")
        
    
with open("data/owt_subset_2G.txt", encoding="utf-8") as f:
    text = f.read(5_000_000)
    parts = text.split("<|endoftext|>")
    parts = [p for p in parts if len(p) > 0]
    sampled_parts_2 = random.sample(parts[:-1], 10)
    tot_bytes = 0
    tot_tokens = 0
    for doc in sampled_parts_2:
        new_doc = owt_vocab.encode(doc)
        tot_bytes += len(doc.encode("utf-8"))
        tot_tokens += len(new_doc)
    print(f"Total bytes: {tot_bytes}, Total tokens: {tot_tokens}, Compression ratio: {tot_bytes / tot_tokens:.2f}")
    
    tot_bytes = 0
    tot_tokens = 0
    for doc in sampled_parts_2:
        new_doc = tiny_vocab.encode(doc)
        tot_bytes += len(doc.encode("utf-8"))
        tot_tokens += len(new_doc)
    print(f"Total bytes: {tot_bytes}, Total tokens: {tot_tokens}, Compression ratio: {tot_bytes / tot_tokens:.2f}")
    
    tot_bytes = 0
    tim = time.time()
    for doc in parts:
        new_doc = owt_vocab.encode(doc)
        tot_bytes += len(doc.encode("utf-8"))
    tim = time.time() - tim
    throughput = tot_bytes / tim
    pile_time = 825 * 1024**3 / throughput / 3600
    print(f"Throughput: {throughput:.2f} bytes/s")
    print(f"Estimated time to process Pile: {pile_time:.2f} hours")
    
with open("data/TinyStoriesV2-GPT4-train.txt", encoding="utf-8") as f:
    ids = tiny_vocab.encode_iterable(f)
    arr = np.fromiter(ids, dtype=np.uint16)
    np.save("experiments/ts_train.npy", arr)
    print(arr.shape, arr.dtype, arr.nbytes)
        
with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
    ids = tiny_vocab.encode_iterable(f)
    arr = np.fromiter(ids, dtype=np.uint16)
    np.save("experiments/ts_valid.npy", arr)
    print(arr.shape, arr.dtype, arr.nbytes)

with open("data/owt_subset_2G.txt", encoding="utf-8") as f:
    ids = owt_vocab.encode_iterable(f)
    arr = np.fromiter(ids, dtype=np.uint16)
    np.save("experiments/owt_train.npy", arr)
    print(arr.shape, arr.dtype, arr.nbytes)
        
with open("data/owt_valid.txt", encoding="utf-8") as f:
    ids = owt_vocab.encode_iterable(f)
    arr = np.fromiter(ids, dtype=np.uint16)
    np.save("experiments/owt_valid.npy", arr)
    print(arr.shape, arr.dtype, arr.nbytes)