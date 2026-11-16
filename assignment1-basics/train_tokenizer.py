import time
from cs336_basics.bpe import train_bpe, Tokenizer
t0 = time.time()
vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
print("耗时:", time.time() - t0)
Tokenizer(vocab, merges, ["<|endoftext|>"]).save("experiments/vocab.json", "experiments/merges.json")
print("最长token:", max(vocab.values(), key=len))