import time
from cs336_basics.bpe import train_bpe, Tokenizer
t0 = time.time()
vocab, merges = train_bpe("data/owt_train.txt", 32000, ["<|endoftext|>"])
print("耗时:", time.time() - t0)
Tokenizer(vocab, merges, ["<|endoftext|>"]).save("experiments/owt_vocab.json", "experiments/owt_merges.json")
print("最长token:", max(vocab.values(), key=len))