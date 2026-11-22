# CS336 Assignment 1 · 实验日志

记录每个训练实验的配置、结果与观察。硬件：RTX 5070 Laptop 8GB（WSL2），PyTorch 2.11+cu130。

## 实验 1：TinyStories Baseline

**日期**：2026-11-22

**命令**：

```sh
uv run python train.py \
  --train_data experiments/ts_train.npy --val_data experiments/ts_valid.npy \
  --batch_size 32 --total_steps 40000 \
  --eval_every 500 --checkpoint_every 2000 \
  --checkpoint_path experiments/ckpt_base.pt --log_path experiments/base_log.txt
```

**配置**：vocab 10000，ctx 256，4 层，16 头，d_model 512，d_ff 1344（约 22.7M 参数）；AdamW lr 1e-3 / min 1e-4，warmup 500，cosine 40k，weight decay 0.01，grad clip 1.0；bf16 autocast + TF32。

**吞吐/资源**：~48k tokens/s（0.2 s/step），峰值显存 5.6GB，总耗时约 1.9h；总 token 327.68M。

**验证曲线（fp32 eval，每 500 步，20 batch）**：

| step | 500 | 1000 | 2000 | 4000 | 8000 | 10000 | 20000 | 30000 | 39500 |
|---|---|---|---|---|---|---|---|---|---|
| val loss | 2.604 | 2.205 | 1.927 | 1.755 | 1.594 | 1.553 | 1.427 | 1.358 | 1.331 |

**最终评估**：100 batch 稳定评估 **1.3360 ± 0.0052**（目标 ≤ 1.45 ✅）。

**生成样例**（prompt `Once upon a time`，temp 0.8 / top-p 0.9，133 token 后 eot）：

> Once upon a time, there was a big box. Inside the box, there was a small toy. The toy was a ball. The ball was in the box. The ball was very useful. It could bounce and roll. One day, a little girl found the box. She saw the ball. She wanted to play with the ball. The little girl took the ball and threw it. The ball was fun to play with. The ball and the little girl played all day. They bounced and rolled. They were very happy. The ball was useful for playing. The little girl and the ball had a lot of fun. They were the best of friends.

**观察**：loss 前几百步快速下降（学会高频词与故事套路），之后进入长尾平台；cosine 退火后段再次明显下降。train/val 接近，无过拟合。

**踩坑**：batch 64 + fp32 峰值 10GB，超出 8GB 后溢出到 WSL 共享显存，step 时间 8.6s（forward 仅 0.28s）；降到 batch 32 + bf16 后 0.2s/step。显存溢出无 OOM 报错，需用 `torch.cuda.max_memory_allocated()` 实测。

---

## 实验 2：学习率 sweep（待填）

短跑 5k 步，lr ∈ {3e-4, 1e-3, 3e-3}（min_lr = lr/10，warmup 500，cosine 5k），比较最终 val loss 与稳定性。

| lr | 最终 val loss | 是否发散 | 备注 |
|---|---|---|---|
| 3e-4 | | | |
| 1e-3 | | | |
| 3e-3 | | | |