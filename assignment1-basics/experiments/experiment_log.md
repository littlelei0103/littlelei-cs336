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

## 实验 2：学习率 sweep

短跑 5k 步（batch 32，warmup 500，cosine 5k，min_lr = lr/10），对最终 checkpoint 做 100 batch 稳定评估：

| lr | final val loss | 是否发散 | 备注 |
|---|---|---|---|
| 3e-4 | 1.8104 ± 0.0057 | 否 | 收敛偏慢 |
| 5e-4 | 1.7190 ± 0.0063 | 否 | 接近最优区间下沿 |
| 1e-3 | 1.6339 ± 0.0052 | 否 | 基线采用（稳定、已验证 40k 全量） |
| **2e-3** | **1.6260 ± 0.0067** | 否 | 5k 步略优，但与 1e-3 差异在 ~1σ 内 |
| 3e-3 | 1.9832 ± 0.0065 | 否 | 明显过热，但未发散 |

**观察**：最优区间为 1e-3–2e-3（5k 步内差异在噪声量级）；3e-4/5e-4 学习过慢；3e-3 更新过大、收敛明显落后。短跑结论与 40k 全量一致（1e-3 全量最终 1.336）。全量训练选 1e-3（保守稳定）；2e-3 的全量表现可作为后续实验。

### 2b. 学习率上界与发散探针

Handout 要求至少一条发散曲线。继续把 lr 往上推（batch 32，warmup 500，cosine 退火到 lr/10），最终 checkpoint 做 100 batch 稳定评估：

| lr | steps | grad clip | final val loss | 现象 |
|---|---|---|---|---|
| 1e-2 | 5000 | 1.0 | 3.9806 ± 0.0078 | 稳定，但明显比 1e-3 差，曲线中段还在 4.9 震荡 |
| 3e-2 | 5000 | 1.0 | 4.1284 ± 0.0073 | 稳定，收敛更慢 |
| 1e-1 | 3000 | 1.0 | 4.4944 ± 0.0070 | 稳定，val 在 4.1–5.2 大幅波动 |
| 3e-1 | 2000 | 1.0 | 4.3743 ± 0.0069 | 稳定，val 一度到 6.0 |
| 1e-1 | 2000 | 关闭 | 3.9520 ± 0.0070 | 与开裁剪几乎一致——裁剪不是此处稳定性的关键 |
| **1.0** | 2000 | 关闭 | **5.8217 ± 0.0043** | **发散**：val 在 step 250–500 飙到 347、362，train loss 100+；退火到 0.1 后恢复到 ~5.8 |

（表中 ± 为 100 batch 评估的标准误。）

**观察**：

- 从 1e-3 一路推到 3e-1，模型都没有崩，只是收敛变差并在 4–6 之间震荡。原因：AdamW 每步位移 ≈ lr（与梯度尺度无关，`m/√v` 近似符号函数），RMSNorm 限制激活尺度，pre-norm 残差保证梯度通路——三者共同把发散边界推得非常高。所以这里的"最优 lr"由收敛速度与更新噪声的平衡决定，而不是由发散阈值决定。
- 关掉梯度裁剪后 1e-1 的结果几乎不变，说明在这套架构下裁剪主要防的是偶发尖峰，而 stabilizing 的主力是归一化 + AdamW。
- lr=1.0 时每步参数位移和参数本身的尺度同量级，warmup 阶段就开始震荡发散（val 347→362）；cosine 把 lr 退到 0.1 后训练又"活"了过来，但只恢复到 5.8。这直观展示了发散与退火的关系：**大 lr 造成的参数破坏可以被退火部分挽救，但已经损失的质量无法追回**。

---

## 实验 3：batch size

固定总 token ≈ 41M、lr 1e-3、min_lr 1e-4，对最终 checkpoint 做 100 batch 稳定评估：

| batch | steps | tokens | final val loss | 备注 |
|---|---|---|---|---|
| 1 | 5000 | 1.28M | 2.8836 ± 0.0074 | 远少于其他 run 的 token，训练不足，仅作极端参考 |
| 8 | 20000 | 41.0M | 1.6392 ± 0.0059 | 与 16/32 差异在噪声内 |
| 16 | 10000 | 41.0M | **1.6152 ± 0.0070** | 更新次数最多，略优 |
| 32 | 5000 | 41.0M | 1.6339 ± 0.0052 | 来自实验 2 |
| 48 | 3333 | 40.96M | 1.6509 ± 0.0061 | 吞吐最高，但更新次数少 |

**观察**：在约 41M token 预算下，batch ≥ 8 的最终 val 都在 1.61–1.65，差异约 0.02（在评估噪声量级）；batch 越小梯度更新越频繁，B=16 略优，但小 batch 的 GPU 利用率更低。B=1 只训了 1.28M token，loss 2.88，说明 token 预算远比更新次数重要。受 8GB 显存限制，B=64 已溢出到共享显存（实测峰值 8.4GB，step 时间从 0.2s 涨到 0.84s），B=128 无法运行；B=48 峰值 7.3GB 已是可行上限附近。

---

## 实验 4：OpenWebText 主实验

按 handout 要求：与 TinyStories **相同迭代数（40k 步）**，OWT 32k 词表。因 32k 词表 + 8GB 显存，batch 降到 16（32k 词表下 B=32 峰值 8.27GB 会溢出）。

**配置**：vocab 32000，ctx 256，4 层 / 16 头 / d_model 512 / d_ff 1344（约 45M 参数）；AdamW lr 1e-3 / min 1e-4，warmup 500，cosine 40k，wd 0.01，clip 1.0；bf16 + TF32。

**吞吐/资源**：~34k tokens/s，峰值显存 5.6GB，耗时约 76 分钟；总 token 164M（B=16）。

**验证曲线（fp32 eval，20 batch）**：

| step | 500 | 1000 | 2000 | 5000 | 10000 | 20000 | 30000 | 39500 |
|---|---|---|---|---|---|---|---|---|
| val loss | 6.108 | 5.751 | 5.277 | 4.928 | 4.619 | 4.347 | 4.177 | 4.099 |

**最终评估**：100 batch 稳定评估 **4.0818 ± 0.0108**。

**生成样例**（prompt `The`，temp 0.8 / top-p 0.9，256 token 截断）：

> The risk of being a lug for a few years in a short time is very low. "That's going to be a lot more difficult," said Ajnevze. "It's a lot more dangerous than my non-sugar. That's a lot of pressure. ..." And here's the catch: That's a bit of a shock. In a recent report by New York Times ... all the other major research groups, have found that this is about the economic success of the research center. ...

**解读**：OWT loss（4.08）远高于 TinyStories（1.336），但两者不可直接比较——(1) 词表不同（32k vs 10k），per-token loss 的"每 token 信息量"不同；(2) OWT 主题/文体极其多样、长尾词多，条件熵本身更高，同容量模型只能学到高频模式；(3) 相同迭代数下 OWT 的 token 数只有一半（B=16 vs 32）。因此生成文本是"新闻语体但事实/逻辑不连贯"，符合小模型 + 难数据集的预期。

**备注**：先跑过一版 20k 步 OWT（val 4.24），再完整重跑 40k 以保证 cosine schedule 干净。

---

## 实验 5：Context length

固定总 token ≈ 41M、lr 1e-3、min_lr 1e-4，对最终 checkpoint 做 100 batch 稳定评估：

| ctx | batch | steps | tokens | final val loss | 吞吐 | 峰值显存 |
|---|---|---|---|---|---|---|
| 128 | 32 | 10000 | 41.0M | 1.6773 ± 0.0068 | ~69k tok/s | 2.7GB |
| 256 | 32 | 5000 | 41.0M | 1.6339 ± 0.0052 | ~48k tok/s | 5.6GB |
| 512 | 16 | 5000 | 41.0M | **1.5905 ± 0.0071** | ~34k tok/s | 5.9GB |

**观察**：相同 token 预算下，更长的 context 明显降低 per-token loss（模型能利用更长的前文，条件熵更低）；代价是 attention 的 $O(s^2)$ 计算使吞吐约减半（ctx128 → ctx512：69k → 34k tok/s），显存也上升。TinyStories 的 ctx 256 是速度/效果的折中；OWT 可用更长 context 但需更多算力。

---

## 实验 6：warmup 与 weight decay

5k 步短跑（B=32，lr 1e-3，min_lr 1e-4，cosine 5k），每次只改一个因素，100 batch 稳定评估：

| 变体 | 设置 | final val loss | 备注 |
|---|---|---|---|
| warmup 100 | wd 0.01 | 1.6483 ± 0.0061 | 更快进入高 lr，短跑内差异小 |
| warmup 500（基线） | wd 0.01 | 1.6339 ± 0.0052 | 全量训练采用 |
| warmup 2000 | wd 0.01 | 1.6460 ± 0.0070 | 2k 占短跑 40%，略吃亏 |
| wd 0.0 | warmup 500 | 1.6462 ± 0.0066 | 略差于 0.01 |
| wd 0.01（基线） | warmup 500 | 1.6339 ± 0.0052 | 全量训练采用 |
| **wd 0.1** | warmup 500 | **1.6285 ± 0.0058** | 组内最优，但差异 ~1σ |

**观察**：短跑内各变体差异 ≤0.02，基本在评估噪声量级（se ~0.006）；warmup 100–2000 影响不显著（长 warmup 在 5k 步中占比过大时略吃亏）；weight decay 0.1 略优但优势很小。全量训练保持 warmup 500 / wd 0.01 的稳定配置。

---

## 实验 7：位置编码消融（RoPE vs NoPE）

10k 步、B=32、ctx 256、cosine 10k、lr 1e-3，两个 run 仅差 `--no_rope` 开关（实现：MHA 可选 `use_rope`，默认 True，现有 47 个测试不受影响）：

| 变体 | final val loss（100 batch） | 备注 |
|---|---|---|
| RoPE（对照） | **1.4920 ± 0.0048** | 10k 步同 schedule |
| NoPE | 1.5932 ± 0.0065 | 差约 0.10 nat |

**观察**：NoPE 仍能正常训练并生成语言——因果 mask 的下三角结构让位置 $i$ 可见的 key 数恰为 $i+1$，隐式泄露了位置；但模型没有显式距离概念，需要"精确相对距离"的模式（如复制前一个 token）只能靠内容统计近似，因此曲线全程落后 RoPE，且差距随训练缓慢扩大。曲线见 `plots/rope_vs_nope.png`。

---

## 实验 8：Post-norm vs Pre-norm

10k 步、B=32、ctx 256、cosine 10k、lr 1e-3、warmup 500、wd 0.01、clip 1.0；两版仅差 `--post_norm` 开关，均保留 `ln_final` 以控制变量。post-norm 定义为 $z=\mathrm{RMSNorm}(x+\mathrm{MHA}(x))$、$y=\mathrm{RMSNorm}(z+\mathrm{FFN}(z))$。

| 变体 | final val loss（100 batch） | 备注 |
|---|---|---|
| Pre-norm（对照） | **1.4920 ± 0.0048** | 10k 步同 schedule |
| Post-norm | 1.5186 ± 0.0062 | 差约 0.027（约 4 个标准误），但全程未出现不稳定 |

**观察**：post-norm 两条曲线几乎贴着 pre-norm（如 step 2500 为 1.908 vs 1.873），没有出现 handout 暗示的明显训练不稳定；最终 loss 略差。可能原因：模型只有 4 层、warmup 500 + 梯度裁剪 1.0 已提供足够稳定性；pre-norm 的优势（残差流无归一化、梯度直通）在小模型上体现为小幅而非数量级差异。曲线见 `plots/postnorm_vs_prenorm.png`。

---

## 实验 9：移除 RMSNorm

10k 步、其余配置同上，`--no_rmsnorm` 把所有 RMSNorm（含 `ln_final`）替换为 `nn.Identity`。

| lr | 结果 | final val loss（100 batch） | 备注 |
|---|---|---|---|
| 1e-3（原最优） | **发散** | — | val：2.72→2.37→2.24→**130.8**（step 2000）→2.00→**nan**（step 3000） |
| 3e-4 | 稳定 | 1.6672 ± 0.0067 | 曲线正常收敛 |

**观察**：没有 RMSNorm 时残差流的尺度随深度累积、无法被拉回，lr=1e-3 下 step 2000 激活/输出尺度过大导致 val 飙到 130，随后出现 nan；把 lr 降低 3 倍到 3e-4 可以稳定训练，但最终比 pre-norm（1.4920）差约 0.175。说明 RMSNorm 不仅是大 lr 稳定性的关键，也直接影响这个预算下的最终质量。曲线见 `plots/no_rmsnorm.png`。

---

## 实验 10：SwiGLU vs SiLU

10k 步、其余配置同上；SwiGLU 取 $d_{\text{ff}}=1344$（3 个矩阵，22.70M 参数），SiLU 取 $d_{\text{ff}}=2048=4d$（2 个矩阵，22.83M 参数，差 0.6%），`--ffn_silu` 开关切换。

| FFN | 参数量 | final val loss（100 batch） | 备注 |
|---|---|---|---|
| SwiGLU（对照） | 22.70M | **1.4920 ± 0.0048** | 10k 步同 schedule |
| SiLU（无门控） | 22.83M | 1.5363 ± 0.0056 | 差约 0.044（约 6 个标准误） |

**观察**：两条曲线前 3k 步几乎重合，之后 SiLU 略高；在参数量基本对齐的前提下，带门控的 SwiGLU 最终更好（与 Shazeer 的结论一致）。门控让 FFN 的输出幅度可以依赖输入内容动态调节，而同样参数预算下 SiLU 只能把 d_ff 加宽来补偿。曲线见 `plots/swiglu_vs_silu.png`。