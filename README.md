# 📒 littlelei-cs336

> 小雷（littlelei）的斯坦福 **CS336: Language Modeling from Scratch** 学习记录 🚀
> 从零开始，亲手构建语言模型的每一个零件——分词器、Transformer、训练流程……

## 🎯 关于这门课

[CS336](https://cs336.stanford.edu/) 是斯坦福的「从零构建语言模型」课程，主打 **implementation-heavy**：不只是调包，而是亲手写 BPE 分词器、Transformer 架构、优化器、分布式训练……本仓库记录实现、踩坑、优化的完整旅程。

**学习方式**：AI老师引导思路 → 自己亲手写代码 → 测试验证 → profile 驱动优化。不抄答案，享受从零造轮子的过程 😄

## 📊 进度

- [x] **Assignment 1 · BPE 分词器**（TinyStories + OWT 训练、编码/解码/序列化、压缩率/吞吐量实验）
- [x] **Assignment 1 · Transformer 语言模型**（RMSNorm / SwiGLU / RoPE / MHA / pre-norm Block / 整栈，模型测试 13/13）
- [x] **Assignment 1 · 训练工具 + 解码**（cross-entropy / 梯度裁剪 / AdamW / cosine 调度 / get_batch / checkpoint / temperature+top-p 采样）
- [x] **Assignment 1 · 训练循环 + TinyStories baseline**（327.68M token，val **1.336** $\le$ 1.45）
- [ ] Assignment 1 · 实验：LR sweep / batch size / RMSNorm / post-norm / NoPE / SwiGLU / OWT
- [ ] 后续：训练系统 / 分布式 / 缩放定律 / 对齐 …（持续更新 🌱）

## ✨ Assignment 1 · BPE

亲手实现 byte-level BPE 分词器，并在 TinyStories 全量数据（2.1G）上训练：

| 指标 | 结果 |
|------|------|
| 词表大小 | 10,000 |
| 训练耗时 | ~3.1 min |
| 峰值内存 | ~6.2 GB |
| 最长 token | ` accomplishment` |

**性能优化之路**（profile 驱动，一步步榨油）：
1. **增量更新** pair 频次——倒排索引 `backtrack`，避免每轮全量重算
2. **堆优化** merge 循环——`heapq` + 自定义 `__lt__`，批量 push 减少懒删除开销
3. **多进程预分词**——`find_chunk_boundaries` + `multiprocessing.Pool`，吃满多核（CPU ~737%）

> 💡 优化心得：并行有固定成本，小数据未必划算；堆的渐近优势要靠压低常数开销才能兑现。一切以 profile 和真实需求为准 📈

## ✨ Assignment 1 · OWT 词表 + 内存优化

在 OpenWebText 上训练 vocab 32,000。因机器内存有限（WSL2 ~15G，宿主机被占用大半），在 **2G 子集**上训练（handout 支持 downscale，资源限制 ≤100GB/≤12h 内完成）。

| 指标 | 结果 |
|------|------|
| 词表大小 | 32,000 |
| 训练语料 | OWT 2G 子集 |
| 训练耗时 | ~14.6 min |
| 峰值内存 | ~3.5 GB |
| 最长 token | 64 个连字符 `----…----`（网页分隔线 😄）|

**内存优化之路**（从反复 OOM 到 2.4G，全程用 `systemd-run` 内存帽保护系统）：
1. 起初 1G/2G 子集 + vocab 32000 都 **>14G OOM**——砍语料没用，瓶颈在 merge 循环
2. **堆定期重建**（每 200 轮 `heapify` 清过期节点）→ 略有改善，但非主犯
3. **关键一击**：清理 `backtrack` 里没人收尸的死词 + 删除零频 `counts` 条目 → 峰值从 >14G **暴跌至 2.4G**（1G）/ 3.5G（2G）

> 💡 优化心得：**实测 > 直觉**。一开始猜堆幽灵是主犯，profile 后才发现真正的泄漏是 backtrack 集合里被死词 tuple 占着、GC 回收不掉的内存。抓泄漏要顺着"谁还指着这块内存"去想 🧠

## ✨ Assignment 1 · Transformer 语言模型

从零手写完整 GPT 式模型，无 `nn.Linear`/`nn.Embedding` 等现成层（全部自实现）：

| 组件 | 实现要点 |
|------|---------|
| `Linear` / `Embedding` | 手写 `Parameter` + 截断正态初始化（$\sigma^2=2/(d_{in}+d_{out})$ / $N(0,1)$，`trunc_normal_`） |
| `RMSNorm` | 最后一维求 RMS，先 upcast fp32 再 downcast；无均值中心化 |
| `SwiGLU` | $W_2(\mathrm{SiLU}(W_1x)\odot W_3x)$，$d_{ff}\approx\frac83 d_{model}$ |
| RoPE | 预计算 cos/sin buffer（不学习），相邻配对旋转 $q,k$，不转 $v$ |
| Attention | $\mathrm{softmax}(QK^\top/\sqrt{d_k}+M)V$，布尔 mask（False→$-\infty$） |
| MHA | 单次 QKV 投影 → 拆头 → RoPE → 因果 mask → 合头 → $W_o$ |
| Block | pre-norm + residual：$x\leftarrow x+\mathrm{Sublayer}(\mathrm{RMSNorm}(x))$ |
| LM | Embedding → N×Block → final RMSNorm → LM head（位置信息全靠 RoPE） |

验收：`uv run pytest` → **47 passed, 1 xfailed**（xfail 为官方标记的预期失败项），其中模型测试 **13/13**。

配套书面题（Resource Accounting）与数学推导整理在 `writeup.tex`、`math_notes.tex` 中。

## ✨ Assignment 1 · 训练工具 + 解码

| 模块 | 实现要点 |
|------|---------|
| cross-entropy | log-sum-exp 数值稳定；返回 batch 平均 |
| gradient clipping | 全局 L2 范数 + 等比缩放（只改长度不改方向），原地更新 |
| AdamW | $m,v$ 二阶矩 + 偏差修正折进步长 + **解耦**权重衰减 |
| LR schedule | 线性 warmup + cosine 退火 + 尾部常数 |
| `get_batch` | 随机起点采样，label 为 input 右移一位 |
| checkpoint | model/optimizer state + iteration 存取，支持续训 |
| decode | 自回归采样：温度缩放 + top-p（nucleus）+ eot 停止 + 贪心兜底 |

`train.py` 的参数框架（argparse）已就绪，训练循环待填；数据已离线 tokenize 为 uint16 `.npy`。

## ✨ Assignment 1 · TinyStories Baseline

按 handout 标准配置训练（327.68M token），硬件 RTX 5070 Laptop 8GB（WSL2）：

| 项目 | 值 |
|------|-----|
| 模型 | 4 层 / 16 头 / d_model 512 / d_ff 1344 / ctx 256 / vocab 10000 |
| 训练 | batch 32 × 40k 步，AdamW lr 1e-3（warmup 500 + cosine 40k），wd 0.01，clip 1.0 |
| 精度/加速 | bf16 autocast + TF32；~48k tokens/s，峰值显存 5.6GB，耗时 ~1.9h |
| 验证 | **val loss 1.336 ± 0.005**（100 batch 稳定评估，fp32） |
| 产物 | `experiments/base_log.txt`、`experiments/ckpt_base.pt`（ckpt 已 gitignore） |

生成样例（prompt `Once upon a time`，temp 0.8 / top-p 0.9，133 token 后自然 eot）：

> Once upon a time, there was a big box. Inside the box, there was a small toy. The toy was a ball. The ball was in the box. The ball was very useful. It could bounce and roll. One day, a little girl found the box. ... They were the best of friends.

> 💡 踩坑记录：batch 64 + fp32 峰值 10GB，溢出到 WSL 共享显存导致 **8.6s/step**；降到 batch 32 + bf16 后 0.2s/step，快 20 倍。显存溢出不会报 OOM，只会悄悄变慢——用 `max_memory_allocated` 实测最靠谱。

完整实验记录见 [`experiments/experiment_log.md`](assignment1-basics/experiments/experiment_log.md)。

## 🧪 Tokenizer 实验

| 实验 | 结果 |
|------|------|
| (a) 压缩率 | TinyStories **4.14** / OWT **4.38** bytes/token |
| (b) 跨词表 | OWT 用 TinyStories 词表 → **3.26**（领域不匹配，token +34%）|
| (c) 吞吐量 | ~**0.83 MB/s**，编码 Pile 825GB ≈ **282 小时**（纯 Python 瓶颈，业界用 Rust）|
| (d) uint16 编码 | train+dev → `.npy`（vocab < 65536，2 字节比 int64 省 4 倍空间）|

## 📁 仓库结构

```
CS336/
├── assignment1-basics/        # Assignment 1
│   ├── cs336_basics/
│   │   ├── bpe.py            # BPE（train_bpe + Tokenizer）
│   │   ├── nn.py             # Linear/Embedding/RMSNorm/SwiGLU + softmax/silu
│   │   ├── transformer.py    # RoPE / attention / MHA / Block / LM
│   │   ├── optim.py          # AdamW
│   │   ├── train_utils.py    # loss/clip/get_batch/schedule/checkpoint
│   │   └── generate.py       # decode（temperature + top-p）
│   ├── train.py              # 训练脚本（argparse 就绪，循环待填）
│   ├── train_tokenizer.py    # TinyStories BPE 训练驱动
│   ├── train_owt.py          # OWT BPE 训练驱动
│   ├── tokenizer_experiments.py  # 压缩率/吞吐量/uint16 编码实验
│   ├── math_notes.tex        # 数学笔记（PE/RoPE/attention/AdamW…）
│   ├── writeup.tex           # 书面题答案（unicode/BPE/tokenizer/记账）
│   ├── experiments/          # 词表/日志/编码数组/实验记录（.npy/.pt 已 gitignore）
│   ├── tests/                # 官方测试
│   └── data/                 # 数据集（已 gitignore，需自行下载）
├── lectures/                 # 课程讲义代码（课件素材已 gitignore）
└── vibe_images/              # 讲解示意图（AI 生成，未纳入版本库）
```

## 🚀 快速开始

环境用 [uv](https://docs.astral.sh/uv/) 管理：

```sh
cd assignment1-basics
uv run pytest -q                      # 全量测试（当前 47 passed, 1 xfailed）
uv run python train.py --help         # 训练脚本参数
uv run python train.py --train_data experiments/ts_train.npy --val_data experiments/ts_valid.npy \
  --batch_size 32 --total_steps 40000 --checkpoint_path experiments/ckpt.pt --log_path experiments/train_log.txt
uv run python train_tokenizer.py      # 训练 TinyStories 分词器
uv run python train_owt.py            # 训练 OWT 分词器
uv run python tokenizer_experiments.py  # 压缩率/吞吐量/uint16 实验
```

数据集需自行下载（见 `assignment1-basics/README.md`），体积较大已忽略不上传。

## 📝 说明

- 课程材料（handout、讲义、官方测试）版权归斯坦福所有，本仓库仅作个人学习记录。
- 大件素材（数据集、课件图片等）已 gitignore，仓库聚焦个人代码与笔记。

---

⭐ 如果这段从零造轮子的旅程对你有启发，欢迎 star～ 持续更新中 🌱