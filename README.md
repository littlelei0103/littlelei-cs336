# 📒 littlelei-cs336

> 小雷（littlelei）的斯坦福 **CS336: Language Modeling from Scratch** 学习记录 🚀
> 从零开始，亲手构建语言模型的每一个零件——分词器、Transformer、训练流程……

## 🎯 关于这门课

[CS336](https://cs336.stanford.edu/) 是斯坦福的「从零构建语言模型」课程，主打 **implementation-heavy**：不只是调包，而是亲手写 BPE 分词器、Transformer 架构、优化器、分布式训练……本仓库记录实现、踩坑、优化的完整旅程。

**学习方式**：AI老师引导思路 → 自己亲手写代码 → 测试验证 → profile 驱动优化。不抄答案，享受从零造轮子的过程 😄

## 📊 进度

- [x] **Assignment 1 · BPE 分词器**（TinyStories + OWT 训练、编码/解码/序列化、压缩率/吞吐量实验）
- [ ] Assignment 1 · Transformer 语言模型
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
│   ├── cs336_basics/bpe.py    # 核心实现（train_bpe + Tokenizer）
│   ├── train_tokenizer.py     # TinyStories 训练驱动
│   ├── train_owt.py           # OWT 训练驱动
│   ├── tokenizer_experiments.py  # 压缩率/吞吐量/uint16 编码实验
│   ├── experiments/           # 训练产物 (vocab/merges/.npy) + 运行日志
│   ├── tests/                 # 官方测试
│   └── data/                  # 数据集（已 gitignore，需自行下载）
└── lectures/                  # 课程讲义代码（课件素材已 gitignore）
```

## 🚀 快速开始

环境用 [uv](https://docs.astral.sh/uv/) 管理：

```sh
cd assignment1-basics
uv run pytest                        # 跑测试
uv run python train_tokenizer.py     # 训练 TinyStories 分词器
uv run python train_owt.py           # 训练 OWT 分词器
uv run python tokenizer_experiments.py  # 压缩率/吞吐量/uint16 实验
```

数据集需自行下载（见 `assignment1-basics/README.md`），体积较大已忽略不上传。

## 📝 说明

- 课程材料（handout、讲义、官方测试）版权归斯坦福所有，本仓库仅作个人学习记录。
- 大件素材（数据集、课件图片等）已 gitignore，仓库聚焦个人代码与笔记。

---

⭐ 如果这段从零造轮子的旅程对你有启发，欢迎 star～ 持续更新中 🌱