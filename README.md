# 📒 littlelei-cs336

> 小雷（littlelei）的斯坦福 **CS336: Language Modeling from Scratch** 学习记录 🚀
> 从零开始，亲手构建语言模型的每一个零件——分词器、Transformer、训练流程……

## 🎯 关于这门课

[CS336](https://cs336.stanford.edu/) 是斯坦福的「从零构建语言模型」课程，主打 **implementation-heavy**：不只是调包，而是亲手写 BPE 分词器、Transformer 架构、优化器、分布式训练……本仓库记录实现、踩坑、优化的完整旅程。

**学习方式**：AI老师引导思路 → 自己亲手写代码 → 测试验证 → profile 驱动优化。不抄答案，享受从零造轮子的过程 😄

## 📊 进度

- [x] **Assignment 1 · BPE 分词器**（训练 + 编码/解码/序列化）
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

## 📁 仓库结构

```
CS336/
├── assignment1-basics/        # Assignment 1
│   ├── cs336_basics/bpe.py    # 核心实现（train_bpe + Tokenizer）
│   ├── train_tokenizer.py     # 训练驱动脚本
│   ├── experiments/           # 训练产物 (vocab/merges) + profile 输出
│   ├── tests/                 # 官方测试
│   └── data/                  # 数据集（已 gitignore，需自行下载）
└── lectures/                  # 课程讲义代码（课件素材已 gitignore）
```

## 🚀 快速开始

环境用 [uv](https://docs.astral.sh/uv/) 管理：

```sh
cd assignment1-basics
uv run pytest                      # 跑测试
uv run python train_tokenizer.py   # 训练分词器
```

数据集需自行下载（见 `assignment1-basics/README.md`），体积较大已忽略不上传。

## 📝 说明

- 课程材料（handout、讲义、官方测试）版权归斯坦福所有，本仓库仅作个人学习记录。
- 大件素材（数据集、课件图片等）已 gitignore，仓库聚焦个人代码与笔记。

---

⭐ 如果这段从零造轮子的旅程对你有启发，欢迎 star～ 持续更新中 🌱