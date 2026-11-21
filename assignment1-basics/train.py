"""训练脚本骨架。循环逻辑由学生填写（handout §6）。"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.nn import softmax  # noqa: F401
from cs336_basics.optim import AdamW
from cs336_basics.train_utils import (
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.transformer import TransformerLM


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Transformer LM on pre-tokenized data.")
    p.add_argument("--train_data", type=str, required=True, help="uint16 .npy of train tokens")
    p.add_argument("--val_data", type=str, required=True, help="uint16 .npy of val tokens")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=500)
    p.add_argument("--cosine_cycle_iters", type=int, default=20000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--checkpoint_path", type=str, default="experiments/ckpt.pt")
    p.add_argument("--log_path", type=str, default="experiments/train_log.txt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    print("配置:", vars(args))
    raise NotImplementedError


if __name__ == "__main__":
    main()