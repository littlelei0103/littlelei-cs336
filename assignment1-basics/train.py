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
    p.add_argument("--cosine_cycle_iters", type=int, default=40000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--checkpoint_path", type=str, default="experiments/ckpt.pt")
    p.add_argument("--log_path", type=str, default="experiments/train_log.txt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings (RoPE)")
    p.add_argument("--post_norm", action="store_true", help="Use post-layer normalization instead of pre-layer normalization")
    p.add_argument("--no_rmsnorm", action="store_true", help="Disable RMSNorm and use Identity instead")
    p.add_argument("--ffn_silu", action="store_true", help="Disable SwiGLU and use FFNSiLU instead")
    return p


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = build_arg_parser().parse_args()
    print("配置:", vars(args))
    train_tokens = np.load(args.train_data, mmap_mode="r")
    val_tokens = np.load(args.val_data, mmap_mode="r")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_rope=not args.no_rope,
        post_norm=args.post_norm,
        use_rmsnorm=not args.no_rmsnorm,
        use_swiglu=not args.ffn_silu
    ).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = load_checkpoint(args.checkpoint_path, model, optimizer) if args.resume else 0
    with open(args.log_path, "a" if args.resume else "w") as log:
        log.write(f"Starting training from step {start}\n")
        log.write(f"Training on {args.device}\n")
        log.flush()
        model.train()
        for step in range(start, args.total_steps):
            t = time.time()
            lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
            for g in optimizer.param_groups:
                g["lr"] = lr
            x, y = get_batch(train_tokens, args.batch_size, args.context_length, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()
            log.write(f"Step {step}: loss={loss.item():.4f}, lr={lr:.6f}, tokens/sec={len(x) * x.shape[1] / (time.time() - t):.2f}\n")
            log.flush()
            if step % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for _ in range(args.eval_batches):
                        x_val, y_val = get_batch(val_tokens, args.batch_size, args.context_length, args.device)
                        logits_val = model(x_val)
                        val_loss += cross_entropy(logits_val.reshape(-1, args.vocab_size), y_val.reshape(-1)).item()
                    val_loss /= args.eval_batches
                log.write(f"Validation loss at step {step}: {val_loss:.4f}\n")
                log.flush()
                model.train()
            if step % args.checkpoint_every == 0:
                save_checkpoint(model, optimizer, step, args.checkpoint_path)
                log.write(f"Checkpoint saved at step {step} to {args.checkpoint_path}\n")
                log.flush()
        save_checkpoint(model, optimizer, args.total_steps, args.checkpoint_path)
        log.write(f"Training completed at step {args.total_steps}\n")
        log.flush()

        


if __name__ == "__main__":
    main()