"""训练工具骨架。逻辑由学生填写（handout §4）。"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    lse = torch.logsumexp(inputs, dim=-1)
    logits = inputs[torch.arange(inputs.size(0)), targets]
    return (lse - logits).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = list(parameters)
    total_norm = torch.sqrt(
        sum(p.grad.data.norm(2).pow(2) for p in params if p.grad is not None)
    )
    if total_norm > max_l2_norm:
        clip = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip)


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    i = np.random.randint(0, len(dataset) - context_length, batch_size)
    x = torch.stack([torch.from_numpy(dataset[j : j + context_length]) for j in i])
    y = torch.stack([torch.from_numpy(dataset[j + 1 : j + context_length + 1]) for j in i])
    return (x.to(device=device, dtype=torch.long), y.to(device=device, dtype=torch.long))


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + np.cos(np.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        )
    else:
        return min_learning_rate


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]