"""模型基础模块骨架（handout §3.1–3.3）。"""
from __future__ import annotations

import math
import torch
from torch import Tensor


class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        epsilon = math.sqrt(2 / (d_in + d_out))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=epsilon, a=-3*epsilon, b=3*epsilon)
        
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        type = x.dtype
        x = x.float()
        RMS = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return (x / RMS * self.weight).to(type)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))


def softmax(x: Tensor, dim: int) -> Tensor:
    m = torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x - m) / torch.sum(torch.exp(x - m), dim=dim, keepdim=True)

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)