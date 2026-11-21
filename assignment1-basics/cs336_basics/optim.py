"""AdamW 优化器骨架。逻辑由学生填写（handout §4.2）。"""
from __future__ import annotations
import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, defaults=dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                beta1, beta2 = group["betas"]
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad.pow(2)
                t = state["step"]
                lr_t = group["lr"] * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.addcdiv_(state["m"], state["v"].sqrt().add_(group["eps"]), value=-lr_t)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                