"""自回归解码骨架。逻辑由学生填写（handout §6）。"""
from __future__ import annotations

import torch


def decode(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
) -> list[int]:
    prompt = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device=next(model.parameters()).device)
    next_token =  []
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(prompt)
            last = logits[0, -1]
            if temperature > 0:
                p = torch.softmax(last / temperature, dim=-1)
                sorted_p, idx = torch.sort(p, descending=True)
                cum = torch.cumsum(sorted_p, dim=0)
                sorted_p[cum - sorted_p >= top_p] = 0
                sorted_p /= sorted_p.sum()
                tok = torch.multinomial(sorted_p, num_samples=1).item()
                next_token.append(idx[tok].item())
            else:
                next_token.append(torch.argmax(last).item())
            if eot_token_id is not None and next_token[-1] == eot_token_id:
                break
            prompt = torch.cat((prompt, torch.tensor([[next_token[-1]]], dtype=torch.long, device=prompt.device)), dim=1)
    return next_token
        
        