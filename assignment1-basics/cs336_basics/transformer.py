"""attention 与 Transformer 骨架。逻辑由学生填写（handout §3.4–3.5）。"""
from __future__ import annotations

import torch
from torch import Tensor
from cs336_basics.nn import softmax, Linear, Embedding, SwiGLU, RMSNorm

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freqs = theta ** (-torch.arange(0, d_k/2) * 2 / d_k)
        angles = torch.arange(max_seq_len)[:, None] * freqs[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        a = x[..., 0::2]
        b = x[..., 1::2]
        c = self.cos[token_positions]
        s_ = self.sin[token_positions]
        a2 = a * c - b * s_
        b2 = a * s_ + b * c
        return torch.stack((a2, b2), dim=-1).flatten(-2)


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    scores = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=Q.dtype))
    scores = scores.masked_fill(~mask, float('-inf')) if mask is not None else scores
    return softmax(scores, dim=-1) @ V

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.h = num_heads
        self.dh = d_model // num_heads
        self.dk = d_model // num_heads
        self.RoPE = RotaryPositionalEmbedding(theta=10000, d_k=self.dh, max_seq_len=2048)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        B, s, d = x.shape
        q = self.q_proj(x).view(B, s, self.h, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, s, self.h, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, s, self.h, self.dh).transpose(1, 2)
        if token_positions is not None:
            pos = token_positions.unsqueeze(-2)
            q = self.RoPE(q, pos)
            k = self.RoPE(k, pos)
        mask = ~torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).reshape(B, s, d)
        return self.output_proj(out)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn.forward(self.ln1.forward(x), token_positions)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
            

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = self.token_embeddings(token_ids)
        token_positions = torch.arange(token_ids.shape[-1]) if token_positions is None else token_positions
        for layer in self.layers:
            x = layer.forward(x, token_positions)
        x = self.ln_final.forward(x)
        return self.lm_head.forward(x)