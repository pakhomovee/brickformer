"""Decoder-only transformer for the native LEGO token stream (LLaMA-style: RoPE, RMSNorm, SwiGLU).

Small and dependency-light -- sized by config so the same code runs a tiny CPU overfit smoke test
here and the 25M / 150-250M models on a GPU. Includes grammar-constrained greedy generation so a
sampled stream always decodes to a valid build.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lego_tf.bnet.tokenizer import Vocab, GrammarState


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None      # defaults to ~8/3 * d_model (SwiGLU), rounded
    max_seq: int = 2048
    rope_base: float = 10000.0

    def ff(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        return 64 * round(8 * self.d_model / 3 / 64) or self.d_model * 2


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w


def _rope_cache(seq, dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ff = cfg.ff()
        self.w1 = nn.Linear(cfg.d_model, ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, ff, bias=False)
        self.w2 = nn.Linear(ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class LegoGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tie
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.tok.weight.numel()  # tied head

    def forward(self, ids, targets=None, ignore_index=-100):
        dh = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _rope_cache(ids.shape[1], dh, self.cfg.rope_base, ids.device)
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss

    @torch.no_grad()
    def _step_logits(self, ids):
        """Logits for the LAST position only -- (B, vocab). Applies the head to just the final
        hidden state, avoiding the (B, T, vocab) tensor that dominates memory during generation."""
        dh = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _rope_cache(ids.shape[1], dh, self.cfg.rope_base, ids.device)
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x[:, -1:, :]))[:, -1, :]

    @torch.no_grad()
    def generate_batch(self, vocab: Vocab, n: int, max_new: int | None = None, device="cpu",
                       greedy: bool = False, min_bricks: int = 1, batch_size: int = 64):
        """Grammar-constrained generation of `n` builds in parallel (per-row GrammarState), in
        chunks of `batch_size`. Returns a list of `n` token streams (each starts with BOS).
        Each row masks logits to its own grammar; finished rows emit PAD until the chunk ends."""
        self.eval()
        cap = max_new or self.cfg.max_seq
        p_lo = vocab.offset["PART"]
        p_hi = p_lo + vocab.size_of["PART"]
        streams: list[list[int]] = []
        for start in range(0, n, batch_size):
            B = min(batch_size, n - start)
            ids = torch.full((B, 1), vocab.BOS, dtype=torch.long, device=device)
            states = [GrammarState(vocab) for _ in range(B)]
            done = [False] * B
            parts = [0] * B
            out = [[vocab.BOS] for _ in range(B)]
            for _ in range(cap):
                logits = self._step_logits(ids[:, -self.cfg.max_seq:])          # (B, vocab)
                mask = torch.full_like(logits, float("-inf"))
                for r in range(B):
                    if done[r]:
                        allowed = [vocab.PAD]
                    else:
                        allowed = states[r].allowed_ids()
                        if parts[r] < min_bricks:
                            allowed = [i for i in allowed if i != vocab.EOS]
                    mask[r, allowed] = logits[r, allowed]
                nxt = (mask.argmax(-1) if greedy
                       else torch.multinomial(mask.softmax(-1), 1).squeeze(-1))
                ids = torch.cat([ids, nxt[:, None]], dim=1)
                for r in range(B):
                    if done[r]:
                        continue
                    t = int(nxt[r])
                    out[r].append(t)
                    if p_lo <= t < p_hi:
                        parts[r] += 1
                    states[r].step(t)
                    if states[r].done:
                        done[r] = True
                if all(done):
                    break
            streams.extend(out)
        return streams

    @torch.no_grad()
    def generate(self, vocab: Vocab, max_new: int = 4096, device="cpu",
                 constrained: bool = True, greedy: bool = True, min_bricks: int = 1):
        """Autoregressive generation from BOS. With `constrained`, logits are masked to the
        grammar's valid segment each step, so the output always decodes to a valid tree.
        `min_bricks` masks EOS until at least that many bricks are placed (the plan's EOS floor;
        also keeps an undertrained model from emitting an empty build)."""
        self.eval()
        ids = torch.tensor([[vocab.BOS]], device=device)
        g = GrammarState(vocab)
        parts = 0
        for _ in range(max_new):
            logits = self(ids[:, -self.cfg.max_seq:])[0][:, -1, :]
            if constrained:
                allowed = [i for i in g.allowed_ids() if not (i == vocab.EOS and parts < min_bricks)]
                mask = torch.full_like(logits, float("-inf"))
                mask[0, allowed] = logits[0, allowed]
                logits = mask
            nxt = int(logits.argmax(-1)) if greedy else int(torch.multinomial(logits.softmax(-1), 1))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
            if vocab.offset["PART"] <= nxt < vocab.offset["PART"] + vocab.size_of["PART"]:
                parts += 1
            if constrained:
                g.step(nxt)
                if g.done:
                    break
            elif nxt == vocab.EOS:
                break
        return ids[0].tolist()
