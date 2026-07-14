"""Decoder-only transformer for the native LEGO token stream (LLaMA-style: RoPE, RMSNorm, SwiGLU).

Small and dependency-light -- sized by config so the same code runs a tiny CPU overfit smoke test
here and the 25M / 150-250M models on a GPU. Includes grammar-constrained greedy generation so a
sampled stream always decodes to a valid build.
"""

from __future__ import annotations

import math
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
    use_pose: bool = False       # v1: add each token's delayed resolved-pose feature to its embedding
    pose_dim: int = 9            # translation (3) + rotation 6D (6); see trees.pose_feature_rows
    pose_fourier_k: int = 8      # Fourier frequencies for the position channel

    def ff(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        return 64 * round(8 * self.d_model / 3 / 64) or self.d_model * 2


class PoseEmbed(nn.Module):
    """Resolved world pose (translation + 6D rotation) -> d_model, added to token embeddings.
    Position is Fourier-featured over geometric wavelengths (~4..1024 LDU) so absolute placement
    and overlap are directly attention-visible; rotation 6D passes through an MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.k = cfg.pose_fourier_k
        wavelengths = torch.logspace(math.log10(4.0), math.log10(1024.0), self.k)
        self.register_buffer("freqs", 2 * math.pi / wavelengths, persistent=False)  # (K,)
        in_dim = 3 * 2 * self.k + 6   # sin/cos of 3 coords x K freqs, + 6D rotation
        self.mlp = nn.Sequential(nn.Linear(in_dim, cfg.d_model), nn.SiLU(),
                                 nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, pose):          # pose: (..., 9) = [tx,ty,tz, r00,r10,r20, r01,r11,r21]
        pos, rot = pose[..., :3], pose[..., 3:]
        ang = pos[..., None] * self.freqs.to(pose.dtype)         # (..., 3, K)
        fourier = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)  # (..., 6K)
        return self.mlp(torch.cat([fourier, rot], dim=-1))


def _default_resolve_pose(vocab):
    """Callable: partial token stream -> pose feature of its last completed brick (zeros on any
    failure). Used during v1 generation to feed each new brick the previous brick's pose."""
    from lego_tf.bnet import trees as T
    from lego_tf.bnet.tokenizer import decode
    import numpy as np
    zeros = np.zeros(T.POSE_DIM, dtype=np.float32)

    def resolve(tokens):
        try:
            poses = T.resolve_poses(decode(tokens, vocab))
            return zeros if poses is None or len(poses) == 0 else T._pose_feat(poses[-1])
        except Exception:
            return zeros

    return resolve


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
        self.pose_embed = PoseEmbed(cfg) if cfg.use_pose else None
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

    def forward(self, ids, targets=None, ignore_index=-100, pose=None):
        dh = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _rope_cache(ids.shape[1], dh, self.cfg.rope_base, ids.device)
        x = self.tok(ids)
        if self.pose_embed is not None and pose is not None:
            x = x + self.pose_embed(pose)
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss

    @torch.no_grad()
    def _step_logits(self, ids, pose=None):
        """Logits for the LAST position only -- (B, vocab). Applies the head to just the final
        hidden state, avoiding the (B, T, vocab) tensor that dominates memory during generation."""
        dh = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _rope_cache(ids.shape[1], dh, self.cfg.rope_base, ids.device)
        x = self.tok(ids)
        if self.pose_embed is not None and pose is not None:
            x = x + self.pose_embed(pose)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x[:, -1:, :]))[:, -1, :]

    @torch.no_grad()
    def _gathered_logits(self, ids, gidx, pose=None):
        """Logits at each row's own position `gidx[r]` -- (B, vocab). Lets rows be right-padded to
        different real lengths (needed when rows roll back independently) without materializing the
        (B, T, vocab) tensor: gather the hidden state at gidx, then apply the head."""
        dh = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _rope_cache(ids.shape[1], dh, self.cfg.rope_base, ids.device)
        x = self.tok(ids)
        if self.pose_embed is not None and pose is not None:
            x = x + self.pose_embed(pose)
        for b in self.blocks:
            x = b(x, cos, sin)
        h = x[torch.arange(ids.shape[0], device=ids.device), gidx]   # (B, D) -- each row's last real token
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate_batch(self, vocab: Vocab, n: int, max_new: int | None = None, device="cpu",
                       greedy: bool = False, min_bricks: int = 1, batch_size: int = 64,
                       resolve_pose=None):
        """Grammar-constrained generation of `n` builds in parallel (per-row GrammarState), in
        chunks of `batch_size`. Returns a list of `n` token streams (each starts with BOS).
        Each row masks logits to its own grammar; finished rows emit PAD until the chunk ends.

        For a v1 (use_pose) model, `resolve_pose(tokens) -> (POSE_DIM,) float array` resolves the
        pose of the last completed brick in a partial stream; each token then carries the previous
        brick's pose (P[i-1]), matching training. Defaults to the built-in resolver."""
        self.eval()
        cap = max_new or self.cfg.max_seq
        p_lo = vocab.offset["PART"]
        p_hi = p_lo + vocab.size_of["PART"]
        use_pose = self.pose_embed is not None
        if use_pose and resolve_pose is None:
            resolve_pose = _default_resolve_pose(vocab)
        pdim = self.cfg.pose_dim
        streams: list[list[int]] = []
        for start in range(0, n, batch_size):
            B = min(batch_size, n - start)
            ids = torch.full((B, 1), vocab.BOS, dtype=torch.long, device=device)
            states = [GrammarState(vocab) for _ in range(B)]
            done = [False] * B
            parts = [0] * B
            out = [[vocab.BOS] for _ in range(B)]
            if use_pose:
                # per-token pose history (BOS row = zeros) and each row's current brick pose P[i-1]
                pose_hist = torch.zeros((B, 1, pdim), dtype=torch.float32, device=device)
                cur_pose = [[0.0] * pdim for _ in range(B)]
            for _ in range(cap):
                pose_arg = pose_hist[:, -self.cfg.max_seq:] if use_pose else None
                logits = self._step_logits(ids[:, -self.cfg.max_seq:], pose=pose_arg)   # (B, vocab)
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
                if use_pose:  # the new token belongs to the in-progress brick -> carries P[i-1]
                    step_pose = torch.tensor(cur_pose, dtype=torch.float32, device=device)
                    pose_hist = torch.cat([pose_hist, step_pose[:, None, :]], dim=1)
                for r in range(B):
                    if done[r]:
                        continue
                    t = int(nxt[r])
                    out[r].append(t)
                    if p_lo <= t < p_hi:
                        parts[r] += 1
                    prev_nb = states[r].n_bricks
                    states[r].step(t)
                    if states[r].done:
                        done[r] = True
                    elif use_pose and states[r].n_bricks > prev_nb:
                        cur_pose[r] = list(resolve_pose(out[r]))   # brick completed -> next pose
                if all(done):
                    break
            streams.extend(out)
        return streams

    @torch.no_grad()
    def generate_batch_cf(self, vocab: Vocab, n: int, max_new: int | None = None, device="cpu",
                          min_bricks: int = 1, batch_size: int = 64, max_retries: int = 8,
                          temperature: float = 1.0):
        """Collision-aware generation: the connector grammar of `generate_batch`, plus a per-build
        incremental collision scene. When a completed brick would collide with the built structure,
        its tokens are rolled back and the brick is resampled (up to `max_retries`); on exhaustion
        the build ends. Every returned stream is therefore **collision-free by construction**
        (bricknet's collision_free_prefix == n_parts).

        The scene mirrors bricknet.first_collision exactly -- same incremental CollisionScene,
        same exact-duplicate `seen` set, same fixed-parent exclusion -- so a stream that passes here
        also passes bricknet's own collision metric. Needs the inset meshes (BRICKNET_DATA). Sampling
        only: greedy cannot escape a rejected placement, so `temperature` must be > 0."""
        import copy
        import numpy as np
        import bricknet
        from bricknet.collision import CollisionScene
        from bricknet.core import FixedEdge
        from lego_tf.bnet import trees as T
        from lego_tf.bnet.tokenizer import decode

        self.eval()
        cap = max_new or self.cfg.max_seq
        max_seq = self.cfg.max_seq
        use_pose = self.pose_embed is not None
        pdim = self.cfg.pose_dim
        streams: list[list[int]] = []

        for start in range(0, n, batch_size):
            B = min(batch_size, n - start)
            buf = [[vocab.BOS] for _ in range(B)]
            gs = [GrammarState(vocab) for _ in range(B)]
            scene = [CollisionScene() for _ in range(B)]
            seen: list[set] = [set() for _ in range(B)]
            done = [False] * B
            pose_hist = [[[0.0] * pdim] for _ in range(B)] if use_pose else None
            cur_pose = [[0.0] * pdim for _ in range(B)] if use_pose else None
            # snapshot at the start of the current (in-progress) brick: (buf_len, gs_copy, attempts, pose)
            snap = [(1, copy.deepcopy(gs[r]), 0, ([0.0] * pdim if use_pose else None)) for r in range(B)]

            def finish(r):        # end a build: append EOS and keep pose_hist 1:1 with buf (v1)
                buf[r].append(vocab.EOS)
                if use_pose:
                    pose_hist[r].append(list(cur_pose[r]))
                done[r] = True

            safety = cap * (max_retries + 2)
            for _ in range(safety):
                if all(done):
                    break
                winlens = [min(len(buf[r]), max_seq) for r in range(B)]
                W = max(winlens)
                ids = torch.full((B, W), vocab.PAD, dtype=torch.long, device=device)
                gidx = torch.zeros(B, dtype=torch.long, device=device)
                posewin = torch.zeros((B, W, pdim), dtype=torch.float32, device=device) if use_pose else None
                for r in range(B):
                    w = winlens[r]
                    ids[r, :w] = torch.tensor(buf[r][-w:], dtype=torch.long, device=device)
                    gidx[r] = w - 1
                    if use_pose:
                        posewin[r, :w] = torch.tensor(pose_hist[r][-w:], dtype=torch.float32, device=device)
                last = self._gathered_logits(ids, gidx, pose=posewin)   # (B, vocab)
                mask = torch.full_like(last, float("-inf"))
                for r in range(B):
                    if done[r]:
                        allowed = [vocab.PAD]
                    else:
                        allowed = gs[r].allowed_ids()
                        if gs[r].n_bricks < min_bricks:
                            allowed = [i for i in allowed if i != vocab.EOS]
                    mask[r, allowed] = last[r, allowed]
                nxt = torch.multinomial((mask / temperature).softmax(-1), 1).squeeze(-1)

                for r in range(B):
                    if done[r]:
                        continue
                    t = int(nxt[r])
                    buf[r].append(t)
                    if use_pose:
                        pose_hist[r].append(list(cur_pose[r]))
                    prev_nb = gs[r].n_bricks
                    gs[r].step(t)
                    if gs[r].done:                       # EOS
                        done[r] = True
                        continue
                    if gs[r].n_bricks <= prev_nb:        # mid-brick token
                        if len(buf[r]) >= cap:           # length cap mid-brick -> end at the last boundary
                            blen = snap[r][0]
                            del buf[r][blen:]            # drop the partial in-progress brick
                            if use_pose:
                                del pose_hist[r][blen:]
                            finish(r)                    # EOS only ever lands at a brick boundary
                        continue

                    # a brick just completed -> resolve its world pose and test collision
                    tree = decode(buf[r], vocab)
                    try:
                        mats = bricknet.decode_graph(bricknet.tree_to_graph(tree))
                    except Exception:
                        mats = None
                    reject = mats is None or len(mats) != len(tree.parts)
                    if not reject:
                        mat = np.asarray(mats[-1], dtype=np.float64)
                        pid = tree.parts[-1].part_id
                        ci = len(tree.parts) - 1
                        edge = next((e for e in tree.edges if e.child == ci), None)
                        key = (pid, mat.tobytes())
                        exclude = edge.parent if isinstance(edge, FixedEdge) else -1
                        reject = key in seen[r] or bool(
                            scene[r].check(pid, mat, exclude=exclude, first_only=True))

                    if reject:
                        blen, gscopy, attempts, cpcopy = snap[r]
                        del buf[r][blen:]
                        if use_pose:
                            del pose_hist[r][blen:]
                            cur_pose[r] = list(cpcopy)
                        gs[r] = copy.deepcopy(gscopy)
                        attempts += 1
                        snap[r] = (blen, gscopy, attempts, cpcopy)
                        if attempts > max_retries:       # give up on this brick -> end the build
                            finish(r)
                    else:
                        scene[r].add(pid, mat)
                        seen[r].add(key)
                        if use_pose:
                            cur_pose[r] = list(T._pose_feat(mat))
                        snap[r] = (len(buf[r]), copy.deepcopy(gs[r]), 0,
                                   (list(cur_pose[r]) if use_pose else None))
                        if len(buf[r]) >= cap:
                            finish(r)
            streams.extend(buf)
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
