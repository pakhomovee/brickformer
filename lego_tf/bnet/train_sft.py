"""Caption-conditioned SFT: fine-tune a pretrained LEGO transformer on (caption -> build) pairs.

Loads a pretrained checkpoint (unconditional), adds the caption-prefix conditioning params, and
fine-tunes on a graphs split paired with precomputed caption embeddings (see captions.py). Classifier
-free guidance is trained by randomly dropping the caption (`--cond-drop`) to the learned null prefix.

    # 1. precompute embeddings once (GPU box):
    python -m lego_tf.bnet.captions --split data/sft.npz --captions data/captions_sft.jsonl --out data/sft
    # 2. fine-tune from the pretrained checkpoint:
    python -m lego_tf.bnet.train_sft --split data/sft.npz --caps data/sft --init weights/brickformer_25M_v0_fix.pt \
        --out runs/sft-25M --max-iters 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from lego_tf.bnet import trees as T
from lego_tf.bnet.model import LegoGPT, ModelConfig
from lego_tf.bnet.tokenizer import Vocab, encode_tree

IGNORE = -100


def _tokenize_split(split_npz: str, vocab: Vocab, limit: int | None = None):
    """Per-graph LEGO token sequences (component 0), aligned to graph order; None where it fails."""
    import bricknet
    graphs = bricknet.load_graphs(split_npz)
    if limit:
        graphs = graphs[:limit]
    seqs: list[np.ndarray | None] = []
    for g in graphs:
        try:
            seqs.append(np.asarray(encode_tree(T.sample_tree(g, seed=0), vocab), dtype=np.int64))
        except Exception:
            seqs.append(None)
    return seqs


def _load_caps(caps_prefix: str):
    meta = json.load(open(caps_prefix + ".capmap.json"))
    emb = np.memmap(caps_prefix + ".capemb.f16", dtype=np.float16, mode="r").reshape(-1, meta["cond_dim"])
    return emb, meta["graph_caps"], meta["cond_dim"]


def _init_model(init_ckpt: str, cond_dim: int, ctx: int, device: str):
    """Build a conditional model from a pretrained (unconditional) checkpoint: keep its size/weights,
    add the caption params (random init)."""
    ck = torch.load(init_ckpt, map_location=device)
    cfg_d = dict(ck["cfg"])
    cfg_d["cond_dim"] = cond_dim
    cfg_d["max_seq"] = ctx
    model = LegoGPT(ModelConfig(**cfg_d)).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    new = [k for k in missing if k.startswith(("cond_", "null_"))]
    print(f"loaded {init_ckpt} | added caption params: {new} | other missing: "
          f"{[m for m in missing if m not in new]} | unexpected: {list(unexpected)}")
    return model, cfg_d


def _batch(seqs, rows, emb, cond_dim, idx, pad_id, device, rng):
    L = max(len(seqs[i]) for i in idx)
    B = len(idx)
    x = np.full((B, L - 1), pad_id, np.int64)
    y = np.full((B, L - 1), IGNORE, np.int64)
    cond = np.zeros((B, 1, cond_dim), np.float32)
    for r, i in enumerate(idx):
        s = seqs[i]
        x[r, :len(s) - 1] = s[:-1]
        y[r, :len(s) - 1] = s[1:]
        cond[r, 0] = emb[rng.choice(rows[i])]            # random caption for this graph
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    cond = torch.from_numpy(cond).to(device)
    return x, y, cond


def lr_at(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    r = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (lr - min_lr)


def train(split, caps, init, out, ctx=1024, batch=64, lr=1e-4, min_lr=1e-5, max_iters=2000,
          warmup=None, cond_drop=0.1, weight_decay=0.1, device=None, seed=0, limit=None,
          val_frac=0.02, eval_every=250):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    warmup = warmup if warmup is not None else max(50, max_iters // 20)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    vocab = Vocab()

    emb, graph_caps, cond_dim = _load_caps(caps)
    seqs = _tokenize_split(split, vocab, limit=limit)
    usable = [i for i in range(len(seqs))
              if i < len(graph_caps) and seqs[i] is not None and len(seqs[i]) >= 2 and graph_caps[i]]
    rng.shuffle(usable)
    n_val = max(1, int(len(usable) * val_frac)) if len(usable) > 50 else 0
    val_idx, train_idx = usable[:n_val], usable[n_val:]
    print(f"usable graphs: {len(usable)} (train {len(train_idx)}, val {len(val_idx)}) | cond_dim {cond_dim}")

    model, cfg_d = _init_model(init, cond_dim, ctx, device)
    print(f"device={device} params={model.num_params()/1e6:.1f}M cond_drop={cond_drop} "
          f"max_iters={max_iters} batch={batch}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

    def run_batch(pool, train_mode=True):
        idx = [pool[j] for j in rng.integers(0, len(pool), size=batch)]
        x, y, cond = _batch(seqs, graph_caps, emb, cond_dim, idx, vocab.PAD, device, rng)
        drop = (torch.rand(len(idx), device=device) < cond_drop) if train_mode else \
               torch.zeros(len(idx), dtype=torch.bool, device=device)
        amp = torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=(device == "cuda"))
        with amp:
            _, loss = model(x, targets=y, cond=cond, cond_drop=drop)
        return loss

    @torch.no_grad()
    def val_loss():
        if not val_idx:
            return float("nan")
        model.eval()
        ls = [run_batch(val_idx, train_mode=False).item() for _ in range(min(20, max(1, len(val_idx) // batch + 1)))]
        model.train()
        return sum(ls) / len(ls)

    t0, best = time.time(), float("inf")
    model.train()
    for it in range(max_iters):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(it, warmup, max_iters, lr, min_lr)
        opt.zero_grad(set_to_none=True)
        loss = run_batch(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 50 == 0 or it == max_iters - 1:
            dt = time.time() - t0
            print(f"iter {it:5d}/{max_iters} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {(it+1)/max(dt,1e-9):.1f} it/s")
        if (it % eval_every == 0 and it > 0) or it == max_iters - 1:
            vl = val_loss()
            print(f"  >> val loss {vl:.4f}")
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "cfg": cfg_d, "iter": it,
                            "val_loss": vl, "caps_model": json.load(open(caps + ".capmap.json"))["model"]},
                           os.path.join(out, "best.pt"))
    torch.save({"model": model.state_dict(), "cfg": cfg_d, "iter": max_iters,
                "caps_model": json.load(open(caps + ".capmap.json"))["model"]},
               os.path.join(out, "last.pt"))
    with open(os.path.join(out, "sft.json"), "w") as f:
        json.dump({"params_M": model.num_params()/1e6, "max_iters": max_iters, "batch": batch,
                   "cond_drop": cond_drop, "best_val": best, "minutes": (time.time()-t0)/60}, f, indent=2)
    print(f"done in {(time.time()-t0)/60:.1f}m; checkpoints in {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="graphs .npz to fine-tune on (e.g. data/sft.npz)")
    ap.add_argument("--caps", required=True, help="caption prefix (<caps>.capemb.f16 + .capmap.json)")
    ap.add_argument("--init", required=True, help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--out", default="runs/sft-25M")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--cond-drop", type=float, default=0.1, help="fraction of steps with the caption dropped (CFG)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap graphs (debug)")
    a = ap.parse_args()
    train(a.split, a.caps, a.init, a.out, ctx=a.ctx, batch=a.batch, lr=a.lr, max_iters=a.max_iters,
          cond_drop=a.cond_drop, eval_every=a.eval_every, device=a.device, limit=a.limit)


if __name__ == "__main__":
    main()
