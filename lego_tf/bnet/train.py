"""Train the native LEGO transformer on a packed token stream (.bin).

GPU-ready: bf16 autocast on CUDA, cosine LR schedule + warmup, gradient accumulation,
checkpointing, periodic val loss. Size presets scale the same model from a CPU smoke run to the
25M ablation twin and the 150-250M main run.

    python -m lego_tf.bnet.train --train data/pretrain.bin --val data/val.bin --size 25M --out runs/pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from lego_tf.bnet.model import LegoGPT, ModelConfig
from lego_tf.bnet.tokenizer import Vocab

# preset -> (d_model, n_layers, n_heads)
SIZES = {
    "tiny": (128, 4, 4),
    "5M": (320, 6, 5),
    "25M": (512, 8, 8),
    "125M": (768, 12, 12),
    "250M": (1024, 16, 16),
}


def load_bin(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, ctx: int, batch: int, device: str):
    ix = np.random.randint(0, len(data) - ctx - 1, size=batch)
    x = np.stack([data[i:i + ctx].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + ctx].astype(np.int64) for i in ix])
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    r = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (lr - min_lr)


@torch.no_grad()
def eval_loss(model, data, ctx, batch, device, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, ctx, batch, device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(train_bin, val_bin, out, size="25M", ctx=1024, batch=32, grad_accum=1,
          lr=3e-4, min_lr=3e-5, max_iters=20000, warmup=None, eval_every=1000,
          weight_decay=0.1, device=None, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    warmup = warmup if warmup is not None else max(100, max_iters // 50)
    os.makedirs(out, exist_ok=True)

    vocab = Vocab()
    d, L, H = SIZES[size]
    cfg = ModelConfig(vocab_size=vocab.total, d_model=d, n_layers=L, n_heads=H, max_seq=ctx)
    model = LegoGPT(cfg).to(device)
    tok_per_iter = batch * grad_accum * ctx
    print(f"device={device} size={size} params={model.num_params()/1e6:.1f}M "
          f"ctx={ctx} batch={batch}x{grad_accum} => {tok_per_iter} tok/iter, "
          f"{max_iters} iters => {tok_per_iter*max_iters/1e6:.0f}M tokens")

    train_data = load_bin(train_bin)
    val_data = load_bin(val_bin) if val_bin and os.path.exists(val_bin) else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

    t0 = time.time()
    best_val = float("inf")
    for it in range(max_iters):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(it, warmup, max_iters, lr, min_lr)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, ctx, batch, device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(x, targets=y)
            (loss / grad_accum).backward()
            loss_acc += loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 50 == 0 or it == max_iters - 1:
            dt = time.time() - t0
            tps = tok_per_iter * (it + 1) / dt
            eta = (max_iters - it - 1) * tok_per_iter / max(tps, 1)
            print(f"iter {it:6d}/{max_iters} | loss {loss_acc:.4f} | lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {tps/1e3:.0f}k tok/s | ETA {eta/60:.0f}m")

        if val_data is not None and (it % eval_every == 0 and it > 0 or it == max_iters - 1):
            vl = eval_loss(model, val_data, ctx, batch, device)
            print(f"  >> val loss {vl:.4f}")
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "iter": it,
                            "val_loss": vl}, os.path.join(out, "best.pt"))

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "iter": max_iters}, os.path.join(out, "last.pt"))
    with open(os.path.join(out, "train.json"), "w") as f:
        json.dump({"size": size, "params_M": model.num_params()/1e6, "ctx": ctx,
                   "tokens": tok_per_iter*max_iters, "best_val": best_val,
                   "minutes": (time.time()-t0)/60}, f, indent=2)
    print(f"done in {(time.time()-t0)/60:.1f}m; checkpoints in {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="data/val.bin")
    ap.add_argument("--out", default="runs/pt")
    ap.add_argument("--size", default="25M", choices=list(SIZES))
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    train(a.train, a.val, a.out, size=a.size, ctx=a.ctx, batch=a.batch, grad_accum=a.grad_accum,
          lr=a.lr, max_iters=a.max_iters, eval_every=a.eval_every, device=a.device)


if __name__ == "__main__":
    main()
