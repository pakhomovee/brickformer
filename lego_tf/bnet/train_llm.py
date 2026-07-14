"""Track B trainer: LoRA fine-tune a pretrained backbone on (caption -> native LEGO tokens).

Same data as the caption SFT (a graphs split tokenized to native LEGO tokens + a captions jsonl), but
the model is a `LegoLLM` (Qwen2.5-0.5B + LoRA + a warm-started LEGO embedding/head, see
`llm_backbone.py`). The caption is a text prefix in the backbone's own tokenizer; classifier-free
guidance is trained by dropping it to the null template (`--cond-drop`). Loss is next-token CE over
the LEGO vocab on the LEGO positions only.

    python -m lego_tf.bnet.train_llm --split data/sft.npz --captions data/captions_sft.jsonl \
        --out runs/llm-qwen0.5b --vehicles-only --max-iters 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from lego_tf.bnet.captions import load_captions
from lego_tf.bnet.llm_backbone import DEFAULT_BACKBONE, LegoLLM
from lego_tf.bnet.tokenizer import Vocab
from lego_tf.bnet.train_sft import VEHICLE_WORDS, _keyword_filter, _tokenize_split, lr_at


def _make_batch(idx, seqs, cap_tok, null_ids, pad_id, vocab, cond_drop, device, rng):
    B = len(idx)
    prefixes = []
    for i in idx:
        toks = cap_tok.get(i) or []
        if not toks or rng.random() < cond_drop:
            prefixes.append(list(null_ids))
        else:
            prefixes.append(list(toks[rng.integers(len(toks))]))
    Cmax = max(len(p) for p in prefixes)
    Tmax = max(len(seqs[i]) for i in idx)
    prefix_ids = np.full((B, Cmax), pad_id, np.int64)
    prefix_mask = np.zeros((B, Cmax), np.int64)
    lego = np.full((B, Tmax), vocab.PAD, np.int64)
    for r, (i, p) in enumerate(zip(idx, prefixes)):
        prefix_ids[r, Cmax - len(p):] = p               # LEFT-pad the caption
        prefix_mask[r, Cmax - len(p):] = 1
        s = seqs[i]
        lego[r, :len(s)] = s
    t = lambda a: torch.from_numpy(a).to(device)        # noqa: E731
    return t(prefix_ids), t(prefix_mask), t(lego)


def train(split, captions_jsonl, out, backbone=DEFAULT_BACKBONE, ctx=1024, batch=8, lr=2e-4,
          min_lr=2e-5, max_iters=2000, warmup=None, cond_drop=0.1, weight_decay=0.0, lora_r=16,
          device=None, seed=0, limit=None, val_frac=0.02, eval_every=250, save_every=500,
          vehicles_only=False, keywords=None):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    warmup = warmup if warmup is not None else max(50, max_iters // 20)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    vocab = Vocab()

    seqs = _tokenize_split(split, vocab, limit=limit)
    caps = load_captions(captions_jsonl)
    usable = [i for i in range(len(seqs)) if seqs[i] is not None and len(seqs[i]) >= 2 and caps.get(i)]

    kw = VEHICLE_WORDS if vehicles_only else ([k.strip() for k in keywords.split(",")] if keywords else None)
    if kw:
        keep = _keyword_filter(captions_jsonl, kw)
        before = len(usable)
        usable = [i for i in usable if i in keep]
        print(f"caption filter ({'vehicles' if vehicles_only else 'keywords'}): {before} -> {len(usable)}")
    rng.shuffle(usable)
    n_val = max(1, int(len(usable) * val_frac)) if len(usable) > 50 else 0
    val_idx, train_idx = usable[:n_val], usable[n_val:]
    print(f"usable graphs: {len(usable)} (train {len(train_idx)}, val {len(val_idx)})")

    print(f">> building {backbone} + LoRA(r={lora_r}) + warm-started LEGO embedding ...")
    model = LegoLLM.build(backbone, ctx=ctx, lora_r=lora_r, device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    null_ids = model.null_ids.tolist()
    cap_tok = {i: [tok(c, add_special_tokens=False)["input_ids"] for c in caps[i]] for i in usable}
    print(f"device={device} trainable params={model.num_params()/1e6:.1f}M "
          f"backbone={backbone} cond_drop={cond_drop} max_iters={max_iters} batch={batch}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    amp = lambda: torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                 enabled=(device == "cuda"))                       # noqa: E731

    def run_batch(pool, train_mode=True):
        idx = [pool[j] for j in rng.integers(0, len(pool), size=batch)]
        drop = cond_drop if train_mode else 0.0
        px, pm, lego = _make_batch(idx, seqs, cap_tok, null_ids, pad_id, vocab, drop, device, rng)
        with amp():
            _, loss = model(px, pm, lego, targets=lego)
        return loss

    @torch.no_grad()
    def val_loss():
        if not val_idx:
            return float("nan")
        model.eval()
        ls = [run_batch(val_idx, train_mode=False).item()
              for _ in range(min(20, max(1, len(val_idx) // batch + 1)))]
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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % 50 == 0 or it == max_iters - 1:
            dt = time.time() - t0
            print(f"iter {it:5d}/{max_iters} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {(it+1)/max(dt,1e-9):.2f} it/s")
        if (it % eval_every == 0 and it > 0) or it == max_iters - 1:
            vl = val_loss()
            print(f"  >> val loss {vl:.4f}")
            if vl < best:
                best = vl
                model.save(os.path.join(out, "best"))
    model.save(os.path.join(out, "last"))
    with open(os.path.join(out, "train_llm.json"), "w") as f:
        json.dump({"backbone": backbone, "trainable_M": model.num_params() / 1e6,
                   "max_iters": max_iters, "batch": batch, "cond_drop": cond_drop,
                   "best_val": best, "minutes": (time.time() - t0) / 60}, f, indent=2)
    print(f"done in {(time.time()-t0)/60:.1f}m; checkpoints in {out}/best and {out}/last")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="graphs .npz to fine-tune on (e.g. data/sft.npz)")
    ap.add_argument("--captions", required=True, help="captions jsonl for that split")
    ap.add_argument("--out", default="runs/llm-qwen0.5b")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap graphs (debug)")
    ap.add_argument("--vehicles-only", action="store_true", help="train only on vehicle captions")
    ap.add_argument("--keywords", default=None, help="comma-separated caption keywords to keep")
    a = ap.parse_args()
    train(a.split, a.captions, a.out, backbone=a.backbone, ctx=a.ctx, batch=a.batch, lr=a.lr,
          max_iters=a.max_iters, cond_drop=a.cond_drop, lora_r=a.lora_r, eval_every=a.eval_every,
          device=a.device, limit=a.limit, vehicles_only=a.vehicles_only, keywords=a.keywords)


if __name__ == "__main__":
    main()
