"""Stage-0 gate: overfit a tiny model on N structures to ~0 loss, then generate a
grammar-constrained sample and confirm it decodes to a valid build (tokens -> model -> valid).

    python -m lego_tf.bnet.train_overfit --n 64 --steps 400
"""

from __future__ import annotations

import argparse
import time

import bricknet
import torch
from torch.utils.data import DataLoader

from lego_tf.bnet.dataset import SeqDataset, collate, tokenize_split, IGNORE
from lego_tf.bnet.model import LegoGPT, ModelConfig
from lego_tf.bnet.tokenizer import Vocab, decode


def run(n=64, steps=400, d_model=128, n_layers=4, n_heads=4, batch=4, lr=3e-4,
        split="data/val.npz", seed=0, device="cpu", max_len=200):
    torch.manual_seed(seed)
    vocab = Vocab()
    # pool then keep the n shortest structures under max_len (bounds CPU memory: the ~21.6k-wide
    # logits over long sequences are the hog on this box).
    pool = tokenize_split(split, vocab, limit=max(n * 6, 128), seed=seed)
    pool = sorted((s for s in pool if len(s) <= max_len), key=len)[:n]
    seqs = pool
    print(f"{len(seqs)} sequences | mean len {sum(map(len, seqs)) / len(seqs):.0f} "
          f"| max {max(map(len, seqs))}")

    ds = SeqDataset(seqs)
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    collate_fn=lambda b: collate(b, vocab.PAD))

    cfg = ModelConfig(vocab_size=vocab.total, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, max_seq=max(map(len, seqs)) + 1)
    model = LegoGPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    model.train()
    t0 = time.time()
    step = 0
    last = None
    while step < steps:
        for inp, tgt in dl:
            _, loss = model(inp.to(device), targets=tgt.to(device), ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            last = loss.item()
            if step % 50 == 0 or step == 1:
                print(f"  step {step:4d}  loss {last:.4f}  ({time.time() - t0:.0f}s)")
            if step >= steps:
                break
    print(f"final loss {last:.4f}")

    # tokens -> model -> build: constrained greedy generation must decode; scoring is best-effort
    gen = model.generate(vocab, max_new=cfg.max_seq, device=device, constrained=True)
    try:
        tree = decode(gen, vocab)
        ok = len(tree.parts) > 0
    except Exception as e:
        print("generated stream failed to decode:", type(e).__name__, e)
        return last, False
    score = "n/a"
    try:
        placed, coll, _ = bricknet.score_text(
            bricknet.graph_to_ldr(bricknet.tree_to_graph(tree)), collision=True)
        score = f"placed={placed} collisions={coll}"
    except Exception as e:
        score = f"score failed ({type(e).__name__})"
    print(f"generated: {len(gen)} tokens -> {len(tree.parts)} bricks | decoded OK | {score}")
    return last, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--split", default="data/val.npz")
    a = ap.parse_args()
    run(n=a.n, steps=a.steps, d_model=a.d_model, n_layers=a.layers,
        n_heads=a.heads, batch=a.batch, split=a.split, max_len=a.max_len)


if __name__ == "__main__":
    main()
