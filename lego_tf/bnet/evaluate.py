"""Evaluate a trained checkpoint: generate builds and measure the plan's §8 headline metrics.

Loads a checkpoint, samples N grammar-constrained builds, and reports:
  - validity           -- fraction that decode to a non-empty build (grammar-constrained -> ~100%)
  - collision-free      -- fraction whose whole build places with no interpenetration
  - collision-free horizon -- longest collision-free placement prefix per build (BrickNet's
                          implied bar is ~20 steps; this is the number to beat)
  - per-step collision curve -- collision rate vs sequence position (the architecture signal;
                          saved to <out>/collision_curve.csv)
  - brick-count distribution + unforced-EOS rate

Collision metrics need the inset collision meshes (from BrickNet's inset.tar.xz). Point bricknet
at them with BRICKNET_DATA=<dir containing inset/> or `python -m bricknet fetch-meshes`; without
them the harness still reports validity / brick counts (parse-only) and says so.

    python -m lego_tf.bnet.evaluate --ckpt runs/pretrain-25M/best.pt --n 256 --export runs/eval
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

import bricknet
import torch

from lego_tf.bnet.model import LegoGPT, ModelConfig
from lego_tf.bnet.tokenizer import Vocab, decode

try:
    from bricknet.score import check_tree, collision_free_prefix
    _HAVE_SCORE = True
except Exception:                                    # pragma: no cover
    _HAVE_SCORE = False


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = LegoGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def evaluate(ckpt: str, *, n: int = 256, device: str | None = None, seed: int = 0,
             greedy: bool = False, min_bricks: int = 2, max_new: int | None = None,
             collision: bool = True, export: str | None = None, export_n: int = 16,
             curve_len: int = 128, batch_size: int = 64, collision_free: bool = False,
             max_retries: int = 8, temperature: float = 1.0) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    vocab = Vocab()
    model, cfg = load_model(ckpt, device)
    cap = max_new or cfg.max_seq
    mode = (f"collision-free decoding (reject+resample, max_retries={max_retries}, temp={temperature})"
            if collision_free else ("greedy" if greedy else "temp=1"))
    print(f"loaded {ckpt} | {model.num_params()/1e6:.1f}M params | device={device} | "
          f"sampling {n} builds (max_new={cap}, batch_size={batch_size}, {mode})")

    do_coll = collision and _HAVE_SCORE
    meshes_missing = False
    trees, n_bricks, natural_eos = [], [], 0
    valid = graph_ok = 0
    horizons, coll_free = [], 0
    coll_hits = [0] * curve_len          # samples colliding exactly at position k
    coll_seen = [0] * curve_len          # samples that reached position k

    t0 = time.time()
    if collision_free:
        streams = model.generate_batch_cf(vocab, n, max_new=cap, device=device,
                                          min_bricks=min_bricks, batch_size=batch_size,
                                          max_retries=max_retries, temperature=temperature)
    else:
        streams = model.generate_batch(vocab, n, max_new=cap, device=device, greedy=greedy,
                                       min_bricks=min_bricks, batch_size=batch_size)
    print(f"  generated {n} builds in {time.time()-t0:.0f}s; scoring...")
    for i, toks in enumerate(streams):
        natural_eos += int(toks[-1] == vocab.EOS)
        try:
            tree = decode(toks, vocab)
        except Exception:
            continue
        nb = len(tree.parts)
        if nb == 0:
            continue
        valid += 1
        n_bricks.append(nb)
        trees.append(tree)

        # connector-validity: does the build realize as geometry? (grammar guarantees structure,
        # not that each (part, connector) pair physically exists -- untrained models emit bad ones)
        try:
            bricknet.tree_to_graph(tree)
        except Exception:
            continue
        graph_ok += 1

        if do_coll:
            try:
                bad = set(check_tree(tree))
                horizon = collision_free_prefix(tree)
            except FileNotFoundError:
                meshes_missing = True
                do_coll = False
            except Exception:
                continue                       # unscoreable geometry -> excluded from collision stats
            else:
                horizons.append(horizon)
                coll_free += int(len(bad) == 0)
                for k in range(min(nb, curve_len)):
                    coll_seen[k] += 1
                    coll_hits[k] += int(k in bad)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} scored | {time.time()-t0:.0f}s")

    m = len(n_bricks)
    rep: dict = {"ckpt": ckpt, "n_requested": n, "n_valid": valid,
                 "validity_rate": round(valid / n, 4) if n else 0.0,
                 "connector_valid_rate": round(graph_ok / valid, 4) if valid else 0.0,
                 "unforced_eos_rate": round(natural_eos / n, 4) if n else 0.0}
    if m:
        rep["bricks"] = {"mean": round(sum(n_bricks) / m, 1), "median": int(st.median(n_bricks)),
                         "min": min(n_bricks), "max": max(n_bricks)}
    if horizons:
        rep["collision_free_build_rate"] = round(coll_free / m, 4)
        rep["horizon"] = {"mean": round(sum(horizons) / len(horizons), 1),
                          "median": int(st.median(horizons)),
                          "ge20": round(sum(h >= 20 for h in horizons) / len(horizons), 4),
                          "ge50": round(sum(h >= 50 for h in horizons) / len(horizons), 4),
                          "ge100": round(sum(h >= 100 for h in horizons) / len(horizons), 4)}
        rep["collision_curve"] = [round(coll_hits[k] / coll_seen[k], 4)
                                  for k in range(curve_len) if coll_seen[k] > 0]
    elif not _HAVE_SCORE:
        rep["collision"] = "unavailable (bricknet.score not importable)"
    elif meshes_missing or not collision:
        rep["collision"] = ("skipped: inset meshes not found -- set BRICKNET_DATA=<dir with inset/> "
                            "or run `python -m bricknet fetch-meshes` (parse-only metrics shown)")

    if export:
        os.makedirs(export, exist_ok=True)
        written = 0
        for tree in trees:
            if written >= export_n:
                break
            try:
                ldr = bricknet.graph_to_ldr(bricknet.tree_to_graph(tree))
            except Exception:
                continue                       # skip builds that don't realize as geometry
            with open(os.path.join(export, f"sample_{written:03d}.ldr"), "w") as f:
                f.write(ldr)
            written += 1
        if written:
            rep["exported_ldr"] = written
        if rep.get("collision_curve"):
            with open(os.path.join(export, "collision_curve.csv"), "w") as f:
                f.write("step,collision_rate\n")
                for k, r in enumerate(rep["collision_curve"]):
                    f.write(f"{k},{r}\n")
        with open(os.path.join(export, "eval.json"), "w") as f:
            json.dump(rep, f, indent=2)

    _print_report(rep, time.time() - t0)
    return rep


def _print_report(rep: dict, secs: float) -> None:
    print("\n===== evaluation =====")
    print(f"valid builds: {rep['n_valid']}/{rep['n_requested']} ({rep['validity_rate']:.1%}) "
          f"| connector-valid: {rep['connector_valid_rate']:.1%} of them "
          f"| unforced-EOS: {rep['unforced_eos_rate']:.1%}")
    if "bricks" in rep:
        b = rep["bricks"]
        print(f"bricks/build: mean {b['mean']} median {b['median']} range [{b['min']}, {b['max']}]")
    if "horizon" in rep:
        h = rep["horizon"]
        print(f"collision-free builds: {rep['collision_free_build_rate']:.1%}")
        print(f"collision-free horizon: mean {h['mean']} median {h['median']} steps "
              f"(BrickNet bar ~20)  | ≥20: {h['ge20']:.0%}  ≥50: {h['ge50']:.0%}  ≥100: {h['ge100']:.0%}")
        cc = rep["collision_curve"]
        marks = [p for p in (10, 20, 50, 100) if p < len(cc)]
        print("per-step collision rate: " + "  ".join(f"@{p}={cc[p]:.0%}" for p in marks))
    elif "collision" in rep:
        print(f"collision: {rep['collision']}")
    if "exported_ldr" in rep:
        print(f"exported {rep['exported_ldr']} .ldr builds for visual inspection")
    print(f"[{secs:.0f}s]\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="path to best.pt / last.pt")
    ap.add_argument("--n", type=int, default=256, help="number of builds to sample")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling (low diversity)")
    ap.add_argument("--min-bricks", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=None, help="cap tokens/build (default: model ctx)")
    ap.add_argument("--no-collision", action="store_true", help="parse-only; skip mesh collision scoring")
    ap.add_argument("--export", default=None, help="dir to write sample .ldr + eval.json + curve.csv")
    ap.add_argument("--export-n", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64, help="builds generated in parallel on the GPU")
    ap.add_argument("--collision-free", action="store_true",
                    help="collision-aware decoding: reject+resample colliding bricks so every build "
                         "is collision-free by construction (needs inset meshes; sampling only)")
    ap.add_argument("--max-retries", type=int, default=8, help="resample attempts per brick before ending the build")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (collision-free mode)")
    a = ap.parse_args()
    evaluate(a.ckpt, n=a.n, device=a.device, seed=a.seed, greedy=a.greedy,
             min_bricks=a.min_bricks, max_new=a.max_new, collision=not a.no_collision,
             export=a.export, export_n=a.export_n, batch_size=a.batch_size,
             collision_free=a.collision_free, max_retries=a.max_retries, temperature=a.temperature)


if __name__ == "__main__":
    main()
