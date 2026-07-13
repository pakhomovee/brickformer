"""Tokenize a BrickNet split (.npz of graphs) into a packed uint16 token stream (.bin).

We tokenize natively from the graphs, so only the compact `*.npz` files are needed -- never the
large pre-serialized `paths_*.jsonl` text. Every connected component of every graph becomes one
`BOS ... EOS` sequence; sequences are concatenated into one stream and written as uint16 (the
25,157 vocab fits). A sidecar `<out>.meta.json` records counts AND the observed max of each
variable-size field, so you can confirm the vocab caps are big enough for a corpus (e.g. the full
`pt` split) -- any structure that exceeds a cap is skipped and counted rather than crashing.

Memory & speed: the split's arrays are memory-mapped and graphs are built one at a time from
array slices (peak RAM ~= a single graph, independent of split size -- so the full `pt` split
prepares on a small box), and the graph-index range is sharded across `--workers` processes for
near-linear speedup. This replaces `bricknet.load_graphs`, which materializes the whole split in
RAM at once and OOMs on large splits.

    python -m lego_tf.bnet.prepare_data --split data/pt.npz --out data/pretrain.bin
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import zipfile
from multiprocessing import Pool

import numpy as np
from bricknet.graph import _graph_from_arrays  # per-graph builder (same one load_graphs uses)

from lego_tf.bnet import trees as T
from lego_tf.bnet.tokenizer import Vocab, encode_tree

_FIELDS = ("ptr", "psub", "csub", "pconn", "cconn", "slide_abs")


def _extract_mmap(split: str, workdir: str) -> list[str]:
    """Extract the (STORED, uncompressed) .npy members so they can be memory-mapped. Returns the
    member base-names. This is a one-time byte copy -- no decompression, no full load into RAM."""
    with zipfile.ZipFile(split) as z:
        names = z.namelist()
        z.extractall(workdir)
    return [os.path.splitext(n)[0] for n in names]


def _open_mmap(workdir: str, names: list[str]) -> dict:
    return {k: np.load(os.path.join(workdir, k + ".npy"), mmap_mode="r") for k in names}


def _iter_graphs(a: dict, lo: int, hi: int):
    """Yield graphs [lo, hi) built from CSR array slices -- identical to load_graphs per graph."""
    pids, colors = a["part_ids"], a["colors"]
    kind, idx = a["edge_kind"], a["edge_idx"]
    yaw, flip, rot = a["edge_yaw"], a["edge_flip"], a["edge_rot"]
    nptr, eptr = a["node_ptr"], a["edge_ptr"]
    mats = a.get("transforms")
    for i in range(lo, hi):
        ns, ne = int(nptr[i]), int(nptr[i + 1])
        es, ee = int(eptr[i]), int(eptr[i + 1])
        m = np.asarray(mats[ns:ne]) if mats is not None else None
        yield _graph_from_arrays(
            np.asarray(pids[ns:ne]), np.asarray(colors[ns:ne]),
            np.asarray(kind[es:ee]), np.asarray(idx[es:ee]),
            np.asarray(yaw[es:ee]), np.asarray(flip[es:ee]), np.asarray(rot[es:ee]), m)


def _track(fm: dict, tree) -> None:
    for e in tree.edges:
        fm["ptr"] = max(fm["ptr"], e.child - e.parent)
        fm["psub"] = max(fm["psub"], int(e.parent_sub))
        fm["csub"] = max(fm["csub"], int(e.child_sub))
        fm["pconn"] = max(fm["pconn"], int(e.parent_conn))
        fm["cconn"] = max(fm["cconn"], int(e.child_conn))
        if hasattr(e, "slide"):
            fm["slide_abs"] = max(fm["slide_abs"], abs(int(e.slide)))


def _encode_range(a: dict, lo: int, hi: int, out_bin: str, seed: int, collision_free: bool) -> dict:
    vocab = Vocab()
    fm = {k: 0 for k in _FIELDS}
    nt = ns = nb = sk = 0
    with open(out_bin, "wb") as f:
        for g in _iter_graphs(a, lo, hi):
            for c in range(len(g.components)):
                try:
                    tree = T.sample_tree(g, component=c, seed=seed, collision_free=collision_free)
                    toks = encode_tree(tree, vocab)
                except Exception:
                    sk += 1
                    continue
                _track(fm, tree)
                np.asarray(toks, dtype=np.uint16).tofile(f)
                nt += len(toks)
                nb += len(tree.parts)
                ns += 1
    return {"n_tokens": nt, "n_seqs": ns, "n_bricks": nb, "skipped": sk, "field_max": fm}


def _worker(args) -> dict:
    workdir, names, lo, hi, out_bin, seed, collision_free = args
    a = _open_mmap(workdir, names)
    return _encode_range(a, lo, hi, out_bin, seed, collision_free)


def prepare(split: str, out: str, *, seed: int = 0, collision_free: bool = True,
            limit: int | None = None, workers: int | None = None) -> dict:
    vocab = Vocab()
    assert vocab.total < 65536, "vocab exceeds uint16"
    workers = workers or os.cpu_count() or 1

    workdir = tempfile.mkdtemp(prefix="bnet_prep_", dir=os.path.dirname(os.path.abspath(out)) or ".")
    t0 = time.time()
    try:
        names = _extract_mmap(split, workdir)
        a = _open_mmap(workdir, names)
        n_all = len(a["node_ptr"]) - 1
        n = n_all if limit is None else min(limit, n_all)
        workers = max(1, min(workers, n))
        print(f"{n} graphs | {workers} worker(s) | mmap peak RAM ~= one graph")

        # contiguous index shards -> one .bin part each
        bounds = [round(i * n / workers) for i in range(workers + 1)]
        parts = [f"{out}.part{i}" for i in range(workers)]
        jobs = [(workdir, names, bounds[i], bounds[i + 1], parts[i], seed, collision_free)
                for i in range(workers)]

        if workers == 1:
            results = [_worker(jobs[0])]
        else:
            with Pool(workers) as pool:
                results = pool.map(_worker, jobs)

        # merge stats + concatenate part files in shard order
        n_tokens = sum(r["n_tokens"] for r in results)
        n_seqs = sum(r["n_seqs"] for r in results)
        n_bricks = sum(r["n_bricks"] for r in results)
        skipped = sum(r["skipped"] for r in results)
        field_max = {k: max(r["field_max"][k] for r in results) for k in _FIELDS}
        with open(out, "wb") as dst:
            for p in parts:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, dst, length=1 << 22)
                os.remove(p)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    caps = {"ptr": vocab.size_of["PTR"], "psub": vocab.size_of["PSUB"], "csub": vocab.size_of["CSUB"],
            "pconn": vocab.size_of["PCONN"], "cconn": vocab.size_of["CCONN"]}
    meta = {"split": split, "vocab_size": vocab.total, "n_tokens": n_tokens,
            "n_seqs": n_seqs, "n_bricks": n_bricks, "skipped": skipped,
            "tokens_per_brick": round(n_tokens / max(n_bricks, 1), 2),
            "field_max": field_max, "field_caps": caps}
    if skipped:
        print(f"WARNING: skipped {skipped} structures exceeding a vocab cap; field_max={field_max}")
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {n_tokens / 1e6:.2f}M tokens ({n_seqs} seqs) -> {out}  [{time.time() - t0:.0f}s]")
    print(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="path to a split .npz (graphs)")
    ap.add_argument("--out", required=True, help="output .bin path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-order", action="store_true", help="random build order (default: collision-free)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: all CPU cores)")
    a = ap.parse_args()
    prepare(a.split, a.out, seed=a.seed, collision_free=not a.random_order,
            limit=a.limit, workers=a.workers)


if __name__ == "__main__":
    main()
