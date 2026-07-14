"""Frozen sentence-embedding encoder for caption conditioning (the SFT phase).

`transformers` is imported lazily so the rest of the package stays dependency-light and this file
only pulls a text model on the GPU/Colab box (not the offline CPU dev box). We pool each caption to
a single vector (the pooled-prefix MVP in model.py); precompute them once, aligned to a split's
graph order, so training and inference never run the text model in the hot loop.

    # one-time, on the GPU box (writes data/sft.capemb.f16 + data/sft.capmap.json):
    python -m lego_tf.bnet.captions --split data/sft.npz --captions data/captions_sft.jsonl \
        --out data/sft

At inference, `CaptionEncoder.encode(["a red race car"])` embeds a typed prompt the same way.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, small, fast, good general text


def load_captions(path: str) -> dict[int, list[str]]:
    """captions_*.jsonl ({"id": g, "caption": "..."} rows, possibly many per id) -> {id: [captions]}."""
    caps: dict[int, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            caps.setdefault(int(r["id"]), []).append(r.get("caption") or r["text"])
    return caps


class CaptionEncoder:
    """Frozen mean-pooled sentence encoder. `dim` is the caption-embedding size (model.cond_dim)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        import os
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")   # HF's 10s default aborts flaky links
        try:
            import hf_transfer  # noqa: F401  (robust resuming downloader, if installed)
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        except Exception:
            pass
        import torch
        from transformers import AutoModel, AutoTokenizer   # model_id may be a HF id OR a local dir
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def encode(self, texts: list[str], batch_size: int = 256, normalize: bool = True) -> np.ndarray:
        """(N, dim) float32 pooled embeddings (attention-masked mean pooling; L2-normalized)."""
        import torch
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state            # (B, T, dim)
            m = enc["attention_mask"].unsqueeze(-1).float()        # (B, T, 1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)     # masked mean
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, self.dim), np.float32)


def precompute(split_npz: str, captions_jsonl: str, out_prefix: str,
               model_id: str = DEFAULT_MODEL, device: str | None = None) -> dict:
    """Embed every caption for a split and write `<out>.capemb.f16` (all embeddings) plus
    `<out>.capmap.json` ({cond_dim, model, graph_caps}) where graph_caps[g] lists the embedding row
    indices for graph g (graph order = the split's order; caption id == graph index)."""
    import bricknet
    n_graphs = len(bricknet.load_graphs(split_npz))
    caps = load_captions(captions_jsonl)
    enc = CaptionEncoder(model_id, device)

    texts, graph_caps = [], []
    for g in range(n_graphs):
        rows = []
        for c in caps.get(g, []):
            rows.append(len(texts))
            texts.append(c)
        graph_caps.append(rows)
    emb = enc.encode(texts)                                        # (num_captions, dim)
    emb.astype(np.float16).tofile(out_prefix + ".capemb.f16")
    meta = {"cond_dim": int(enc.dim), "model": model_id, "n_graphs": n_graphs,
            "n_captions": len(texts), "graph_caps": graph_caps}
    with open(out_prefix + ".capmap.json", "w") as f:
        json.dump(meta, f)
    n_missing = sum(not gc for gc in graph_caps)
    print(f"embedded {len(texts)} captions ({enc.dim}-d) for {n_graphs} graphs "
          f"({n_missing} graphs have no caption) -> {out_prefix}.capemb.f16")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Precompute caption embeddings for SFT conditioning.")
    ap.add_argument("--split", required=True, help="the graphs .npz whose order defines graph ids")
    ap.add_argument("--captions", required=True, help="captions_*.jsonl for that split")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.capemb.f16 + .capmap.json)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    precompute(a.split, a.captions, a.out, model_id=a.model, device=a.device)


if __name__ == "__main__":
    main()
