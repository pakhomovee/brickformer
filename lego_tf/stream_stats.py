"""Report flat-stream stats over real models: tokens/brick, vocab size, segment sizes.

Usage: python -m lego_tf.stream_stats data/samples/*.mpd
"""

from __future__ import annotations

import os
import sys

from lego_tf.data.ldraw import flatten_file
from lego_tf.data.parts import PartLibrary, ConnectorExtractor
from lego_tf.tokenize.absolute import PartVocab
from lego_tf.tokenize.attach import AttachTokenizer
from lego_tf.tokenize.rotations import RotationCodebook
from lego_tf.tokenize.stream import ColorVocab, build_vocab, encode_stream

LDRAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "ldraw_lib", "ldraw")


def main(paths):
    ext = ConnectorExtractor(PartLibrary(LDRAW_ROOT))
    cb = RotationCodebook()

    # shared corpus vocab across all models
    all_bricks, seqs, all_parts, all_colors = [], [], PartVocab(), ColorVocab()
    per_model = []
    for p in paths:
        bricks = [b for b in flatten_file(p) if _canon(cb, b)]
        if not bricks:
            continue
        for b in bricks:
            all_parts.add(b.part); all_colors.add(int(b.color))
        tok = AttachTokenizer(all_parts, ext)
        seq = tok.encode(bricks)
        seqs.append(seq)
        per_model.append((os.path.basename(p), bricks, seq))

    vocab = build_vocab(seqs, all_parts, all_colors)
    print(f"corpus vocab size = {vocab.size}")
    for seg, (start, n) in vocab.segments.items():
        print(f"  {seg:8s} {n:6d}  [{start}..{start+n})")
    print()
    print(f"{'model':22s} {'bricks':>7s} {'tokens':>7s} {'tok/brick':>9s}")
    tot_b = tot_t = 0
    tok = AttachTokenizer(all_parts, ext)
    for name, bricks, seq in per_model:
        ids = encode_stream(seq, all_parts, all_colors, vocab)
        tot_b += len(bricks); tot_t += len(ids)
        print(f"{name:22s} {len(bricks):7d} {len(ids):7d} {len(ids)/len(bricks):9.2f}")
    print(f"{'TOTAL':22s} {tot_b:7d} {tot_t:7d} {tot_t/tot_b:9.2f}")


def _canon(cb, b):
    try:
        cb.id_of(b.rot); return True
    except KeyError:
        return False


if __name__ == "__main__":
    main(sys.argv[1:])
