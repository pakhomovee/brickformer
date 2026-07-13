"""Inspect a real LDR/MPD file: flatten, report on-grid coverage, round-trip the on-grid part.

Usage: python -m lego_tf.inspect_model <file.mpd> [more.mpd ...]
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from lego_tf.data.ldraw import flatten_file, write_ldr, flatten, parse_ldr
from lego_tf.tokenize.absolute import AbsolutePoseTokenizer, PartVocab
from lego_tf.tokenize.rotations import RotationCodebook


def analyze(path: str) -> None:
    bricks = flatten_file(path)
    cb = RotationCodebook()
    n = len(bricks)

    on_grid_pos = 0
    canon_rot = 0
    both = []
    for b in bricks:
        pg = bool(np.all(np.abs(b.pos - np.round(b.pos)) < 1e-3))
        try:
            cb.id_of(b.rot)
            rg = True
        except KeyError:
            rg = False
        on_grid_pos += pg
        canon_rot += rg
        if pg and rg:
            both.append(b)

    parts = Counter(b.part for b in bricks)
    print(f"\n=== {path} ===")
    print(f"  flattened bricks : {n}")
    print(f"  unique parts     : {len(parts)}")
    print(f"  on-grid position : {on_grid_pos}/{n} ({100*on_grid_pos/max(n,1):.1f}%)")
    print(f"  canonical rot    : {canon_rot}/{n} ({100*canon_rot/max(n,1):.1f}%)")
    print(f"  fully on-grid    : {len(both)}/{n} ({100*len(both)/max(n,1):.1f}%)")
    print(f"  top parts        : {', '.join(f'{p}x{c}' for p,c in parts.most_common(5))}")

    # Round-trip the fully-on-grid subset losslessly.
    if both:
        vocab = PartVocab.from_bricks([both])
        tok = AbsolutePoseTokenizer(vocab)
        tokens = tok.encode(both, canonicalize=False)
        back = flatten(parse_ldr(write_ldr(tok.decode(tokens))))
        a = Counter(x.key() for x in both)
        b = Counter(x.key() for x in back)
        status = "EXACT" if a == b else "MISMATCH"
        print(f"  on-grid round-trip: {status} ({len(tokens)} bricks)")
    return n, on_grid_pos, canon_rot, len(both)


if __name__ == "__main__":
    tot = np.zeros(4, dtype=np.int64)
    for p in sys.argv[1:]:
        tot += np.array(analyze(p))
    if len(sys.argv) > 2:
        n, og, cr, both = tot
        print(f"\n=== TOTAL over {len(sys.argv)-1} files ===")
        print(f"  bricks {n} | on-grid-pos {100*og/n:.1f}% | canon-rot {100*cr/n:.1f}% | "
              f"fully-on-grid {100*both/n:.1f}%")
