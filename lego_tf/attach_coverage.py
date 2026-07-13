"""Report v1 attachment connectivity over real models (canonical-rotation subset).

Usage: python -m lego_tf.attach_coverage data/samples/*.mpd
"""

from __future__ import annotations

import os
import sys

from lego_tf.data.ldraw import flatten_file
from lego_tf.data.parts import PartLibrary, ConnectorExtractor
from lego_tf.tokenize.absolute import PartVocab
from lego_tf.tokenize.attach import AttachTokenizer
from lego_tf.tokenize.rotations import RotationCodebook

LDRAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "ldraw_lib", "ldraw")


def main(paths):
    ext = ConnectorExtractor(PartLibrary(LDRAW_ROOT))
    cb = RotationCodebook()
    for p in paths:
        allb = flatten_file(p)
        bricks = []
        for b in allb:
            try:
                cb.id_of(b.rot)
                bricks.append(b)
            except KeyError:
                pass
        if not bricks:
            continue
        tok = AttachTokenizer(PartVocab.from_bricks([bricks]), ext)
        seq = tok.encode(bricks)
        roots = sum(1 for t in seq.tokens if t.parent < 0)
        ok = "roundtrip=EXACT" if _rt(tok, seq, bricks) else "roundtrip=MISMATCH"
        print(f"{os.path.basename(p):22s} bricks={len(allb):4d} "
              f"canon={len(bricks):4d} attached={seq.parent_fraction()*100:5.1f}% "
              f"roots={roots:3d} {ok}")


def _rt(tok, seq, bricks):
    from collections import Counter
    a = Counter(b.key() for b in bricks)
    b = Counter(x.key() for x in tok.decode(seq))
    return a == b


if __name__ == "__main__":
    main(sys.argv[1:])
