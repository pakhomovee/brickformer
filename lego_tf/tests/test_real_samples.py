"""Regression tests on real OMR vehicle MPDs (skipped if samples are absent)."""

from __future__ import annotations

import glob
import os
from collections import Counter

import numpy as np
import pytest

from lego_tf.data.ldraw import flatten_file, flatten, parse_ldr, write_ldr
from lego_tf.tokenize.absolute import AbsolutePoseTokenizer, PartVocab
from lego_tf.tokenize.rotations import RotationCodebook

SAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples", "*.mpd")))

pytestmark = pytest.mark.skipif(not SAMPLES, reason="no sample MPDs downloaded")


@pytest.mark.parametrize("path", SAMPLES, ids=[os.path.basename(p) for p in SAMPLES])
def test_inlined_dat_parts_not_exploded(path):
    """Inlined custom .dat parts must be leaf bricks, never expanded into primitives."""
    bricks = flatten_file(path)
    prim_markers = ("edge.dat", "cyli.dat", "ring", "disc.dat", "-4ndis.dat")
    exploded = [b for b in bricks if any(m in b.part for m in prim_markers)]
    assert not exploded, f"flatten leaked primitives (parser over-recursed): {exploded[:3]}"


@pytest.mark.parametrize("path", SAMPLES, ids=[os.path.basename(p) for p in SAMPLES])
def test_on_grid_subset_roundtrips_exact(path):
    """The fully-on-grid subset of each real model must round-trip to an identical multiset."""
    bricks = flatten_file(path)
    cb = RotationCodebook()
    on_grid = []
    for b in bricks:
        if not np.all(np.abs(b.pos - np.round(b.pos)) < 1e-3):
            continue
        try:
            cb.id_of(b.rot)
        except KeyError:
            continue
        on_grid.append(b)
    assert on_grid, "expected at least some on-grid bricks"

    tok = AbsolutePoseTokenizer(PartVocab.from_bricks([on_grid]))
    tokens = tok.encode(on_grid, canonicalize=False)
    back = flatten(parse_ldr(write_ldr(tok.decode(tokens))))
    assert Counter(b.key() for b in back) == Counter(b.key() for b in on_grid)
