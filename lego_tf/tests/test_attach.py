"""Round-trip + connectivity tests for the parent-relative attachment tokenizer (v1)."""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pytest

from lego_tf.data.ldraw import Brick, parse_ldr, flatten, flatten_file
from lego_tf.data.parts import PartLibrary, ConnectorExtractor
from lego_tf.tokenize.absolute import PartVocab
from lego_tf.tokenize.attach import AttachTokenizer

LDRAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ldraw_lib", "ldraw")
pytestmark = pytest.mark.skipif(not os.path.isdir(LDRAW_ROOT), reason="LDraw library not downloaded")


@pytest.fixture(scope="module")
def tok_factory():
    ext = ConnectorExtractor(PartLibrary(LDRAW_ROOT))

    def make(bricks):
        return AttachTokenizer(PartVocab.from_bricks([bricks]), ext)

    return make


def multiset(bricks):
    return Counter(b.key() for b in bricks)


def brick_stack(n, part="3001.dat", dy=24):
    lines = [f"1 {4 + k % 10} 0 {-dy*k} 0 1 0 0 0 1 0 0 0 1 {part}" for k in range(n)]
    return flatten(parse_ldr("\n".join(lines)))


def test_stack_all_attached_and_roundtrips(tok_factory):
    bricks = brick_stack(6)
    tok = tok_factory(bricks)
    seq = tok.encode(bricks)
    # first brick is the root (absolute); the other 5 attach to a parent
    assert seq.tokens[0].parent == -1
    assert all(t.parent >= 0 for t in seq.tokens[1:])
    assert seq.parent_fraction() == pytest.approx(5 / 6)
    back = tok.decode(seq)
    assert multiset(back) == multiset(bricks)


def test_plate_stack_uses_plate_height(tok_factory):
    """Stacked 2x4 plates (height 8) must attach and round-trip with the right spacing."""
    bricks = brick_stack(4, part="3020.dat", dy=8)
    tok = tok_factory(bricks)
    seq = tok.encode(bricks)
    assert all(t.parent >= 0 for t in seq.tokens[1:])
    back = tok.decode(seq)
    assert multiset(back) == multiset(bricks)


def test_offset_brick_attaches(tok_factory):
    """A brick shifted by whole studs still lands on the grid and attaches."""
    lines = [
        "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat",       # lower 2x4
        "1 5 40 -24 0 1 0 0 0 1 0 0 0 1 3001.dat",     # upper, shifted +40 (2 studs) in x
    ]
    bricks = flatten(parse_ldr("\n".join(lines)))
    tok = tok_factory(bricks)
    seq = tok.encode(bricks)
    assert seq.tokens[1].parent == 0
    assert multiset(tok.decode(seq)) == multiset(bricks)


def test_bfs_order_independent_of_file_order(tok_factory):
    """A stack listed top-down still attaches every brick (BFS roots at the lowest)."""
    # list bricks from top (most negative y) to bottom (y=0): reverse of build order
    lines = [f"1 {4 + k} 0 {-24*(5-k)} 0 1 0 0 0 1 0 0 0 1 3001.dat" for k in range(6)]
    bricks = flatten(parse_ldr("\n".join(lines)))
    tok = tok_factory(bricks)
    seq = tok.encode(bricks)
    assert seq.num_roots() == 1               # single connected stack -> one root
    assert seq.tokens[0].parent == -1         # root emitted first
    assert seq.parent_fraction() == pytest.approx(5 / 6)
    assert multiset(tok.decode(seq)) == multiset(bricks)


def test_attach_from_below(tok_factory):
    """A brick placed under an earlier-listed brick still attaches (bidirectional ports)."""
    lines = [
        "1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat",   # upper brick listed first
        "1 5 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat",      # lower brick listed second
    ]
    bricks = flatten(parse_ldr("\n".join(lines)))
    tok = tok_factory(bricks)
    seq = tok.encode(bricks)
    assert seq.num_roots() == 1
    assert seq.parent_fraction() == pytest.approx(0.5)
    assert multiset(tok.decode(seq)) == multiset(bricks)


def test_tile_attaches_on_brick(tok_factory):
    """A smooth-top 2x4 tile (no studs) attaches to the brick below via footprint ports."""
    lines = [
        "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat",      # 2x4 brick, studs at y=0 plane
        "1 5 0 -8 0 1 0 0 0 1 0 0 0 1 87079.dat",     # 2x4 tile sitting on top (bottom at y=0)
    ]
    bricks = flatten(parse_ldr("\n".join(lines)))
    tok = tok_factory(bricks)
    seq = tok.encode(bricks)
    assert seq.num_roots() == 1                       # tile attaches, not a second root
    assert seq.parent_fraction() == pytest.approx(0.5)
    assert multiset(tok.decode(seq)) == multiset(bricks)


@pytest.mark.parametrize("name", ["6927-atv.mpd", "30051-xwing.mpd"])
def test_real_model_canonical_subset_roundtrips(tok_factory, name):
    """On the canonical-rotation subset of a real model, encoding is lossless.

    Non-canonical (hinged/angled) bricks need the fine-geom channel (not in v1 yet), so we
    scope the round-trip to canonically-oriented bricks -- the domain v1 claims to cover.
    """
    from lego_tf.tokenize.rotations import RotationCodebook

    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples", name)
    if not os.path.isfile(path):
        pytest.skip(f"{name} not present")
    cb = RotationCodebook()
    bricks = []
    for b in flatten_file(path):
        try:
            cb.id_of(b.rot)
        except KeyError:
            continue
        bricks.append(b)
    assert bricks
    tok = tok_factory(bricks)
    seq = tok.encode(bricks)
    assert multiset(tok.decode(seq)) == multiset(bricks)
    # at least some bricks should attach to a stud parent (studs-up grid connectivity)
    assert seq.parent_fraction() > 0.0
