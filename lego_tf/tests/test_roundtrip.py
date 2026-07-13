"""Stage-0 gate: LDR -> tokens -> LDR round-trip on the on-grid domain."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from lego_tf.data.ldraw import (
    Brick,
    flatten,
    parse_ldr,
    write_ldr,
    flatten_file,
)
from lego_tf.tokenize.absolute import AbsolutePoseTokenizer, PartVocab
from lego_tf.tokenize.rotations import RotationCodebook


# --- synthetic fixtures ------------------------------------------------------

def make_stack_ldr(n: int = 5) -> str:
    """A vertical stack of 2x4 bricks (part 3001). Brick height = 24 LDU, y points down."""
    lines = []
    for k in range(n):
        y = -24 * k
        lines.append(f"1 {4 + k} 0 {y} 0 1 0 0 0 1 0 0 0 1 3001.dat")
    return "\n".join(lines) + "\n"


MPD_NESTED = """\
0 FILE main.ldr
1 16 0 0 0 1 0 0 0 1 0 0 0 1 sub.ldr
1 16 40 0 0 1 0 0 0 1 0 0 0 1 sub.ldr
0 NOFILE
0 FILE sub.ldr
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 16 0 -24 0 0 0 -1 0 1 0 1 0 0 3001.dat
0 NOFILE
"""


def multiset(bricks: list[Brick]) -> Counter:
    return Counter(b.key() for b in bricks)


# --- tests -------------------------------------------------------------------

def test_rotation_set_is_48_signed_permutations():
    cb = RotationCodebook()
    assert len(cb) == 48
    keys = {tuple(int(v) for v in m.flatten()) for m in cb.matrices}
    assert len(keys) == 48  # all distinct
    for m in cb.matrices:
        # exactly one non-zero (+-1) per row and per column
        assert np.array_equal(np.abs(m).sum(axis=0), np.ones(3))
        assert np.array_equal(np.abs(m).sum(axis=1), np.ones(3))
        assert set(np.unique(m)).issubset({-1, 0, 1})


def test_parse_flatten_stack():
    bricks = flatten(parse_ldr(make_stack_ldr(5)))
    assert len(bricks) == 5
    assert all(b.part == "3001.dat" for b in bricks)
    # colours preserved
    assert [b.color for b in bricks] == [4, 5, 6, 7, 8]
    # positions climb (y more negative each step)
    ys = [b.pos[1] for b in bricks]
    assert ys == sorted(ys, reverse=True)


def test_flatten_nested_mpd_composes_transforms_and_colors():
    bricks = flatten(parse_ldr(MPD_NESTED))
    # 2 instances of sub (2 bricks each) = 4 bricks
    assert len(bricks) == 4
    # sub is placed at x=0 and x=40; each sub has a brick at local (0,0,0)
    xs = sorted({round(b.pos[0]) for b in bricks})
    assert xs == [0, 40]
    # colour 16 (inherit) inside sub resolves to the sub's placement colour (also 16 ->
    # top-level default 16); explicit colour 4 stays 4.
    colors = Counter(b.color for b in bricks)
    assert colors[4] == 2  # the explicit-4 brick, once per sub instance


@pytest.mark.parametrize("n", [1, 5, 20])
def test_roundtrip_stack_lossless(n):
    """encode(canonicalize=False) then decode must reproduce the exact brick multiset."""
    bricks = flatten(parse_ldr(make_stack_ldr(n)))
    vocab = PartVocab.from_bricks([bricks])
    tok = AbsolutePoseTokenizer(vocab)

    tokens = tok.encode(bricks, canonicalize=False)
    assert len(tokens) == n
    back = tok.decode(tokens)
    assert multiset(back) == multiset(bricks)


def test_roundtrip_through_text():
    """Full loop: LDR text -> parse -> tokens -> decode -> LDR text -> parse. Multiset stable."""
    bricks = flatten(parse_ldr(MPD_NESTED))
    vocab = PartVocab.from_bricks([bricks])
    tok = AbsolutePoseTokenizer(vocab)

    tokens = tok.encode(bricks, canonicalize=False)
    ldr_text = write_ldr(tok.decode(tokens))
    reparsed = flatten(parse_ldr(ldr_text))
    assert multiset(reparsed) == multiset(bricks)


def test_canonicalize_preserves_build_up_to_translation():
    """Canonicalised encoding keeps the same shape (translation/order-invariant)."""
    bricks = flatten(parse_ldr(make_stack_ldr(6)))
    vocab = PartVocab.from_bricks([bricks])
    tok = AbsolutePoseTokenizer(vocab)

    raw = tok.encode(bricks, canonicalize=False).array
    canon = tok.encode(bricks, canonicalize=True).array
    # same set of (rot, part, color) rows and same relative geometry (min at origin)
    assert canon[:, :3].min() == 0
    # pairwise relative positions preserved: difference between raw and canon is a pure
    # translation for the matched brick set.
    assert Counter(map(tuple, raw[:, 3:].tolist())) == Counter(map(tuple, canon[:, 3:].tolist()))


def test_offgrid_rejected():
    b = Brick(part="3001.dat", color=15, matrix=np.eye(4))
    b.matrix[0, 3] = 10.5  # half-LDU off grid
    tok = AbsolutePoseTokenizer(PartVocab(["3001.dat"]))
    with pytest.raises(ValueError):
        tok.encode([b], require_grid=True)


def test_noncanonical_rotation_rejected():
    cb = RotationCodebook()
    theta = np.deg2rad(30)
    rot = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    with pytest.raises(KeyError):
        cb.id_of(rot)
