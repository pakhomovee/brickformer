"""Round-trip + grammar tests for the flat token stream."""

from __future__ import annotations

import os
from collections import Counter

import pytest

from lego_tf.data.ldraw import parse_ldr, flatten, flatten_file
from lego_tf.data.parts import PartLibrary, ConnectorExtractor
from lego_tf.tokenize.absolute import PartVocab
from lego_tf.tokenize.attach import AttachTokenizer
from lego_tf.tokenize.rotations import RotationCodebook
from lego_tf.tokenize.stream import (
    ColorVocab, build_vocab, encode_stream, decode_stream, allowed_next_segment,
)

LDRAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ldraw_lib", "ldraw")
pytestmark = pytest.mark.skipif(not os.path.isdir(LDRAW_ROOT), reason="LDraw library not downloaded")


@pytest.fixture(scope="module")
def ext():
    return ConnectorExtractor(PartLibrary(LDRAW_ROOT))


def _pipeline(bricks, ext):
    parts = PartVocab.from_bricks([bricks])
    colors = ColorVocab([int(b.color) for b in bricks])
    tok = AttachTokenizer(parts, ext)
    seq = tok.encode(bricks)
    vocab = build_vocab([seq], parts, colors)
    return parts, colors, tok, seq, vocab


def multiset(bricks):
    return Counter(b.key() for b in bricks)


def stack(n, part="3001.dat", dy=24):
    lines = [f"1 {4 + k % 10} 0 {-dy*k} 0 1 0 0 0 1 0 0 0 1 {part}" for k in range(n)]
    return flatten(parse_ldr("\n".join(lines)))


def test_stream_roundtrip_stack(ext):
    bricks = stack(6)
    parts, colors, tok, seq, vocab = _pipeline(bricks, ext)
    ids = encode_stream(seq, parts, colors, vocab)
    assert ids[0] == vocab.BOS and ids[-1] == vocab.EOS
    seq2 = decode_stream(ids, colors, vocab)
    back = tok.decode(seq2)
    assert multiset(back) == multiset(bricks)


def test_all_ids_in_range(ext):
    bricks = stack(6)
    parts, colors, tok, seq, vocab = _pipeline(bricks, ext)
    ids = encode_stream(seq, parts, colors, vocab)
    assert all(0 <= i < vocab.size for i in ids)


def test_grammar_predicts_every_position(ext):
    """The grammar's expected segment must match the actual segment at each stream position."""
    bricks = stack(5)
    parts, colors, tok, seq, vocab = _pipeline(bricks, ext)
    ids = encode_stream(seq, parts, colors, vocab)
    for k in range(1, len(ids)):
        expected = allowed_next_segment(ids[:k], vocab)
        seg, _ = vocab.decode(ids[k])
        if expected == "PART_OR_EOS":
            assert seg in ("PART", "SPECIAL")
        elif expected == "END":
            pytest.fail("grammar said END before stream ended")
        else:
            assert seg == expected, f"pos {k}: grammar {expected} != actual {seg}"
    assert allowed_next_segment(ids, vocab) == "END"


@pytest.mark.parametrize("name", ["6927-atv.mpd", "8464-loader.mpd"])
def test_stream_roundtrip_real_model(ext, name):
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples", name)
    if not os.path.isfile(path):
        pytest.skip(name)
    cb = RotationCodebook()
    bricks = [b for b in flatten_file(path) if _canon(cb, b)]
    parts, colors, tok, seq, vocab = _pipeline(bricks, ext)
    ids = encode_stream(seq, parts, colors, vocab)
    seq2 = decode_stream(ids, colors, vocab)
    assert multiset(tok.decode(seq2)) == multiset(bricks)


def _canon(cb, b):
    try:
        cb.id_of(b.rot)
        return True
    except KeyError:
        return False
