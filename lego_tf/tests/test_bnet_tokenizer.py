"""Native LEGO tokenizer round-trip on real val graphs: Tree -> tokens -> Tree."""

from __future__ import annotations

import os
from dataclasses import fields as dcf

import pytest

bricknet = pytest.importorskip("bricknet")

from lego_tf.bnet import trees as T
from lego_tf.bnet.tokenizer import Vocab, encode_tree, decode, SLIDE_MIN, SLIDE_MAX

VAL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "val.npz")
pytestmark = pytest.mark.skipif(not os.path.isfile(VAL), reason="val.npz not downloaded")


@pytest.fixture(scope="module")
def vocab():
    return Vocab()


@pytest.fixture(scope="module")
def graphs():
    return bricknet.load_graphs(VAL)


def _norm_edge(e):
    d = {f.name: getattr(e, f.name) for f in dcf(e)}
    out = dict(type=type(e).__name__, parent=d["parent"], child=d["child"],
               psub=int(d["parent_sub"]), csub=int(d["child_sub"]),
               pconn=int(d["parent_conn"]), cconn=int(d["child_conn"]))
    if "yaw" in d:
        out["yaw"] = int(d["yaw"]) % 360
    if "flip" in d:
        out["flip"] = bool(d["flip"])
    if "slide" in d:
        out["slide"] = max(SLIDE_MIN, min(SLIDE_MAX, int(d["slide"])))
    for c in ("rx", "ry", "rz"):
        if c in d:
            out[c] = int(d[c]) % 360
    return out


def _edges_key(edges):
    return sorted(map(_norm_edge, edges), key=lambda x: x["child"])


def test_vocab_layout_is_consistent(vocab):
    assert vocab.total == 21568
    # gid / split are inverse
    for seg in ("PART", "COLOR", "PTR", "PCONN", "CCONN", "ANGLE", "SLIDE"):
        g = vocab.gid(seg, 0)
        assert vocab.seg_of(g) == seg and vocab.local(g, seg) == 0
    with pytest.raises(ValueError):
        vocab.gid("FLIP", 99)   # FLIP has size 2 -> local 99 is out of range


def test_structural_roundtrip_all_val(vocab, graphs):
    exact = 0
    for g in graphs:
        tree = T.coerce_colors(T.sample_tree(g, seed=0))
        toks = encode_tree(tree, vocab)
        assert toks[0] == vocab.BOS and toks[-1] == vocab.EOS
        dt = decode(toks, vocab)
        if (len(dt.parts) == len(tree.parts)
                and all(a.part_id == b.part_id and a.color == b.color
                        for a, b in zip(dt.parts, tree.parts))
                and _edges_key(dt.edges) == _edges_key(tree.edges)):
            exact += 1
    assert exact == len(graphs), f"only {exact}/{len(graphs)} exact"


def test_score_roundtrip_sample(vocab, graphs):
    """A decoded tree reconstructs to a graph that scores identically to the original."""
    for g in graphs[:10]:
        tree = T.coerce_colors(T.sample_tree(g, seed=0))
        dt = decode(encode_tree(tree, vocab), vocab)
        s1 = bricknet.score_text(bricknet.graph_to_ldr(bricknet.tree_to_graph(tree)), collision=True)
        s2 = bricknet.score_text(bricknet.graph_to_ldr(bricknet.tree_to_graph(dt)), collision=True)
        assert s1 == s2


def test_token_density_reasonable(vocab, graphs):
    ntok = sum(len(encode_tree(T.sample_tree(g, seed=0), vocab)) for g in graphs[:50])
    nbrick = sum(len(g.part_ids) for g in graphs[:50])
    assert 5 <= ntok / nbrick <= 8  # ~6 tok/brick (compact flat connectors; was ~9)
