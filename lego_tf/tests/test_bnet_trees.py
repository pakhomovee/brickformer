"""Tests for the bnet build-order tree helpers (on real val graphs)."""

from __future__ import annotations

import os

import pytest

bricknet = pytest.importorskip("bricknet")

from lego_tf.bnet import trees as T

VAL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "val.npz")
pytestmark = pytest.mark.skipif(not os.path.isfile(VAL), reason="val.npz not downloaded")


@pytest.fixture(scope="module")
def graphs():
    return bricknet.load_graphs(VAL)


def test_sample_tree_is_deterministic(graphs):
    a = T.sample_tree(graphs[0], seed=0)
    b = T.sample_tree(graphs[0], seed=0)
    assert T.brick_count(a) == T.brick_count(b) == len(graphs[0].part_ids)


def test_truncate_keeps_first_k_and_bounds(graphs):
    tree = T.sample_tree(graphs[2], seed=0)
    n = T.brick_count(tree)
    assert n >= 3
    for k in (1, 2, n // 2, n):
        sub = T.truncate_tree(tree, k)
        assert T.brick_count(sub) == k
        assert sub.parts == tree.parts[:k]
        assert all(e.parent < k and e.child < k for e in sub.edges)
    with pytest.raises(ValueError):
        T.truncate_tree(tree, 0)


def test_coerce_colors_makes_all_known(graphs):
    known = set(T.catalog().code_to_color)
    coerced = 0
    for g in graphs:
        tree = T.coerce_colors(T.sample_tree(g, seed=0))
        assert all(p.color in known for p in tree.parts)
        coerced += 1
    assert coerced == len(graphs)
