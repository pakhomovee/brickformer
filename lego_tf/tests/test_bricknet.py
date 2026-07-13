"""Smoke test for the bricknet substrate (representation layer we now build on).

Pins the primitives our go-forward work depends on: a bundled connector-annotated catalog,
graph parsing, spanning-tree sampling, text serialization, and the collision scorer. Skipped
if bricknet isn't installed so the legacy tokenizer tests still run standalone.
"""

from __future__ import annotations

import os

import pytest

bricknet = pytest.importorskip("bricknet")

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples", "6608-tractor.mpd")


def test_catalog_and_connectors_bundled():
    """The connector-annotated catalog ships with the package (usable offline)."""
    cat = bricknet.load_catalog()
    conns = bricknet.load_connectors()
    assert cat is not None
    assert len(conns) > 10_000  # ~14.5k parts annotated


def test_edge_types_expose_articulation_dof():
    """Typed edges carry the articulation DOF that supersedes our deferred fine-geom channel."""
    assert "yaw" in bricknet.StudEdge.__annotations__
    assert {"flip", "yaw", "slide"} <= set(bricknet.AxleEdge.__annotations__)
    assert {"rx", "ry", "rz"} <= set(bricknet.BallEdge.__annotations__)
    assert {"flip", "yaw"} <= set(bricknet.HingeEdge.__annotations__)


@pytest.mark.skipif(not os.path.isfile(SAMPLE), reason="sample MPD missing")
def test_graph_pipeline_runs_end_to_end():
    """parse -> sample tree -> serialize -> score, on a real sample."""
    cat = bricknet.load_catalog()
    g = bricknet.parse_ldr(open(SAMPLE).read())
    assert len(g.part_ids) > 0
    assert len(g.components) >= 1

    tree = bricknet.sample_tree(g, method="random", seed=0)
    text = bricknet.serialize_tree(tree, cat)
    assert isinstance(text, str) and len(text) > 0

    placed, collisions, _ = bricknet.score_text(bricknet.graph_to_ldr(g), collision=True)
    assert placed > 0
    assert collisions is None or collisions >= 0
