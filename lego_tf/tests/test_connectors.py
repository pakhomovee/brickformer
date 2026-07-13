"""Validate stud connector extraction against parts with known stud layouts."""

from __future__ import annotations

import os

import numpy as np
import pytest

from lego_tf.data.parts import PartLibrary, ConnectorExtractor, ConnType

LDRAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ldraw_lib", "ldraw")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(LDRAW_ROOT), reason="LDraw library not downloaded"
)


@pytest.fixture(scope="module")
def extractor():
    return ConnectorExtractor(PartLibrary(LDRAW_ROOT))


def males(conns):
    return [c for c in conns if c.type == ConnType.STUD]


# part -> expected number of top studs (male)
KNOWN_STUD_COUNTS = {
    "3024.dat": 1,   # plate 1x1
    "3005.dat": 1,   # brick 1x1
    "3003.dat": 4,   # brick 2x2
    "3020.dat": 8,   # plate 2x4
    "3001.dat": 8,   # brick 2x4
    "3010.dat": 4,   # brick 1x4
    "3031.dat": 16,  # plate 4x4
}


@pytest.mark.parametrize("part,count", KNOWN_STUD_COUNTS.items())
def test_male_stud_counts(extractor, part, count):
    conns = extractor.connectors(part)
    assert len(males(conns)) == count, f"{part}: got {len(males(conns))} studs, want {count}"


def test_3001_stud_grid(extractor):
    """2x4 brick studs sit on a 4x2 grid at x in +-{10,30}, z in +-10, all on the top plane."""
    studs = males(extractor.connectors("3001.dat"))
    xs = sorted({round(c.pos[0]) for c in studs})
    zs = sorted({round(c.pos[2]) for c in studs})
    ys = {round(c.pos[1]) for c in studs}
    assert xs == [-30, -10, 10, 30]
    assert zs == [-10, 10]
    assert ys == {0}  # top plane at y=0


def test_stud_axis_points_up(extractor):
    """Male studs protrude 'up' = LDraw -Y."""
    studs = males(extractor.connectors("3001.dat"))
    for c in studs:
        assert np.allclose(c.axis, [0, -1, 0], atol=1e-6)


def test_brick_has_underside_antistuds(extractor):
    """A 2x4 brick has female tubes on its underside (count may differ from top studs)."""
    conns = extractor.connectors("3001.dat")
    anti = [c for c in conns if c.type == ConnType.ANTISTUD]
    assert len(anti) >= 1
    # antistuds are below the top plane (larger y, since Y points down)
    assert all(c.pos[1] > 0 for c in anti)


def _by_type(conns):
    from collections import Counter
    return Counter(c.type for c in conns)


def test_technic_beam_has_holes(extractor):
    """6632 (Technic beam 1x3) exposes pin holes and an axle hole, and no studs."""
    c = _by_type(extractor.connectors("6632.dat"))
    assert c[ConnType.PIN_HOLE] >= 2
    assert c[ConnType.AXLE_HOLE] >= 1
    assert c[ConnType.STUD] == 0


def test_technic_pin_is_male(extractor):
    """3673 (Technic pin 2L) exposes male pin connectors and nothing female."""
    c = _by_type(extractor.connectors("3673.dat"))
    assert c[ConnType.PIN] >= 1
    assert c[ConnType.PIN_HOLE] == 0 and c[ConnType.STUD] == 0


def test_technic_brick_with_holes_has_both(extractor):
    """3701 (Technic Brick 1x4 with holes) has studs AND pin holes."""
    c = _by_type(extractor.connectors("3701.dat"))
    assert c[ConnType.STUD] >= 1 and c[ConnType.PIN_HOLE] >= 1


def test_clip_ball_socket_detected(extractor):
    """Clip, ball and socket connector families are extracted by name."""
    assert _by_type(extractor.connectors("4085a.dat"))[ConnType.CLIP] >= 1   # clip
    assert _by_type(extractor.connectors("14417.dat"))[ConnType.BALL] >= 1   # ball joint
    assert _by_type(extractor.connectors("14419.dat"))[ConnType.SOCKET] >= 1  # socket


def test_tile_has_footprint_but_no_studs(extractor):
    """A 2x4 tile exposes an 8-cell underside footprint and no top studs."""
    assert _by_type(extractor.connectors("87079.dat"))[ConnType.STUD] == 0
    assert len(extractor.footprint_cells("87079.dat")) == 8
