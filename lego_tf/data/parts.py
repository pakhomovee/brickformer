"""LDraw parts-library access + connector (stud) geometry extraction.

Connectors are the backbone of representation v1 (plan sec 3): to attach a brick
parent-relatively we need to know where its studs / anti-studs sit. LDraw encodes studs
structurally -- a part references stud *primitives* (``stud.dat`` on top, ``stud4*`` tubes
underneath) at explicit local transforms. We recover connector geometry by walking a part's
reference tree, composing transforms, and recording every stud-primitive placement.

Convention (from ``p/stud.dat``): a male stud sits with its base at the local origin and
protrudes along **-Y** (LDraw Y points down, so -Y is "up"). We store the connector origin
and its outward axis (transformed local -Y for males).

The stud-primitive taxonomy is large and genuinely ambiguous by name (``stud2`` = male open
stud, ``stud20`` = female tube pattern), so we use explicit curated sets and treat anything
else as ``unknown`` rather than guessing. This is exact on the brick/plate/SNOT domain we
validate; the round-part tube family can be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import functools
import os

import numpy as np

# Protruding "stud" primitives (solid / hollow / logo / small). The SAME primitive is a top
# stud when placed upright and an underside receptacle when placed inverted (e.g. stud3 inside
# a stug3-* group is flipped with a -5 Y-scale), so polarity is decided by orientation, not
# name -- see `_classify`.
STUD_PRIMS = {
    "stud", "stud2", "stud2a", "stud2s", "stud2s2", "stud2s2e", "stud3", "stud3a",
    "stud6", "stud6a", "studa", "studp01", "studel", "studx", "studxa",
    "studh", "studhl", "studhr", "studline",
    "stud-logo", "stud-logo2", "stud-logo3", "stud-logo4", "stud-logo5",
    "stud2-logo", "stud2-logo2", "stud2-logo3", "stud2-logo4", "stud2-logo5",
}
# Tube primitives: always a female receptacle regardless of orientation.
TUBE_PRIMS = {
    "stud4", "stud4a", "stud4o", "stud4od", "stud4oda", "stud4h", "stud4s", "stud4s2",
    "stud4f1n", "stud4f1s", "stud4f1w", "stud4f2n", "stud4f2s", "stud4f2w",
    "stud4f3n", "stud4f3s", "stud4f4n", "stud4f4s", "stud4f5n",
}


class ConnType(str, Enum):
    STUD = "stud"            # male, protruding
    ANTISTUD = "antistud"    # female tube/receptacle
    PIN_HOLE = "pin_hole"    # female round Technic hole
    AXLE_HOLE = "axle_hole"  # female cross Technic hole
    PIN = "pin"              # male Technic pin
    AXLE = "axle"            # male Technic axle (cross shaft)
    BALL = "ball"            # male ball joint
    SOCKET = "socket"        # female ball socket
    BAR = "bar"              # male bar / handle
    CLIP = "clip"            # female clip


@dataclass
class Connector:
    type: ConnType
    pos: np.ndarray   # (3,) local part coordinates (LDU)
    axis: np.ndarray  # (3,) unit outward direction

    def transformed(self, xform: np.ndarray) -> "Connector":
        p = xform[:3, :3] @ self.pos + xform[:3, 3]
        a = xform[:3, :3] @ self.axis
        n = np.linalg.norm(a)
        return Connector(self.type, p, a / n if n else a)


def _base_name(target: str) -> str:
    """Strip directory and extension: 's\\3001s01.dat' -> '3001s01'."""
    t = target.replace("\\", "/").split("/")[-1]
    return t[:-4] if t.lower().endswith(".dat") else t


class PartLibrary:
    """Resolve LDraw part/primitive names to files under an unzipped ``ldraw/`` root."""

    def __init__(self, root: str):
        # root points at the dir containing parts/ and p/
        self.root = root
        self._search = [
            os.path.join(root, "parts"),
            os.path.join(root, "parts", "s"),
            os.path.join(root, "p"),
            os.path.join(root, "p", "48"),
            os.path.join(root, "p", "8"),
        ]

    def resolve(self, name: str) -> tuple[str | None, str]:
        """Return (path, category). category in {part, subpart, primitive, missing}."""
        rel = name.replace("\\", "/")
        cats = ["part", "subpart", "primitive", "primitive", "primitive"]
        for base, cat in zip(self._search, cats):
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p, cat
            # try just the filename in this dir (handles nested-path names)
            p2 = os.path.join(base, os.path.basename(rel))
            if os.path.isfile(p2):
                return p2, cat
        return None, "missing"

    @functools.lru_cache(maxsize=8192)
    def _read_refs(self, path: str) -> tuple:
        """Parse type-1 refs of a file: tuple of (4x4-bytes, target). Cached by path."""
        refs = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.split()
                if len(p) >= 15 and p[0] == "1":
                    try:
                        nums = list(map(float, p[2:14]))
                    except ValueError:
                        continue
                    a, b, c, d, e, ff, g, h, i = nums[3:]
                    x, y, z = nums[0:3]
                    m = np.array([[a, b, c, x], [d, e, ff, y], [g, h, i, z], [0, 0, 0, 1]])
                    refs.append((m, " ".join(p[14:])))
        return tuple(refs)

    @functools.lru_cache(maxsize=8192)
    def _read_local_points(self, path: str) -> tuple:
        """Parse drawing-primitive vertices (type 2/3/4/5) of a file as (N,3) local points."""
        pts = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.split()
                if not p or p[0] not in ("2", "3", "4", "5"):
                    continue
                nverts = {"2": 2, "3": 3, "4": 4, "5": 4}[p[0]]
                try:
                    vals = list(map(float, p[2:2 + 3 * nverts]))
                except ValueError:
                    continue
                for k in range(nverts):
                    pts.append(vals[3 * k:3 * k + 3])
        return tuple(map(tuple, pts))


class ConnectorExtractor:
    def __init__(self, lib: PartLibrary, max_depth: int = 12):
        self.lib = lib
        self.max_depth = max_depth

    @functools.lru_cache(maxsize=8192)
    def y_extent(self, part_name: str) -> tuple:
        """Local (min_y, max_y) of a part's geometry: stud tips (min) to bottom plane (max)."""
        lo = [np.inf]
        hi = [-np.inf]
        self._extent_walk(part_name, np.eye(4), 0, lo, hi)
        return (float(lo[0]), float(hi[0]))

    def _extent_walk(self, name, xform, depth, lo, hi):
        if depth > self.max_depth:
            return
        path, cat = self.lib.resolve(name)
        if path is None:
            return
        for px, py, pz in self.lib._read_local_points(path):
            wy = xform[1, 0] * px + xform[1, 1] * py + xform[1, 2] * pz + xform[1, 3]
            if wy < lo[0]:
                lo[0] = wy
            if wy > hi[0]:
                hi[0] = wy
        for mat, target in self.lib._read_refs(path):
            self._extent_walk(target, xform @ mat, depth + 1, lo, hi)

    @functools.lru_cache(maxsize=8192)
    def xz_extent(self, part_name: str) -> tuple:
        """Local (xmin, xmax, zmin, zmax) of a part's geometry."""
        box = [np.inf, -np.inf, np.inf, -np.inf]  # xmin xmax zmin zmax
        self._box_walk(part_name, np.eye(4), 0, box)
        return tuple(float(v) for v in box)

    def _box_walk(self, name, xform, depth, box):
        if depth > self.max_depth:
            return
        path, cat = self.lib.resolve(name)
        if path is None:
            return
        for px, py, pz in self.lib._read_local_points(path):
            wx = xform[0, 0]*px + xform[0, 1]*py + xform[0, 2]*pz + xform[0, 3]
            wz = xform[2, 0]*px + xform[2, 1]*py + xform[2, 2]*pz + xform[2, 3]
            box[0] = min(box[0], wx); box[1] = max(box[1], wx)
            box[2] = min(box[2], wz); box[3] = max(box[3], wz)
        for mat, target in self.lib._read_refs(path):
            self._box_walk(target, xform @ mat, depth + 1, box)

    @functools.lru_cache(maxsize=8192)
    def footprint_cells(self, part_name: str) -> tuple:
        """Underside stud-grid cells (x, z) a part covers, from its x,z bounding box.

        Studs are 20 LDU apart; cell centres are at (min + 10 + 20k). This is where a part
        can *receive* a stud from below, so tiles/smooth parts (no top studs) still expose
        female ports. Empty if the part is smaller than a stud (e.g. bars, pins).
        """
        xmin, xmax, zmin, zmax = self.xz_extent(part_name)
        if not np.isfinite([xmin, xmax, zmin, zmax]).all():
            return ()
        cells = []
        nx = max(0, round((xmax - xmin) / 20.0))
        nz = max(0, round((zmax - zmin) / 20.0))
        for ix in range(nx):
            for iz in range(nz):
                cells.append((xmin + 10 + 20 * ix, zmin + 10 + 20 * iz))
        return tuple(cells)

    @functools.lru_cache(maxsize=8192)
    def connectors(self, part_name: str) -> tuple:
        """Return a tuple of local-space `Connector`s for a part (cached)."""
        out: list[Connector] = []
        self._walk(part_name, np.eye(4), 0, out)
        return tuple(out)

    def _walk(self, name: str, xform: np.ndarray, depth: int, out: list) -> None:
        if depth > self.max_depth:
            return
        path, cat = self.lib.resolve(name)
        if path is None:
            return
        if cat == "primitive":
            base = _base_name(name)
            conn = _classify(base, xform)
            if conn is not None:
                out.append(conn)
                return
            if base.startswith("stug"):
                # stud-group primitive (e.g. stug-2x2): a container of stud.dat refs.
                pass  # fall through to descend into its refs below
            else:
                # other primitives (edges, cylinders, boxes) are not connectors
                return
        # part or subpart: descend
        for mat, target in self.lib._read_refs(path):
            self._walk(target, xform @ mat, depth + 1, out)


def _classify(base: str, xform: np.ndarray) -> Connector | None:
    """Classify a primitive placement into a Connector, or None if it isn't a stud primitive.

    A stud primitive's outward direction is its transformed local -Y (studs protrude along
    -Y in LDraw). Polarity:
      * tube primitives (stud4*) are always female receptacles;
      * a protruding stud is male if it points up/sideways (outward.y <= 0) and a female
        underside receptacle if inverted to point down (outward.y > 0).
    """
    pos = xform[:3, 3].copy()
    up = xform[:3, :3] @ np.array([0.0, -1.0, 0.0])  # transformed protrusion direction
    n = np.linalg.norm(up)
    up = up / n if n else up

    if base in TUBE_PRIMS:
        # opening faces the same way the tube is oriented; keep the tube's outward -Y.
        return Connector(ConnType.ANTISTUD, pos, up)
    if base in STUD_PRIMS:
        if up[1] > 0.5:  # flipped to point down -> underside receptacle
            return Connector(ConnType.ANTISTUD, pos, up)
        return Connector(ConnType.STUD, pos, up)

    # Technic connectors: axis is the primitive's transformed local Y (up computed above,
    # but Technic axes are usually horizontal). Use the raw transformed +Y as the axis.
    axis = xform[:3, :3] @ np.array([0.0, 1.0, 0.0])
    na = np.linalg.norm(axis)
    axis = axis / na if na else axis
    tech = _connector_type_by_name(base)
    if tech is not None:
        return Connector(tech, pos, axis)
    return None


def _connector_type_by_name(base: str) -> "ConnType | None":
    """Classify a Technic / ball / clip primitive by name. Holes checked before shafts."""
    # Female Technic holes
    if base.startswith(("peghole", "npeghol", "connhol", "dnpeghol", "dconnhol")) \
            or base in ("wpinhole", "beamhole"):
        return ConnType.PIN_HOLE
    if base.startswith(("axlehol", "axleho", "daxlehole")) \
            or _axl_hole(base) or base == "axleend2hole":
        return ConnType.AXLE_HOLE
    # Male Technic shafts
    if base.startswith(("connect", "connectcollar", "connectslit", "connectring")):
        return ConnType.PIN
    if base.startswith(("axle", "axl2end", "axl3end", "axl5end", "daxle")):
        return ConnType.AXLE
    # Ball joints
    if base.startswith("joint8ball"):
        return ConnType.BALL
    if base.startswith("joint8socket"):
        return ConnType.SOCKET
    # Clips (female) grip bars/handles (male)
    if base.startswith(("clip", "clh")):
        return ConnType.CLIP
    if base.startswith(("handle", "phandle", "duphandle", "finger")):
        return ConnType.BAR
    return None


def _axl_hole(base: str) -> bool:
    """Match axle-hole primitives like axl2hol8, axl3hole, axl4hol2."""
    import re
    return bool(re.match(r"^axl\d+hol", base))
