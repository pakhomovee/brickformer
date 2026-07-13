"""LDraw (.ldr / .mpd) parsing, flattening, and writing.

We only care about *type-1* lines (sub-file references), which is where every placed
brick or sub-model lives:

    1 <colour> x y z  a b c  d e f  g h i  <file>

The 12 numbers form an LDraw transform: translation (x, y, z) and a row-major 3x3
rotation/scale matrix [[a,b,c],[d,e,f],[g,h,i]]. In LDraw a placed part `p` maps a
local point `v` to world as `M @ v + t`, i.e. the homogeneous 4x4::

    | a b c x |
    | d e f y |
    | g h i z |
    | 0 0 0 1 |

An `.mpd` bundles several named sub-files (`0 FILE name` ... `0 NOFILE`); a type-1 line
may reference another sub-file in the same document (a *sub-model*) or a library part
(e.g. ``3001.dat``). `flatten` recursively expands sub-models, composing transforms and
resolving LDraw colour inheritance (16 = inherit, 24 = edge/inherit), so the result is a
flat list of library-part placements in world space -- the representation the tokenizer
consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

# LDraw colour code meaning "inherit from the referencing part".
COLOR_INHERIT = 16
COLOR_EDGE_INHERIT = 24
# Default main colour used when the very top level inherits (no parent to inherit from).
COLOR_DEFAULT = 16


@dataclass
class Brick:
    """A single library-part placement in world space."""

    part: str  # library part filename, lower-cased, e.g. "3001.dat"
    color: int  # resolved LDraw colour code
    matrix: np.ndarray  # 4x4 float64 homogeneous transform

    @property
    def pos(self) -> np.ndarray:
        return self.matrix[:3, 3]

    @property
    def rot(self) -> np.ndarray:
        return self.matrix[:3, :3]

    def key(self, pos_round: int = 0):
        """Hashable canonical identity: (part, color, rounded pos, rounded rot).

        `pos_round` is decimal places for position; rotation is rounded to ints (the
        on-grid domain). Used to compare brick multisets across a round trip.
        """
        p = tuple(np.round(self.pos, pos_round).tolist())
        r = tuple(np.round(self.rot).astype(np.int64).flatten().tolist())
        return (self.part.lower(), int(self.color), p, r)


@dataclass
class SubModel:
    name: str
    refs: list[tuple[int, np.ndarray, str]] = field(default_factory=list)
    # each ref: (colour, 4x4 matrix, target filename)


@dataclass
class Model:
    submodels: dict[str, SubModel]
    main: str  # name of the top-level sub-model

    def is_submodel(self, name: str) -> bool:
        return _norm(name) in self.submodels


def _norm(name: str) -> str:
    return name.strip().lower()


def _mat_from_line(nums: list[float]) -> np.ndarray:
    x, y, z, a, b, c, d, e, f, g, h, i = nums
    return np.array(
        [
            [a, b, c, x],
            [d, e, f, y],
            [g, h, i, z],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def parse_ldr(text: str) -> Model:
    """Parse LDR/MPD text into a `Model` (sub-models + their type-1 references)."""
    submodels: dict[str, SubModel] = {}
    order: list[str] = []

    # MPD: split on "0 FILE". If there is no FILE marker, it's a single anonymous model.
    current = SubModel(name="__main__")
    submodels[_norm(current.name)] = current
    order.append(_norm(current.name))
    have_named = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        ltype = parts[0]
        if ltype == "0":
            # meta line; we care about FILE / NOFILE for MPD structure.
            if len(parts) >= 3 and parts[1].upper() == "FILE":
                name = _norm(" ".join(parts[2:]))
                if not have_named:
                    # First FILE: rename the anonymous main to this, keep it as main.
                    del submodels[_norm(current.name)]
                    order.pop()
                    current = SubModel(name=name)
                    submodels[name] = current
                    order.append(name)
                    have_named = True
                else:
                    current = SubModel(name=name)
                    submodels[name] = current
                    order.append(name)
            elif len(parts) >= 2 and parts[1].upper() == "NOFILE":
                pass  # boundary; next FILE opens a new sub-model
            continue
        if ltype == "1":
            # 1 colour x y z a b c d e f g h i file...
            if len(parts) < 15:
                continue  # malformed; skip
            color = int(parts[1])
            nums = list(map(float, parts[2:14]))
            target = _norm(" ".join(parts[14:]))
            current.refs.append((color, _mat_from_line(nums), target))
        # line types 2-5 are drawing primitives (only inside part .dat files); ignore.

    main = order[0]
    return Model(submodels=submodels, main=main)


def _is_submodel_ref(model: Model, target: str) -> bool:
    """True if `target` is a genuine sub-assembly to descend into.

    LDraw semantics: ``.dat`` = a part or primitive (a *leaf brick*), ``.ldr``/``.mpd`` =
    a model/sub-model (an assembly). OMR files often inline *custom parts* as ``.dat``
    sub-files so the MPD is self-contained; those are still single bricks and must not be
    exploded into their primitive edges/rings. So we recurse only into non-``.dat``
    sub-files that exist in this document.
    """
    t = _norm(target)
    if t not in model.submodels:
        return False
    return not t.endswith(".dat")


def _resolve_color(child_color: int, parent_color: int) -> int:
    if child_color in (COLOR_INHERIT, COLOR_EDGE_INHERIT):
        return parent_color
    return child_color


def flatten(model: Model, max_depth: int = 64) -> list[Brick]:
    """Recursively expand sub-models into a flat list of world-space library bricks.

    Preserves reference order (depth-first, matching file order) so downstream code can
    choose to keep or re-order it.
    """
    bricks: list[Brick] = []

    def recurse(name: str, xform: np.ndarray, color: int, depth: int):
        if depth > max_depth:
            raise RecursionError(f"sub-model nesting exceeded {max_depth} (cycle?): {name}")
        sub = model.submodels[name]
        for child_color, mat, target in sub.refs:
            rc = _resolve_color(child_color, color)
            world = xform @ mat
            if _is_submodel_ref(model, target):
                recurse(_norm(target), world, rc, depth + 1)
            else:
                bricks.append(Brick(part=target, color=rc, matrix=world))

    recurse(model.main, np.eye(4), COLOR_DEFAULT, 0)
    return bricks


def parse_file(path: str) -> Model:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_ldr(f.read())


def flatten_file(path: str) -> list[Brick]:
    return flatten(parse_file(path))


def _fmt_num(v: float) -> str:
    """Format a number the LDraw way: integers without a trailing ``.0``."""
    r = round(v)
    if abs(v - r) < 1e-6:
        return str(int(r))
    return f"{v:g}"


def brick_to_line(brick: Brick) -> str:
    """Serialise a world-space brick back to an LDraw type-1 line."""
    m = brick.matrix
    x, y, z = m[0, 3], m[1, 3], m[2, 3]
    a, b, c = m[0, 0], m[0, 1], m[0, 2]
    d, e, f = m[1, 0], m[1, 1], m[1, 2]
    g, h, i = m[2, 0], m[2, 1], m[2, 2]
    vals = [x, y, z, a, b, c, d, e, f, g, h, i]
    body = " ".join(_fmt_num(v) for v in vals)
    return f"1 {brick.color} {body} {brick.part}"


def write_ldr(bricks: list[Brick], path: str | None = None, step_between: bool = False) -> str:
    """Serialise bricks to LDR text (and optionally write to `path`)."""
    lines: list[str] = []
    for b in bricks:
        lines.append(brick_to_line(b))
        if step_between:
            lines.append("0 STEP")
    text = "\n".join(lines) + "\n"
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
