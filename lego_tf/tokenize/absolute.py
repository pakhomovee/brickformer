"""Absolute-pose tokenizer (representation v0).

Each brick becomes a small integer row. The common native-tokenizer layout is 5 ints
(``x y z rot_id part_id``); we add an explicit **colour** field so the encoding is *lossless*
on the on-grid domain -- this is what lets us assert the Stage-0 bit-exact round-trip gate.
(Dropping colour and re-origining/re-sorting is fine for generation but not lossless.)

Row layout (int64), one per brick::

    [x, y, z, rot_id, part_id, color]

Position is in LDU (LDraw units). On the on-grid domain (integer positions, orientation in
the 48 canonical set) encode/decode is exact up to an optional canonicalisation (translate
to a fixed origin and sort into a canonical build order) which preserves the *build* exactly
but not absolute placement/order. `decode` inverts `encode` losslessly when
``canonicalize=False`` (default).
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from lego_tf.data.ldraw import Brick
from lego_tf.tokenize.rotations import RotationCodebook

UNK = "<unk>"


class PartVocab:
    """Part filename <-> id. Id 0 is reserved for ``<unk>`` (excluded from generation)."""

    def __init__(self, parts: list[str] | None = None):
        self.id_to_part = [UNK]
        self.part_to_id = {UNK: 0}
        for p in parts or []:
            self.add(p)

    def add(self, part: str) -> int:
        p = part.lower()
        if p not in self.part_to_id:
            self.part_to_id[p] = len(self.id_to_part)
            self.id_to_part.append(p)
        return self.part_to_id[p]

    def id_of(self, part: str) -> int:
        return self.part_to_id.get(part.lower(), 0)

    def part_of(self, pid: int) -> str:
        return self.id_to_part[pid]

    def __len__(self) -> int:
        return len(self.id_to_part)

    @classmethod
    def from_bricks(cls, bricks_iter) -> "PartVocab":
        v = cls()
        for bricks in bricks_iter:
            for b in bricks:
                v.add(b.part)
        return v

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.id_to_part, f)

    @classmethod
    def load(cls, path: str) -> "PartVocab":
        with open(path) as f:
            parts = json.load(f)
        v = cls()
        v.id_to_part = list(parts)
        v.part_to_id = {p: i for i, p in enumerate(v.id_to_part)}
        return v


@dataclass
class Tokens:
    array: np.ndarray  # (N, 6) int64: x, y, z, rot_id, part_id, color

    def __len__(self) -> int:
        return int(self.array.shape[0])


class AbsolutePoseTokenizer:
    def __init__(self, part_vocab: PartVocab, rotations: RotationCodebook | None = None,
                 pos_tol: float = 1e-3):
        self.parts = part_vocab
        self.rotations = rotations or RotationCodebook()
        self.pos_tol = pos_tol

    # -- encode ---------------------------------------------------------------
    def encode(self, bricks: list[Brick], canonicalize: bool = False,
               require_grid: bool = True) -> Tokens:
        rows = []
        for b in bricks:
            pos = b.pos
            rp = np.round(pos)
            if require_grid and np.any(np.abs(pos - rp) > self.pos_tol):
                raise ValueError(
                    f"off-grid position {pos.tolist()} for part {b.part}; "
                    "off-grid parts need the fine-geom channel (representation v1+)"
                )
            rot_id = self.rotations.id_of(b.rot)  # raises KeyError if non-canonical
            part_id = self.parts.id_of(b.part)
            rows.append([int(rp[0]), int(rp[1]), int(rp[2]), rot_id, part_id, int(b.color)])
        arr = np.array(rows, dtype=np.int64) if rows else np.zeros((0, 6), np.int64)

        if canonicalize and len(arr):
            # Canonical build order: sort by y (height), then x, then z. LDraw y points
            # down, so ascending y is top->down; we sort descending y for bottom->up.
            order = np.lexsort((arr[:, 2], arr[:, 0], -arr[:, 1]))
            arr = arr[order]
            arr[:, :3] -= arr[:, :3].min(axis=0)  # translate min corner to origin
        return Tokens(array=arr)

    # -- decode ---------------------------------------------------------------
    def decode(self, tokens: Tokens) -> list[Brick]:
        bricks = []
        for row in tokens.array:
            x, y, z, rot_id, part_id, color = (int(v) for v in row)
            m = np.eye(4)
            m[:3, :3] = self.rotations.matrix_of(rot_id)
            m[:3, 3] = [x, y, z]
            bricks.append(Brick(part=self.parts.part_of(part_id), color=color, matrix=m))
        return bricks
