"""Shared segmented vocabulary for the flat model token stream (plan sec 4).

One contiguous id space is partitioned into segments; a global id decodes to (segment, value).
The decoder uses the grammar in `stream.py` to know which segment is valid at each position, so
the same id space is reused across fields (dynamic vocabulary masking at decode -- plan sec 4).

Segments::

    SPECIAL  BOS, EOS                      (sequence delimiters)
    PART     part ids (0 = <unk>)          from PartVocab
    COLOR    dense color ids               from the corpus' LDraw colour codes
    ROT      48 canonical orientations
    PTR      0 = ROOT (absolute), 1..P = parent-relative distance back
    PORT     connector-port index on a brick (parent_port / child_port share this segment)
    COORD    root absolute coordinate bins (model canonicalised to min-corner 0)
"""

from __future__ import annotations

from dataclasses import dataclass

SPECIALS = ["BOS", "EOS"]


@dataclass
class Vocab:
    n_parts: int
    n_colors: int
    n_rot: int
    max_ptr: int      # largest parent-relative distance representable (PTR ids: 0..max_ptr)
    max_port: int     # number of PORT ids (0..max_port-1)
    coord_max: int    # COORD ids: 0..coord_max (inclusive)

    def __post_init__(self):
        segs = {}
        off = 0
        segs["SPECIAL"] = (off, len(SPECIALS)); off += len(SPECIALS)
        segs["PART"] = (off, self.n_parts); off += self.n_parts
        segs["COLOR"] = (off, self.n_colors); off += self.n_colors
        segs["ROT"] = (off, self.n_rot); off += self.n_rot
        segs["PTR"] = (off, self.max_ptr + 1); off += self.max_ptr + 1
        segs["PORT"] = (off, self.max_port); off += self.max_port
        segs["COORD"] = (off, self.coord_max + 1); off += self.coord_max + 1
        self.segments = segs
        self.size = off
        self.BOS = self.gid("SPECIAL", 0)
        self.EOS = self.gid("SPECIAL", 1)

    def gid(self, segment: str, value: int) -> int:
        start, n = self.segments[segment]
        if not (0 <= value < n):
            raise ValueError(f"{segment} value {value} out of range [0,{n})")
        return start + value

    def decode(self, gid: int) -> tuple[str, int]:
        for seg, (start, n) in self.segments.items():
            if start <= gid < start + n:
                return seg, gid - start
        raise ValueError(f"gid {gid} out of vocab range [0,{self.size})")

    def segment_range(self, segment: str) -> range:
        start, n = self.segments[segment]
        return range(start, start + n)
