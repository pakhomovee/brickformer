"""Per-part connector enumeration + mate compatibility, reused from bricknet's catalog.

The compact tokenizer identifies a connector by its **flat index** `k` into a part's connector list
(`bricknet.graph._part_info(pid).conns`), instead of the old wide `(sub, polarity, conn)` integers.
This module is the thin reuse layer:
  - encode:  edge -> (parent flat index, child flat index)         [flat_indices]
  - decode:  (part, flat index) -> edge sub/conn fields + family    [edge_fields_from_flat]
  - masking: valid parent connectors / compatible child connectors  [valid_parent_conns / compatible_child_conns]

Compatibility follows bricknet's own mate rules (`_STUD_MATE_MAT`, `_AXLE_MATE_MAT`, in/on polarity)
so a masked edge is exactly one bricknet would accept -> generation is connector-valid by construction.
"""

from __future__ import annotations

from functools import lru_cache

from bricknet.core import StudSub, AxleSub
from bricknet.graph import _part_info, _tree_edge_row, _STUD_MATE_MAT, _AXLE_MATE_MAT, _SUB_ENC

# connector tuple layout: conns[k] = (kind, sub, polarity, conn_index, size)
_KIND, _SUB, _POL, _CONN = 0, 1, 2, 3
_STUD_KINDS = frozenset({"stud", "hole"})
_POLAR_KINDS = frozenset({"hinge", "ball", "fixed"})


def conns(pid: int):
    return _part_info(pid).conns


def n_conn(pid: int) -> int:
    return len(_part_info(pid).conns)


# ---- encode: tree edge -> flat connector indices ------------------------------------------------

def flat_indices(edge, parent_idx: int, child_idx: int, parent_pid: int, child_pid: int):
    """(parent flat index, child flat index) for a build-order edge. Reuses bricknet's own
    edge->row map and undoes its ball canonicalization (which may swap the a/b side)."""
    a, b, ac, bc, *_ = _tree_edge_row(edge, parent_idx, child_idx, parent_pid, child_pid)
    return (ac, bc) if a == parent_idx else (bc, ac)


# ---- decode: (part, flat index) -> edge fields --------------------------------------------------

def family_of(kind_p: str, kind_c: str) -> str:
    s = {kind_p, kind_c}
    if s <= _STUD_KINDS:
        return "stud"
    if kind_p == kind_c and kind_p in ("axle", "hinge", "fixed", "ball"):
        return kind_p
    raise ValueError(f"no family for connector kinds {kind_p}/{kind_c}")


def _sub_conn(c) -> tuple[int, int]:
    """connector tuple -> (edge sub-int, conn index), inverting bricknet's sub/polarity encoding."""
    kind, sub, pol, conn = c[_KIND], c[_SUB], c[_POL], c[_CONN]
    if kind in _STUD_KINDS:
        return int(StudSub[sub].value), conn
    if kind == "axle":
        return int(AxleSub[sub].value), conn
    return int(sub) * 2 + (0 if pol == "in" else 1), conn   # hinge/ball/fixed: in=even, on=odd


def edge_fields_from_flat(parent_pid: int, k_parent: int, child_pid: int, k_child: int):
    """(family, parent_sub, child_sub, parent_conn, child_conn) from the two flat indices."""
    cp, cc = _part_info(parent_pid).conns[k_parent], _part_info(child_pid).conns[k_child]
    fam = family_of(cp[_KIND], cc[_KIND])
    psub, pconn = _sub_conn(cp)
    csub, cconn = _sub_conn(cc)
    return fam, psub, csub, pconn, cconn


def family_from_flat(parent_pid: int, k_parent: int, child_pid: int, k_child: int) -> str:
    """Edge family from the two flat connector indices (for driving the DOF grammar)."""
    cp, cc = _part_info(parent_pid).conns[k_parent], _part_info(child_pid).conns[k_child]
    return family_of(cp[_KIND], cc[_KIND])


# ---- masking: which connectors form a real, compatible edge -------------------------------------

def _mate(cp, cc) -> bool:
    kp, sp, polp = cp[_KIND], cp[_SUB], cp[_POL]
    kc, sc, polc = cc[_KIND], cc[_SUB], cc[_POL]
    if {kp, kc} <= _STUD_KINDS and kp != kc:            # one stud, one hole
        return bool(_STUD_MATE_MAT[_SUB_ENC[sp], _SUB_ENC[sc]])
    if kp == kc == "axle":
        return bool(_AXLE_MATE_MAT[_SUB_ENC[sp], _SUB_ENC[sc]])
    if kp == kc and kp in _POLAR_KINDS:                 # same sub, opposite in/on polarity
        return sp == sc and polp is not None and polc is not None and polp != polc
    return False


@lru_cache(maxsize=200_000)
def compatible_child_conns(parent_pid: int, k_parent: int, child_pid: int) -> tuple[int, ...]:
    """Flat indices on the child part whose connector mates the parent's connector k_parent."""
    cp = _part_info(parent_pid).conns[k_parent]
    return tuple(k for k, cc in enumerate(_part_info(child_pid).conns) if _mate(cp, cc))


@lru_cache(maxsize=100_000)
def valid_parent_conns(parent_pid: int, child_pid: int) -> tuple[int, ...]:
    """Flat indices on the parent part that have at least one compatible connector on the child."""
    ccs = _part_info(child_pid).conns
    return tuple(kp for kp, cp in enumerate(_part_info(parent_pid).conns)
                if any(_mate(cp, cc) for cc in ccs))
