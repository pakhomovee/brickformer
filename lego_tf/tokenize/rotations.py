"""The 48 canonical axis-aligned brick orientations (signed permutation matrices).

A signed permutation matrix has exactly one non-zero entry (+1 or -1) per row and per
column. There are 3! * 2^3 = 48 of them, split 24 proper rotations (det +1) and 24
improper/mirror (det -1). This is the standard axis-aligned orientation set used across the
native-tokenizer LEGO literature.

We build them deterministically and assign stable integer ids by sorting on the flattened
matrix, so the id<->matrix mapping is reproducible without shipping a JSON file.
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np


def _all_signed_permutations() -> list[np.ndarray]:
    mats = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=np.int64)
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            mats.append(m)
    return mats


def _mat_key(m: np.ndarray) -> tuple:
    return tuple(int(v) for v in m.flatten())


class RotationCodebook:
    """Bidirectional map between the 48 canonical orientations and integer ids."""

    def __init__(self):
        mats = _all_signed_permutations()
        assert len(mats) == 48, len(mats)
        # Stable ordering by flattened matrix for reproducible ids.
        mats.sort(key=_mat_key)
        self.matrices = mats
        self.key_to_id = {_mat_key(m): i for i, m in enumerate(mats)}

    def __len__(self) -> int:
        return len(self.matrices)

    def id_of(self, rot: np.ndarray, atol: float = 1e-3) -> int:
        """Return the class id for a 3x3 rotation.

        Raises KeyError if the orientation is off the canonical grid (e.g. a hinge angle).
        We check the *rounded* matrix is a signed permutation AND that the input is actually
        close to it -- otherwise a 30 deg rotation would silently snap to the identity class
        instead of being rejected for the fine-geom channel.
        """
        r = np.round(rot)
        key = _mat_key(r.astype(np.int64))
        if key not in self.key_to_id or not np.allclose(rot, r, atol=atol):
            raise KeyError(f"non-canonical orientation: {rot.flatten().tolist()}")
        return self.key_to_id[key]

    def matrix_of(self, rot_id: int) -> np.ndarray:
        return self.matrices[rot_id].astype(np.float64)

    def is_canonical(self, rot: np.ndarray, atol: float = 1e-6) -> bool:
        r = np.round(rot)
        if _mat_key(r.astype(np.int64)) not in self.key_to_id:
            return False
        return bool(np.allclose(rot, r, atol=atol))
