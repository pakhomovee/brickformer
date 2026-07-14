"""Build-order trees over bricknet graphs (structural layer for native tokenization).

A bricknet graph is pose-free: the spanning tree of typed connectors (with their DOF) fully
determines geometry via the catalog. We sample a build order, optionally truncate it at a brick
boundary (the interactive-completion signal), and coerce unknown colors -- then the native
tokenizer (see bnet/tokenizer.py) turns a tree into an integer token stream. No text, no
coordinates.
"""

from __future__ import annotations

import dataclasses
import functools

import bricknet
import numpy as np
from bricknet.core import Tree

POSE_DIM = 9  # per-token resolved-pose feature: translation (3) + rotation 6D (first two columns)


@functools.lru_cache(maxsize=1)
def catalog():
    """The bundled connector-annotated catalog (14.5k parts). Cached; load is not cheap."""
    return bricknet.load_catalog()


def _fallback_color() -> int:
    known = catalog().code_to_color
    return 0 if 0 in known else next(iter(known))


def coerce_colors(tree, fallback: int | None = None):
    """Remap any part color the catalog can't name to a known fallback.

    ~0.6% of BrickNet graphs carry color codes outside the catalog's 219-color map (a few
    real-but-unmapped codes plus some corrupt packed-RGB ints). Coercing keeps the structure
    (only a handful of bricks' colour is approximated). Returns the tree unchanged when every
    color is already known.
    """
    known = catalog().code_to_color
    if all(p.color in known for p in tree.parts):
        return tree
    fb = _fallback_color() if fallback is None else fallback
    parts = tuple(p if p.color in known else dataclasses.replace(p, color=fb) for p in tree.parts)
    return Tree(parts=parts, edges=tree.edges)


def sample_tree(graph, *, component: int = 0, seed: int = 0, collision_free: bool = True):
    """Sample a build order (spanning tree) for one component of a graph.

    Prefers a collision-free order (buildable, matches inference); falls back to a random order
    if the collision-free sampler can't produce one.
    """
    if collision_free:
        try:
            return bricknet.sample_collision_free_tree(graph, component=component, seed=seed)
        except Exception:
            pass
    return bricknet.sample_tree(graph, component=component, method="random", seed=seed)


def brick_count(tree) -> int:
    return len(tree.parts)


def truncate_tree(tree, k: int):
    """First `k` bricks as a valid partial build (edges kept only among those bricks)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    parts = tree.parts[:k]
    edges = tuple(e for e in tree.edges if e.parent < k and e.child < k)
    return Tree(parts=parts, edges=edges)


def resolve_poses(tree):
    """Resolved world pose (4x4) of each brick, aligned to `tree.parts`; None if the tree's
    connectors don't realize as geometry (catalog-only -- no collision meshes needed)."""
    try:
        mats = bricknet.decode_graph(bricknet.tree_to_graph(tree))
    except Exception:
        return None
    return np.stack([np.asarray(m, dtype=np.float32) for m in mats])  # (N, 4, 4)


def _pose_feat(mat) -> np.ndarray:
    """4x4 world matrix -> 9-float feature: translation + rotation 6D (first two columns)."""
    return np.concatenate([mat[:3, 3], mat[:3, 0], mat[:3, 1]]).astype(np.float32)


def pose_feature_rows(tree, toks, vocab) -> np.ndarray:
    """Per-token pose feature aligned to `toks` (which must be encode_tree(tree)): each token of
    brick i carries the resolved pose of the PREVIOUS brick, P[i-1] (zeros for BOS and the root's
    tokens). Leak-free (a brick's own pose never sits on its own tokens) and inference-consistent
    (P[i-1] is known once brick i-1 is placed). Returns float32 (len(toks), POSE_DIM); all zeros if
    the tree's poses can't be resolved."""
    from lego_tf.bnet.tokenizer import GrammarState  # local: tokenizer imports trees.coerce_colors

    feats = np.zeros((len(toks), POSE_DIM), dtype=np.float32)
    poses = resolve_poses(tree)
    if poses is None:
        return feats
    gs = GrammarState(vocab)                       # started AFTER BOS, like generation
    for t in range(1, len(toks)):                  # toks[0] is BOS -> row stays zero
        nb = gs.n_bricks                           # bricks completed BEFORE this token
        if nb >= 1:
            feats[t] = _pose_feat(poses[nb - 1])
        gs.step(toks[t])
    return feats
