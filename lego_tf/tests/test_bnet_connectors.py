"""Compact connector encoding: mate-compatibility masks + connector-valid-by-construction decoding.

The tokenizer identifies a connector by a compact per-part flat index and the grammar masks PTR /
PCONN / CCONN to real, mutually-compatible connectors. This is the connector analogue of the
structural grammar: every sampled stream must reconstruct to a build whose (part, connector) pairs
physically realize -- i.e. `bricknet.tree_to_graph` never raises -- even from an *untrained* model.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bricknet")

import bricknet

from lego_tf.bnet import connectors as K
from lego_tf.bnet.connectors import _mate, _part_info
from lego_tf.bnet.tokenizer import Vocab, decode
from lego_tf.bnet.model import LegoGPT, ModelConfig


def test_masks_are_mutually_consistent():
    """valid_parent_conns/compatible_child_conns agree with bricknet's mate rule, and a valid
    parent connector always has at least one compatible child connector."""
    checked = 0
    for ppid in range(40):
        for cpid in range(40):
            for kp in K.valid_parent_conns(ppid, cpid):
                cc = K.compatible_child_conns(ppid, kp, cpid)
                assert cc, "a valid parent connector must have a compatible child connector"
                pc = _part_info(ppid).conns[kp]
                assert all(_mate(pc, _part_info(cpid).conns[kc]) for kc in cc)
                # and it must be the *complete* set of mates
                full = [k for k, c in enumerate(_part_info(cpid).conns) if _mate(pc, c)]
                assert list(cc) == full
                checked += 1
    assert checked > 0, "expected some compatible connector pairs among the first parts"


def _has_meshes():
    try:
        from bricknet.collision import data_dir
        return (data_dir() / "inset").is_dir()
    except Exception:
        return False


@pytest.mark.skipif(not _has_meshes(), reason="inset collision meshes not available")
@pytest.mark.parametrize("use_pose", [False, True])   # pose path keeps a per-token stream aligned
def test_collision_free_decoding_by_construction(use_pose):
    """generate_batch_cf rejects+resamples colliding bricks, so every build is collision-free by
    construction: bricknet's own collision_free_prefix equals the part count for every stream."""
    from bricknet.score import collision_free_prefix
    v = Vocab()
    cfg = ModelConfig(vocab_size=v.total, d_model=32, n_layers=2, n_heads=2, max_seq=128,
                      use_pose=use_pose)
    m = LegoGPT(cfg)                   # untrained: the collision scene must guarantee it regardless
    torch.manual_seed(0)
    streams = m.generate_batch_cf(v, 6, max_new=128, device="cpu", min_bricks=2, batch_size=6,
                                  max_retries=6, temperature=1.0)
    n = 0
    for toks in streams:
        tree = decode(toks, v)
        if not tree.parts:
            continue
        n += 1
        assert collision_free_prefix(tree) == len(tree.parts)   # no collision anywhere in the build
    assert n > 0


def test_generation_is_connector_valid_by_construction():
    """An untrained model, constrained only by the connector-aware grammar, must still produce
    100%-connector-valid builds: every decoded tree passes tree_to_graph without raising."""
    v = Vocab()
    cfg = ModelConfig(vocab_size=v.total, d_model=48, n_layers=2, n_heads=2, max_seq=128)
    m = LegoGPT(cfg)
    torch.manual_seed(0)
    streams = m.generate_batch(v, 16, max_new=128, device="cpu", min_bricks=3, batch_size=16)
    n = valid = 0
    for toks in streams:
        tree = decode(toks, v)
        if not tree.parts:
            continue
        n += 1
        bricknet.tree_to_graph(tree)   # must not raise -- every edge is a real, compatible connector
        valid += 1
    assert n > 0
    assert valid == n                  # connector-valid by construction, even untrained
