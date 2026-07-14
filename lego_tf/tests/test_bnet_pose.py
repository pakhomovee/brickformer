"""v1 resolved-pose feature: delayed/leak-free per-token pose, and pose-aware model paths."""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bricknet")

from lego_tf.bnet import trees as T
from lego_tf.bnet.tokenizer import Vocab, encode_tree, decode, GrammarState
from lego_tf.bnet.model import LegoGPT, ModelConfig

VAL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "val.npz")
needs_val = pytest.mark.skipif(not os.path.isfile(VAL), reason="val.npz not downloaded")


@pytest.fixture(scope="module")
def vocab():
    return Vocab()


# ---- pose feature construction (needs a real graph to resolve geometry) --------------------------

@needs_val
def test_resolve_poses_aligns_with_parts(vocab):
    import bricknet
    tr = T.sample_tree(bricknet.load_graphs(VAL)[0], seed=0)
    poses = T.resolve_poses(tr)
    assert poses is not None and poses.shape == (len(tr.parts), 4, 4)


@needs_val
def test_pose_feature_rows_delayed_and_leakfree(vocab):
    import bricknet
    tr = T.sample_tree(bricknet.load_graphs(VAL)[0], seed=0)
    toks = encode_tree(tr, vocab)
    feats = T.pose_feature_rows(tr, toks, vocab)
    poses = T.resolve_poses(tr)
    assert feats.shape == (len(toks), T.POSE_DIM)
    assert np.all(feats[0] == 0)                       # BOS carries no pose
    # every token of brick i must carry P[i-1] (delayed) -- never its own pose P[i] (leak-free)
    gs = GrammarState(vocab)
    for i in range(1, len(toks)):
        nb = gs.n_bricks                               # bricks completed before this token
        expected = np.zeros(T.POSE_DIM, np.float32) if nb == 0 else T._pose_feat(poses[nb - 1])
        assert np.allclose(feats[i], expected)
        gs.step(toks[i])


# ---- pose-aware model (no dataset needed) --------------------------------------------------------

def test_v1_forward_with_and_without_pose(vocab):
    cfg = ModelConfig(vocab_size=vocab.total, d_model=32, n_layers=2, n_heads=2, max_seq=64,
                      use_pose=True)
    m = LegoGPT(cfg)
    assert m.pose_embed is not None
    ids = torch.randint(4, vocab.total, (2, 10))
    logits, _ = m(ids, pose=torch.randn(2, 10, T.POSE_DIM))
    assert logits.shape == (2, 10, vocab.total)
    logits2, _ = m(ids)                                # use_pose but pose=None -> token-only, no crash
    assert logits2.shape == (2, 10, vocab.total)


def test_v1_generate_batch_decodes(vocab):
    cfg = ModelConfig(vocab_size=vocab.total, d_model=32, n_layers=2, n_heads=2, max_seq=64,
                      use_pose=True)
    m = LegoGPT(cfg)
    torch.manual_seed(0)
    # default resolver (bundled catalog) -- exercises incremental pose resolution incl. failures
    for toks in m.generate_batch(vocab, 4, max_new=64, device="cpu", min_bricks=2, batch_size=4):
        assert toks[3] == vocab.ROOT
        assert len(decode(toks, vocab).parts) >= 2
    # explicit stub resolver: plumbing works without geometry
    stub = lambda t: np.ones(T.POSE_DIM, np.float32)
    for toks in m.generate_batch(vocab, 3, max_new=48, device="cpu", min_bricks=2, batch_size=3,
                                 resolve_pose=stub):
        assert len(decode(toks, vocab).parts) >= 2
