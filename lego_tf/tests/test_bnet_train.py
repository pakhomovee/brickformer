"""Fast smoke test of the training loop: tiny model learns + constrained generation decodes."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("bricknet")

VAL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "val.npz")
pytestmark = pytest.mark.skipif(not os.path.isfile(VAL), reason="val.npz not downloaded")


def test_overfit_loop_learns_and_generates_valid():
    from lego_tf.bnet.train_overfit import run
    final_loss, ok = run(n=3, steps=150, d_model=64, n_layers=2, n_heads=2,
                         batch=3, max_len=120, split=VAL)
    assert final_loss < 5.0, f"loss did not drop enough: {final_loss}"
    assert ok, "constrained generation did not decode to a non-empty build"


def test_model_param_count_scales():
    import torch
    from lego_tf.bnet.model import LegoGPT, ModelConfig
    small = LegoGPT(ModelConfig(vocab_size=1000, d_model=64, n_layers=2, n_heads=2))
    big = LegoGPT(ModelConfig(vocab_size=1000, d_model=128, n_layers=4, n_heads=4))
    assert big.num_params() > small.num_params()
    # forward runs and is causal-shaped
    ids = torch.randint(0, 1000, (2, 16))
    logits, loss = small(ids, targets=ids)
    assert logits.shape == (2, 16, 1000) and loss.ndim == 0
