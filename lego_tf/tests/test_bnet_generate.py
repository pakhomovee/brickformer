"""Generation grammar + decode robustness (no dataset needed).

Covers two constrained-decoding invariants that guarantee every sampled stream is a valid tree:
  - the grammar forces exactly one root (brick 0 = ROOT, every later brick attaches via PTR);
  - decode stays token-synced even when a sampler emits an out-of-range pointer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bricknet")

from lego_tf.bnet.tokenizer import Vocab, decode, GrammarState
from lego_tf.bnet.model import LegoGPT, ModelConfig
from lego_tf.bnet.evaluate import evaluate


def test_grammar_forces_single_root():
    v = Vocab()
    g = GrammarState(v)
    g.step(v.gid("PART", 5)); g.step(v.gid("COLOR", 0))
    assert g.allowed_segments() == ["ROOT"]        # first brick must be the root
    g.step(v.ROOT)
    assert g.n_bricks == 1
    g.step(v.gid("PART", 6)); g.step(v.gid("COLOR", 0))
    assert g.allowed_segments() == ["PTR"]         # every later brick must attach


def test_decode_survives_bad_pointer():
    v = Vocab()
    toks = [v.BOS,
            v.gid("PART", 1), v.gid("COLOR", 0), v.ROOT,                       # brick 0: root
            v.gid("PART", 2), v.gid("COLOR", 0), v.gid("PTR", 50),             # brick 1: ptr past start
            v.gid("PSUB", 0), v.gid("CSUB", 0), v.gid("PCONN", 0), v.gid("CCONN", 0),
            v.gid("FAMILY", 0), v.gid("ANGLE", 0),                             # STUD dof
            v.gid("PART", 3), v.gid("COLOR", 0), v.gid("PTR", 1),              # brick 2: valid ptr
            v.gid("PSUB", 0), v.gid("CSUB", 0), v.gid("PCONN", 0), v.gid("CCONN", 0),
            v.gid("FAMILY", 4),                                               # FIXED, no dof
            v.EOS]
    tree = decode(toks, v)
    assert len(tree.parts) == 3        # bad pointer did NOT desync the stream
    assert len(tree.edges) == 1        # bad edge dropped; the valid one kept


def test_generation_always_decodes():
    v = Vocab()
    cfg = ModelConfig(vocab_size=v.total, d_model=32, n_layers=2, n_heads=2, max_seq=64)
    m = LegoGPT(cfg)
    for s in range(8):
        torch.manual_seed(s)
        toks = m.generate(v, max_new=64, device="cpu", constrained=True, greedy=False, min_bricks=2)
        assert toks[3] == v.ROOT       # first brick is the root
        tree = decode(toks, v)         # must not raise
        assert len(tree.parts) >= 2


def test_generate_batch_matches_single_and_decodes():
    v = Vocab()
    cfg = ModelConfig(vocab_size=v.total, d_model=32, n_layers=2, n_heads=2, max_seq=64)
    m = LegoGPT(cfg)
    # greedy is deterministic: batched rows must equal the single-sequence path
    single = [m.generate(v, max_new=64, device="cpu", constrained=True, greedy=True, min_bricks=2)
              for _ in range(4)]
    batch = m.generate_batch(v, 4, max_new=64, device="cpu", greedy=True, min_bricks=2, batch_size=4)
    assert single == batch
    # uneven chunking still yields valid, decodable streams
    torch.manual_seed(0)
    for toks in m.generate_batch(v, 5, max_new=64, device="cpu", min_bricks=2, batch_size=3):
        assert toks[3] == v.ROOT
        assert len(decode(toks, v).parts) >= 2


def test_evaluate_smoke(tmp_path):
    v = Vocab()
    cfg = ModelConfig(vocab_size=v.total, d_model=32, n_layers=2, n_heads=2, max_seq=64)
    m = LegoGPT(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg.__dict__}, ckpt)
    rep = evaluate(str(ckpt), n=3, device="cpu", max_new=64, collision=False, export=str(tmp_path / "e"))
    assert rep["n_requested"] == 3
    assert 0.0 <= rep["validity_rate"] <= 1.0
    assert "connector_valid_rate" in rep
    assert (tmp_path / "e" / "eval.json").exists()
