"""Generation grammar + decode robustness (no dataset needed).

Covers two constrained-decoding invariants that guarantee every sampled stream is a valid tree:
  - the grammar forces exactly one root (brick 0 = ROOT, every later brick attaches via PTR);
  - decode stays token-synced even when a sampler emits an out-of-range pointer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bricknet")

from lego_tf.bnet import connectors as K
from lego_tf.bnet.tokenizer import Vocab, decode, GrammarState, _FAM_STR_TO_INT, _DOF_SEGS, FIXED
from lego_tf.bnet.model import LegoGPT, ModelConfig
from lego_tf.bnet.evaluate import evaluate


def test_grammar_forces_single_root():
    v = Vocab()
    g = GrammarState(v)
    g.step(v.gid("PART", 5)); g.step(v.gid("COLOR", 0))
    assert g.allowed_ids() == [v.ROOT]             # first brick must be the root
    g.step(v.ROOT)
    assert g.n_bricks == 1
    g.step(v.gid("PART", 6)); g.step(v.gid("COLOR", 0))
    later = g.allowed_ids()                         # every later brick must attach via PTR
    assert later and all(v.seg_of(i) == "PTR" for i in later)
    assert v.ROOT not in later


def _dof_zeros(v, parent_pid, kp, child_pid, kc):
    """DOF tokens (all zero) matching the family the decoder will derive for this connector pair."""
    try:
        fam = _FAM_STR_TO_INT[K.family_from_flat(parent_pid, kp, child_pid, kc)]
    except Exception:
        fam = FIXED
    return [v.gid(seg, 0) for seg in _DOF_SEGS[fam]]


def test_decode_truncates_at_bad_pointer():
    """A bad pointer can't form an aligned edge, so decode truncates the build there (keeping the
    valid prefix) instead of desyncing -- bricknet's tree_to_graph assumes edge i connects part i+1,
    so a dropped mid-build edge would corrupt every later one."""
    v = Vocab()
    # decode resolves an out-of-range parent to brick 0's part_id (=1), so match its DOF count.
    dof = _dof_zeros(v, 1, 0, 2, 0)
    toks = [v.BOS,
            v.gid("PART", 1), v.gid("COLOR", 0), v.ROOT,                # brick 0: root
            v.gid("PART", 2), v.gid("COLOR", 0), v.gid("PTR", 50),      # brick 1: ptr past start
            v.gid("PCONN", 0), v.gid("CCONN", 0), *dof,
            v.EOS]
    tree = decode(toks, v)
    assert len(tree.parts) == 1 and len(tree.edges) == 0   # truncated cleanly at the bad brick


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
