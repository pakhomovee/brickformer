"""Caption conditioning (SFT): the caption prefix steers generation, CFG blends, loss ignores the
prefix. Uses synthetic caption embeddings (no text encoder needed -- that lives in captions.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lego_tf.bnet.tokenizer import Vocab, decode
from lego_tf.bnet.model import LegoGPT, ModelConfig

COND = 16


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    v = Vocab()
    cfg = ModelConfig(vocab_size=v.total, d_model=64, n_layers=2, n_heads=4, max_seq=96,
                      cond_dim=COND, **kw)
    return v, LegoGPT(cfg)


def test_forward_shapes_and_prefix_stripped():
    v, m = _model()
    ids = torch.randint(4, v.total, (2, 10))
    # unconditional path unaffected
    assert m(ids)[0].shape == (2, 10, v.total)
    # conditioned: logits come back LEGO-aligned (caption prefix stripped), loss computes
    cond = torch.randn(2, 1, COND)
    logits, loss = m(ids[:, :-1], targets=ids[:, 1:], cond=cond)
    assert logits.shape == (2, 9, v.total) and loss.item() > 0
    # variable-length caption via mask (per-word path)
    cond3 = torch.randn(2, 3, COND)
    cmask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    assert m(ids, cond=cond3, cond_mask=cmask)[0].shape == (2, 10, v.total)


def test_grad_reaches_caption_pathway():
    v, m = _model()
    ids = torch.randint(4, v.total, (2, 12))
    _, loss = m(ids[:, :-1], targets=ids[:, 1:], cond=torch.randn(2, 1, COND))
    loss.backward()
    assert m.cond_proj.weight.grad.norm() > 0


def test_caption_steers_output():
    """Overfit c_A->s_A and c_B->s_B; the wrong caption must predict a sequence far worse."""
    v, m = _model(seed=1)
    torch.manual_seed(2)
    cA, cB = torch.randn(1, 1, COND), torch.randn(1, 1, COND)
    sA = torch.randint(4, v.total, (1, 12))
    sB = torch.randint(4, v.total, (1, 12))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(200):
        for c, s in ((cA, sA), (cB, sB)):
            _, l = m(s[:, :-1], targets=s[:, 1:], cond=c)
            opt.zero_grad(); l.backward(); opt.step()
    m.eval()

    def tf(c, s):
        with torch.no_grad():
            return m(s[:, :-1], targets=s[:, 1:], cond=c)[1].item()

    assert tf(cA, sA) < tf(cB, sA) - 0.1     # right caption fits its sequence much better
    assert tf(cB, sB) < tf(cA, sB) - 0.1


def test_conditioned_generation_and_cfg_decode():
    v, m = _model()
    cap = torch.randn(COND)
    for w in (1.0, 4.0):                       # cfg_weight 1 = plain conditional, >1 = guided
        streams = m.generate_batch(v, 3, max_new=80, device="cpu", min_bricks=2, batch_size=3,
                                   cond=cap, cfg_weight=w)
        assert all(len(decode(t, v).parts) >= 1 for t in streams)
    # a cond model still generates unconditionally when cond is omitted
    assert len(decode(m.generate_batch(v, 1, max_new=80, device="cpu", min_bricks=2, batch_size=1)[0],
                      v).parts) >= 1
