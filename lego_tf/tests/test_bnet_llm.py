"""LegoLLM (Track B) wrapper logic, exercised with a STUB backbone -- no HF download needed.

Verifies the parts that make the reuse work: warm-start writes the LEGO embedding from token
descriptions, the hooks return LEGO-vocab-width logits, and the reused grammar sampler
(`generate_batch`) drives the wrapper to valid, connector-valid builds. Needs `bricknet` (for the
grammar/decoder), so it runs on the GPU/Colab box, not the offline CPU box.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bricknet")          # grammar + decode live in bricknet; skip on the CPU box

import torch
import torch.nn as nn

from lego_tf.bnet.llm_backbone import LegoLLM, token_descriptions
from lego_tf.bnet.tokenizer import Vocab, decode

D = 32
BASE_VOCAB = 64


class _StubTrunk(nn.Module):
    """Stand-in for the Qwen trunk: (inputs_embeds) -> object with .last_hidden_state (B,S,d)."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None,
                use_cache=False, **kw):
        return type("O", (), {"last_hidden_state": self.lin(inputs_embeds)})()


class _FakeTok:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.pad_token_id, self.eos_token_id = 0, 1

    def _enc(self, t):
        ids = [(abs(hash(w)) % (self.vocab_size - 2)) + 2 for w in t.split()]
        return ids or [2]

    def __call__(self, text, add_special_tokens=True, padding=False, truncation=False,
                 max_length=None, return_tensors=None):
        many = not isinstance(text, str)
        seqs = [self._enc(t) for t in (text if many else [text])]
        if return_tensors == "pt":
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), self.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), T), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s)
                mask[r, :len(s)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": seqs if many else seqs[0]}


def _stub_model():
    vocab = Vocab()
    model = LegoLLM(_StubTrunk(D), nn.Embedding(BASE_VOCAB, D), d_model=D, ctx=64,
                    vocab=vocab, null_ids=[2, 3])
    model.tokenizer = _FakeTok(BASE_VOCAB)
    model.eval()
    return model, vocab


def test_warm_start_writes_lego_embedding():
    model, vocab = _stub_model()
    descs = token_descriptions(vocab, catalog=None)
    assert len(descs) == vocab.total
    model.warm_start(descs, model.tokenizer, device="cpu")

    # a specific row must equal the masked-mean of the backbone embedding of its description
    gid = vocab.gid("ANGLE", 90)
    enc = model.tokenizer([descs[gid]], padding=True, return_tensors="pt")
    ids, m = enc["input_ids"], enc["attention_mask"].unsqueeze(-1).float()
    want = (model.base_embed(ids) * m).sum(1) / m.sum(1).clamp(min=1.0)
    assert torch.allclose(model.lego_embed.weight[gid], want[0], atol=1e-5)
    assert torch.isfinite(model.lego_embed.weight).all()


def test_hooks_return_lego_width_logits():
    model, vocab = _stub_model()
    B = 3
    ids = torch.tensor([[vocab.BOS]] * B)
    cond = model.encode_prompt("a small red car")
    cp, npf = model._cond_prefixes(cond, B, "cpu")
    assert cp.shape[0] == B and cp.shape[2] == D
    logits = model._step_logits(ids, cond_prefix=cp)
    assert logits.shape == (B, vocab.total)             # sliced to the LEGO vocab
    g = model._gathered_logits(ids, torch.zeros(B, dtype=torch.long), cond_prefix=cp)
    assert g.shape == (B, vocab.total)


def test_reused_sampler_makes_valid_builds():
    import bricknet
    model, vocab = _stub_model()
    model.warm_start(token_descriptions(vocab, None), model.tokenizer, device="cpu")
    torch.manual_seed(0)
    for cond in (None, model.encode_prompt("a red race car")):
        streams = model.generate_batch(vocab, n=2, max_new=48, device="cpu",
                                       min_bricks=1, batch_size=2, cond=cond)
        assert len(streams) == 2
        for s in streams:
            assert s[0] == vocab.BOS
            tree = decode(s, vocab)                      # grammar-constrained -> always decodes
            assert len(tree.parts) >= 1
            bricknet.tree_to_graph(tree)                 # connector-valid by construction


def test_collision_free_sampler_if_meshes_present():
    model, vocab = _stub_model()
    model.warm_start(token_descriptions(vocab, None), model.tokenizer, device="cpu")
    torch.manual_seed(0)
    try:
        streams = model.generate_batch_cf(vocab, n=2, max_new=48, device="cpu",
                                          min_bricks=1, batch_size=2, max_retries=2,
                                          cond=model.encode_prompt("a car"))
    except FileNotFoundError:
        pytest.skip("inset collision meshes not available")
    assert len(streams) == 2
    for s in streams:
        decode(s, vocab)
