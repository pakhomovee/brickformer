"""Native LEGO tokens on a pretrained LLM backbone (Track B).

Wraps a HuggingFace causal LM (default Qwen2.5-0.5B) so our compact native LEGO tokens ride on top of
a pretrained model, keeping the whole win over the BrickNet paper -- native compact tokens *and*
validity-by-construction decoding (grammar + connector-aware + collision-free) -- while adding the
LLM's priors. It reuses the constrained/collision-free/CFG sampler in `model.py` verbatim: `LegoLLM`
subclasses `LegoGPT` and overrides only the five hooks the sampler touches
(`_cond_prefixes`, `_step_logits`, `_gathered_logits`, plus `cfg`/`pose_embed`), so
`generate_batch` / `generate_batch_cf` work unchanged.

Design (differs from a naive vocab resize):
  - The backbone vocabulary and weights stay frozen; LoRA adapters train the trunk.
  - A separate trainable `lego_embed` (n_lego x d) is the LEGO **input embedding** and, tied, the
    LEGO **output head** (`F.linear(h, lego_embed.weight)`). So only the new LEGO tokens + LoRA move,
    the frozen text head (~152k wide) is never materialized, and generation stays memory-safe.
  - The caption is fed as an ordinary text prefix (the backbone's own tokenizer), embedded by the
    frozen input embedding; classifier-free guidance blends it against a null-caption prefix.
  - Each LEGO embedding is **warm-started** from the backbone's text encoding of what the token means
    ("lego brick 3001", "colour red", "rotation angle 90 degrees", ...), so the pretrained priors
    transfer into our token space instead of starting cold.

`transformers` / `peft` are imported lazily (only on the GPU/Colab box, like `captions.py`); the
low-level `__init__` takes pre-built modules so a stub backbone can exercise the reuse logic in tests
without downloading a model.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lego_tf.bnet.model import LegoGPT, ModelConfig
from lego_tf.bnet.tokenizer import SLIDE_MIN, Vocab

DEFAULT_BACKBONE = "Qwen/Qwen2.5-0.5B"
BUILD_TEMPLATE = "Build this LEGO model: {caption}"
NULL_TEXT = "Build this LEGO model:"          # CFG unconditional prefix (same template, no description)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ----- warm-start token descriptions -------------------------------------------------------------

def _part_name(catalog, k: int) -> str:
    """Human-ish name for bricknet part id k, if the catalog exposes one; else the numeric id."""
    if catalog is not None:
        for getter in ("part_name", "name_of", "describe"):
            try:
                nm = getattr(catalog, getter)(k)
                if nm:
                    return str(nm)
            except Exception:
                pass
        for attr in ("parts", "part_list", "names"):
            try:
                nm = getattr(catalog, attr)[k]
                nm = getattr(nm, "name", nm)
                if nm:
                    return str(nm)
            except Exception:
                pass
    return f"number {k}"


def _color_name(catalog, code: int) -> str:
    if catalog is not None:
        try:
            info = catalog.code_to_color[code]
            return str(getattr(info, "name", info))
        except Exception:
            pass
    return f"code {code}"


def token_descriptions(vocab: Vocab, catalog=None) -> list[str]:
    """A short natural-language description per LEGO token id (index == token id in [0, total))."""
    descs: list[str] = []
    for gid in range(vocab.total):
        seg = vocab.seg_of(gid)
        loc = vocab.local(gid, seg)
        if seg == "PAD":
            d = "padding"
        elif seg == "BOS":
            d = "start of a lego model"
        elif seg == "EOS":
            d = "end of a lego model"
        elif seg == "ROOT":
            d = "the first brick"
        elif seg == "PART":
            d = f"lego brick part {_part_name(catalog, loc)}"
        elif seg == "COLOR":
            d = f"colour {_color_name(catalog, vocab.idx_to_color[loc])}"
        elif seg == "PTR":
            d = f"attach to the brick {loc} steps earlier"
        elif seg == "PCONN":
            d = f"parent connection point {loc}"
        elif seg == "CCONN":
            d = f"child connection point {loc}"
        elif seg == "FLIP":
            d = "flipped orientation" if loc else "normal orientation"
        elif seg == "ANGLE":
            d = f"rotation angle {loc} degrees"
        elif seg == "SLIDE":
            d = f"axle slide offset {loc + SLIDE_MIN}"
        else:
            d = "lego token"
        descs.append(d)
    return descs


class LegoLLM(LegoGPT):
    """Pretrained backbone + native LEGO tokens. Reuses LegoGPT's generation via the five hooks.

    Construct with `LegoLLM.build(...)` on a GPU/Colab box (loads the HF backbone). The bare
    `__init__` wires pre-built modules and is what `build`, `load`, and the tests call.
    """

    def __init__(self, trunk, base_embed, d_model: int, ctx: int, *, vocab: Vocab | None = None,
                 peft_model=None, backbone_id: str | None = None, null_ids=None,
                 lego_embed: nn.Embedding | None = None):
        nn.Module.__init__(self)                      # bypass LegoGPT.__init__ (no native params)
        self.vocab = vocab or Vocab()
        n_lego = self.vocab.total
        # config the reused sampler reads (max_seq for the window, pose_dim unused since pose_embed=None)
        self.cfg = ModelConfig(vocab_size=n_lego, d_model=d_model, max_seq=ctx)
        self.pose_embed = None                        # LLM path has no resolved-pose input
        self.backbone_id = backbone_id
        self._peft = peft_model                       # registered as a submodule (owns LoRA params)
        self.trunk = trunk                            # Qwen2Model with LoRA injected in place
        self.base_embed = base_embed                  # frozen text input embedding (for captions)
        self.lego_embed = lego_embed or nn.Embedding(n_lego, d_model)
        self.register_buffer("null_ids",
                             torch.as_tensor(null_ids if null_ids is not None else [], dtype=torch.long),
                             persistent=True)

    # -- construction ----------------------------------------------------------------------------
    @classmethod
    def build(cls, backbone_id: str = DEFAULT_BACKBONE, ctx: int = 1024, lora_r: int = 16,
              lora_alpha: int = 32, lora_dropout: float = 0.05, device: str | None = None,
              vocab: Vocab | None = None, warm_start: bool = True):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vocab = vocab or Vocab()

        tok = AutoTokenizer.from_pretrained(backbone_id)
        base = AutoModelForCausalLM.from_pretrained(backbone_id)
        base.config.use_cache = False
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
        peft_model = get_peft_model(base, lora)       # injects LoRA into base.model in place
        trunk = base.model                            # same object -> runs LoRA when called directly
        base_embed = trunk.embed_tokens               # frozen (peft froze all non-LoRA params)
        d_model = base.config.hidden_size

        null_ids = tok(NULL_TEXT, add_special_tokens=False)["input_ids"]
        self = cls(trunk, base_embed, d_model, ctx, vocab=vocab, peft_model=peft_model,
                   backbone_id=backbone_id, null_ids=null_ids)
        self.tokenizer = tok
        self.to(device)
        if warm_start:
            catalog = _try_catalog()
            self.warm_start(token_descriptions(vocab, catalog), tok, device=device)
        return self

    @torch.no_grad()
    def warm_start(self, descriptions: list[str], tokenizer, device: str = "cpu", chunk: int = 512):
        """Set lego_embed[i] = mean backbone-embedding of tokenize(descriptions[i]).
        Rows whose description tokenizes empty fall back to the mean text embedding."""
        emb = self.base_embed.weight
        fallback = emb.mean(0)
        rows = torch.empty(self.vocab.total, emb.shape[1], device=emb.device, dtype=emb.dtype)
        for s in range(0, len(descriptions), chunk):
            batch = descriptions[s:s + chunk]
            enc = tokenizer(batch, add_special_tokens=False, padding=True, return_tensors="pt")
            ids = enc["input_ids"].to(emb.device)
            m = enc["attention_mask"].to(emb.device).unsqueeze(-1).to(emb.dtype)   # (b,T,1)
            pooled = (self.base_embed(ids) * m).sum(1) / m.sum(1).clamp(min=1.0)   # (b,d)
            empty = m.sum(1).squeeze(-1) == 0
            pooled[empty] = fallback
            rows[s:s + len(batch)] = pooled
        self.lego_embed.weight.copy_(rows.to(self.lego_embed.weight.dtype))

    # -- the five hooks the reused sampler calls -------------------------------------------------
    def _cond_prefixes(self, cond, B, device):
        """cond = caption token ids (backbone vocab) -> (caption prefix, null prefix), each (B,C,d)."""
        if cond is None or self.base_embed is None:
            return None, None
        cap = torch.as_tensor(cond, dtype=torch.long, device=device).reshape(1, -1)
        cp = self.base_embed(cap).expand(B, -1, -1)
        nul = self.null_ids.to(device).reshape(1, -1)
        npf = self.base_embed(nul).expand(B, -1, -1) if nul.numel() else cp[:, :1] * 0
        return cp, npf

    @torch.no_grad()
    def _step_logits(self, ids, pose=None, cond_prefix=None):
        x = self.lego_embed(ids)
        if cond_prefix is not None:
            x = torch.cat([cond_prefix, x], dim=1)
        h = self.trunk(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]   # (B, d)
        return F.linear(h, self.lego_embed.weight)                                  # (B, n_lego)

    @torch.no_grad()
    def _gathered_logits(self, ids, gidx, pose=None, cond_prefix=None):
        x = self.lego_embed(ids)
        C = 0
        if cond_prefix is not None:
            x = torch.cat([cond_prefix, x], dim=1)
            C = cond_prefix.shape[1]
        h = self.trunk(inputs_embeds=x, use_cache=False).last_hidden_state           # (B, C+T, d)
        hg = h[torch.arange(ids.shape[0], device=ids.device), gidx + C]              # (B, d)
        return F.linear(hg, self.lego_embed.weight)

    # -- training forward (used by train_llm.py) -------------------------------------------------
    def forward(self, prefix_ids, prefix_mask, lego_ids, targets=None, ignore_index=-100):
        """prefix_ids/prefix_mask: (B, Cp) LEFT-padded caption text (per-row, may be null via CFG
        drop). lego_ids: (B, T) LEGO-local ids (BOS..EOS, right-padded with PAD). Loss is next-token
        CE over the LEGO vocab on the LEGO positions only (caption + pad ignored)."""
        Cp = prefix_ids.shape[1]
        ecap = self.base_embed(prefix_ids)
        eleg = self.lego_embed(lego_ids)
        x = torch.cat([ecap, eleg], dim=1)                                          # (B, Cp+T, d)
        lego_mask = (lego_ids != self.vocab.PAD).long()
        attn = torch.cat([prefix_mask, lego_mask], dim=1)                           # (B, Cp+T)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)                                     # left-pad safe
        h = self.trunk(inputs_embeds=x, attention_mask=attn, position_ids=pos,
                       use_cache=False).last_hidden_state
        hp = h[:, Cp - 1:-1]                                                         # predicts lego_ids[t]
        logits = F.linear(hp, self.lego_embed.weight)                               # (B, T, n_lego)
        loss = None
        if targets is not None:
            tgt = targets.clone()
            tgt[targets == self.vocab.PAD] = ignore_index
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss

    # -- prompt encoding + persistence -----------------------------------------------------------
    def encode_prompt(self, text: str) -> list[int]:
        return self.tokenizer(BUILD_TEMPLATE.format(caption=text), add_special_tokens=False)["input_ids"]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        if self._peft is not None:
            self._peft.save_pretrained(out_dir)                                     # LoRA adapter
        torch.save(self.lego_embed.state_dict(), os.path.join(out_dir, "lego_embed.pt"))
        meta = {"backbone": self.backbone_id, "ctx": self.cfg.max_seq,
                "d_model": self.cfg.d_model, "null_ids": self.null_ids.tolist(),
                "is_llm": True}
        with open(os.path.join(out_dir, "lego_llm.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, out_dir: str, device: str | None = None, vocab: Vocab | None = None):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vocab = vocab or Vocab()
        meta = json.load(open(os.path.join(out_dir, "lego_llm.json")))
        tok = AutoTokenizer.from_pretrained(meta["backbone"])
        base = AutoModelForCausalLM.from_pretrained(meta["backbone"])
        base.config.use_cache = False
        peft_model = PeftModel.from_pretrained(base, out_dir)                        # re-inject LoRA
        trunk = base.model
        self = cls(trunk, trunk.embed_tokens, meta["d_model"], meta["ctx"], vocab=vocab,
                   peft_model=peft_model, backbone_id=meta["backbone"], null_ids=meta["null_ids"])
        self.lego_embed.load_state_dict(torch.load(os.path.join(out_dir, "lego_embed.pt"),
                                                   map_location="cpu"))
        self.tokenizer = tok
        self.to(device)
        return self


def _try_catalog():
    try:
        from lego_tf.bnet.trees import catalog
        return catalog()
    except Exception:
        return None
