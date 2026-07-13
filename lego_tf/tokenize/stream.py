"""Flatten an AttachSequence into a single int token stream over the shared Vocab.

Per-brick token group (grammar-conditional on the pointer, plan sec 3/4)::

    attached:  PART COLOR ROT PTR(dist>=1) PORT(parent) PORT(child)      -> 6 tokens
    root:      PART COLOR ROT PTR(=ROOT)   COORD(x) COORD(y) COORD(z)    -> 7 tokens

wrapped in BOS ... EOS. Root coordinates use zig-zag so signed LDU map to non-negative COORD
ids, keeping the round-trip exact (including absolute position) without canonicalising.
`allowed_next_segment` gives the one grammar-valid segment for the next position -- the hook a
decoder uses to mask logits (dynamic vocabulary masking).
"""

from __future__ import annotations

from lego_tf.tokenize.attach import AttachSequence, AttachToken
from lego_tf.tokenize.absolute import PartVocab
from lego_tf.tokenize.vocab import Vocab

DEFAULT_COLOR = 16


class ColorVocab:
    """Dense id <-> LDraw colour code, id 0 reserved for the default colour (16)."""

    def __init__(self, codes: list[int] | None = None):
        self.id_to_code = [DEFAULT_COLOR]
        self.code_to_id = {DEFAULT_COLOR: 0}
        for c in codes or []:
            self.add(c)

    def add(self, code: int) -> int:
        if code not in self.code_to_id:
            self.code_to_id[code] = len(self.id_to_code)
            self.id_to_code.append(code)
        return self.code_to_id[code]

    def id_of(self, code: int) -> int:
        return self.code_to_id.get(code, 0)

    def code_of(self, cid: int) -> int:
        return self.id_to_code[cid]

    def __len__(self):
        return len(self.id_to_code)


def build_vocab(sequences: list[AttachSequence], parts: PartVocab, colors: ColorVocab,
                margin_ptr: int = 0) -> Vocab:
    """Size a Vocab tightly to a corpus of AttachSequences (data-driven segment sizes)."""
    max_ptr = 1
    max_port = 1
    coord_max = 0
    for seq in sequences:
        for i, tok in enumerate(seq.tokens):
            if tok.parent < 0:
                for c in tok.pos:
                    coord_max = max(coord_max, _zigzag(int(c)))
            else:
                max_ptr = max(max_ptr, i - tok.parent)
                max_port = max(max_port, tok.parent_port, tok.child_port)
    return Vocab(
        n_parts=len(parts),
        n_colors=len(colors),
        n_rot=48,
        max_ptr=max_ptr + margin_ptr,
        max_port=max_port + 1,   # ids 0..max_port
        coord_max=coord_max,
    )


def _zigzag(n: int) -> int:
    return 2 * n if n >= 0 else -2 * n - 1


def _unzigzag(z: int) -> int:
    return z // 2 if z % 2 == 0 else -(z + 1) // 2


def encode_stream(seq: AttachSequence, parts: PartVocab, colors: ColorVocab, vocab: Vocab) -> list[int]:
    ids = [vocab.BOS]
    for i, tok in enumerate(seq.tokens):
        ids.append(vocab.gid("PART", tok.part_id))
        ids.append(vocab.gid("COLOR", colors.id_of(tok.color)))
        ids.append(vocab.gid("ROT", tok.rot_id))
        if tok.parent < 0:
            ids.append(vocab.gid("PTR", 0))  # ROOT
            for c in tok.pos:
                ids.append(vocab.gid("COORD", _zigzag(int(c))))
        else:
            ids.append(vocab.gid("PTR", i - tok.parent))
            ids.append(vocab.gid("PORT", tok.parent_port))
            ids.append(vocab.gid("PORT", tok.child_port))
    ids.append(vocab.EOS)
    return ids


def decode_stream(ids: list[int], colors: ColorVocab, vocab: Vocab) -> AttachSequence:
    assert vocab.decode(ids[0]) == ("SPECIAL", 0), "stream must start with BOS"
    seq = AttachSequence()
    p = 1
    while True:
        seg, val = vocab.decode(ids[p])
        if seg == "SPECIAL" and val == 1:  # EOS
            break
        assert seg == "PART", f"expected PART at {p}, got {seg}"
        part_id = val; p += 1
        seg, val = vocab.decode(ids[p]); assert seg == "COLOR"; color = colors.code_of(val); p += 1
        seg, val = vocab.decode(ids[p]); assert seg == "ROT"; rot_id = val; p += 1
        seg, val = vocab.decode(ids[p]); assert seg == "PTR"; ptr = val; p += 1
        i = len(seq.tokens)
        if ptr == 0:
            coords = []
            for _ in range(3):
                seg, val = vocab.decode(ids[p]); assert seg == "COORD"; coords.append(_unzigzag(val)); p += 1
            seq.tokens.append(AttachToken(part_id, color, rot_id, parent=-1,
                                          parent_port=0, child_port=0, pos=tuple(coords)))
        else:
            seg, pp = vocab.decode(ids[p]); assert seg == "PORT"; p += 1
            seg, cp = vocab.decode(ids[p]); assert seg == "PORT"; p += 1
            seq.tokens.append(AttachToken(part_id, color, rot_id, parent=i - ptr,
                                          parent_port=pp, child_port=cp))
    return seq


def allowed_next_segment(prefix: list[int], vocab: Vocab) -> str:
    """Grammar: the single segment valid at the next position given a token prefix.

    Returns one of the segment names, or "PART_OR_EOS" when a new brick or end may follow.
    This is what a constrained decoder consults to mask logits.
    """
    if not prefix:
        return "BOS"
    # walk the grammar from the last BOS
    state = "brick"  # expecting PART or EOS
    field = None
    ptr_is_root = None
    coord_left = 0
    port_left = 0
    for gid in prefix[1:]:
        seg, val = vocab.decode(gid)
        if state == "brick":
            if seg == "SPECIAL":  # EOS
                return "END"
            state, field = "in_brick", "COLOR"
        elif field == "COLOR":
            field = "ROT"
        elif field == "ROT":
            field = "PTR"
        elif field == "PTR":
            if val == 0:
                ptr_is_root, coord_left, field = True, 3, "COORD"
            else:
                ptr_is_root, port_left, field = False, 2, "PORT"
        elif field == "COORD":
            coord_left -= 1
            if coord_left == 0:
                state, field = "brick", None
        elif field == "PORT":
            port_left -= 1
            if port_left == 0:
                state, field = "brick", None
    if state == "brick":
        return "PART_OR_EOS"
    return field  # the next expected field segment
