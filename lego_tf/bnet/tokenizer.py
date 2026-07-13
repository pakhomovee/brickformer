"""Native LEGO tokenizer: bricknet build-order Tree <-> integer token stream.

A bricknet graph is pose-free, so a brick along the build-order tree is fully specified by its
part + color and (for non-root bricks) the attachment edge to an already-placed brick: which
sub-parts and connectors mate, the connector family, and the family's DOF. No coordinates.

Per-brick token group (segmented vocab -> a constrained-decoding grammar can mask each position
to one field):

    root brick :  PART COLOR ROOT
    non-root   :  PART COLOR PTR PSUB CSUB PCONN CCONN FAMILY <dof...>
      stud  dof: ANGLE(yaw)
      fixed dof: (none)
      hinge dof: FLIP ANGLE(yaw)
      axle  dof: FLIP ANGLE(yaw) SLIDE
      ball  dof: ANGLE(rx) ANGLE(ry) ANGLE(rz)

Angles are integer degrees, so 360 one-degree bins are *exact*; `slide` is an integer offset in
[-128, 127]. The stream is thus discrete and (barring rare slide clamping) losslessly reversible:
tokens -> Tree -> bricknet graph -> scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields

from bricknet.core import Tree, Part, StudEdge, AxleEdge, HingeEdge, BallEdge, FixedEdge

from lego_tf.bnet.trees import catalog, coerce_colors

# families (stable order = token id)
STUD, AXLE, HINGE, BALL, FIXED = range(5)
_FAMILY_OF = {"StudEdge": STUD, "AxleEdge": AXLE, "HingeEdge": HINGE, "BallEdge": BALL, "FixedEdge": FIXED}
_EDGE_CLS = {STUD: StudEdge, AXLE: AxleEdge, HINGE: HingeEdge, BALL: BallEdge, FIXED: FixedEdge}

SLIDE_MIN, SLIDE_MAX = -512, 511  # inclusive; token index = slide - SLIDE_MIN (observed |slide|<=322)

# segment name -> size (order here defines vocab layout). Specials first. Field sizes are set from
# the true data ceiling, not the val split: connector indices reach ~2303 (a 48x48 baseplate has
# 2304 studs), so CONN spans a 64x64 baseplate (4096); sub-part indices reached 123, slide +-322.
# prepare_data skips (and counts) any structure that still exceeds a cap.
_SEGMENTS = [
    ("PAD", 1), ("BOS", 1), ("EOS", 1), ("ROOT", 1),
    ("PART", 14583),
    ("COLOR", 219),
    ("PTR", 256),
    ("PSUB", 256),
    ("CSUB", 256),
    ("PCONN", 4096),
    ("CCONN", 4096),
    ("FAMILY", 5),
    ("FLIP", 2),
    ("ANGLE", 360),
    ("SLIDE", SLIDE_MAX - SLIDE_MIN + 1),
]


class Vocab:
    """Segmented vocabulary: contiguous id range per field, plus gid()/split() helpers."""

    def __init__(self, segments=_SEGMENTS):
        self.size_of: dict[str, int] = {}
        self.offset: dict[str, int] = {}
        off = 0
        for name, size in segments:
            self.offset[name] = off
            self.size_of[name] = size
            off += size
        self.total = off
        # color code <-> dense index (catalog knows 219 codes)
        codes = sorted(catalog().code_to_color)
        self.color_to_idx = {c: i for i, c in enumerate(codes)}
        self.idx_to_color = codes
        # scalar specials
        self.PAD = self.offset["PAD"]
        self.BOS = self.offset["BOS"]
        self.EOS = self.offset["EOS"]
        self.ROOT = self.offset["ROOT"]

    def gid(self, seg: str, local: int) -> int:
        if not 0 <= local < self.size_of[seg]:
            raise ValueError(f"{seg} local {local} out of range [0,{self.size_of[seg]})")
        return self.offset[seg] + local

    def seg_of(self, gid: int) -> str:
        for name, off in self.offset.items():
            if off <= gid < off + self.size_of[name]:
                return name
        raise ValueError(f"gid {gid} out of vocab")

    def local(self, gid: int, seg: str) -> int:
        return gid - self.offset[seg]

    def seg_range(self, seg: str) -> range:
        return range(self.offset[seg], self.offset[seg] + self.size_of[seg])


# DOF segments that follow FAMILY, per family (stable order = decode order)
_DOF_SEGS = {
    STUD: ["ANGLE"],
    FIXED: [],
    HINGE: ["FLIP", "ANGLE"],
    AXLE: ["FLIP", "ANGLE", "SLIDE"],
    BALL: ["ANGLE", "ANGLE", "ANGLE"],
}


class GrammarState:
    """Per-position grammar for constrained decoding: tracks where we are in the per-brick
    template and reports which segment(s) may come next, so a sampler can mask logits to only
    grammatically-valid ids -- guaranteeing every generated stream decodes to a valid build.
    """

    def __init__(self, vocab: Vocab):
        self.v = vocab
        self.reset()

    def reset(self):
        self.exp = ["PART"]      # segments still required for the current brick
        self.done = False
        self.n_bricks = 0        # bricks completed so far (brick 0 is the tree root)

    def allowed_segments(self) -> list[str]:
        seg = self.exp[0]
        if seg == "PART":        # brick boundary: start another brick or stop
            return ["PART", "EOS"]
        if seg == "ATTACH":      # exactly one root (brick 0); every later brick attaches via PTR
            return ["ROOT"] if self.n_bricks == 0 else ["PTR"]
        return [seg]

    def allowed_ids(self) -> list[int]:
        ids: list[int] = []
        for seg in self.allowed_segments():
            if seg == "EOS":
                ids.append(self.v.EOS)
            elif seg == "ROOT":
                ids.append(self.v.ROOT)
            else:
                ids.extend(self.v.seg_range(seg))
        return ids

    def step(self, gid: int) -> None:
        seg = self.v.seg_of(gid)
        exp = self.exp[0]
        if exp == "PART":
            if seg == "EOS":
                self.done = True
                return
            self.exp = ["COLOR", "ATTACH"]
        elif exp == "COLOR":
            self.exp = self.exp[1:]
        elif exp == "ATTACH":
            if seg == "ROOT":
                self.exp = ["PART"]
                self.n_bricks += 1
            else:  # PTR
                self.exp = ["PSUB", "CSUB", "PCONN", "CCONN", "FAMILY"]
        elif exp == "FAMILY":
            dof = list(_DOF_SEGS[self.v.local(gid, "FAMILY")])
            self.exp = dof if dof else ["PART"]
            if not dof:
                self.n_bricks += 1
        else:  # PSUB/CSUB/PCONN/CCONN/FLIP/ANGLE/SLIDE
            self.exp = self.exp[1:] or ["PART"]
            if self.exp == ["PART"]:
                self.n_bricks += 1


def _edge_fields(e) -> dict:
    return {f.name: getattr(e, f.name) for f in dc_fields(e)}


def encode_tree(tree, vocab: Vocab) -> list[int]:
    """Build-order Tree -> token ids. Colors are coerced to known codes first."""
    tree = coerce_colors(tree)
    edge_by_child = {e.child: e for e in tree.edges}
    toks = [vocab.BOS]
    for i, p in enumerate(tree.parts):
        toks.append(vocab.gid("PART", p.part_id))
        toks.append(vocab.gid("COLOR", vocab.color_to_idx[p.color]))
        e = edge_by_child.get(i)
        if e is None:  # root of the (single-component) tree
            toks.append(vocab.ROOT)
            continue
        fam = _FAMILY_OF[type(e).__name__]
        toks.append(vocab.gid("PTR", i - e.parent))
        toks.append(vocab.gid("PSUB", int(e.parent_sub)))
        toks.append(vocab.gid("CSUB", int(e.child_sub)))
        toks.append(vocab.gid("PCONN", int(e.parent_conn)))
        toks.append(vocab.gid("CCONN", int(e.child_conn)))
        toks.append(vocab.gid("FAMILY", fam))
        if fam == STUD:
            toks.append(vocab.gid("ANGLE", int(e.yaw) % 360))
        elif fam == FIXED:
            pass
        elif fam == HINGE:
            toks.append(vocab.gid("FLIP", int(bool(e.flip))))
            toks.append(vocab.gid("ANGLE", int(e.yaw) % 360))
        elif fam == AXLE:
            toks.append(vocab.gid("FLIP", int(bool(e.flip))))
            toks.append(vocab.gid("ANGLE", int(e.yaw) % 360))
            slide = max(SLIDE_MIN, min(SLIDE_MAX, int(e.slide)))
            toks.append(vocab.gid("SLIDE", slide - SLIDE_MIN))
        elif fam == BALL:
            toks.append(vocab.gid("ANGLE", int(e.rx) % 360))
            toks.append(vocab.gid("ANGLE", int(e.ry) % 360))
            toks.append(vocab.gid("ANGLE", int(e.rz) % 360))
    toks.append(vocab.EOS)
    return toks


def decode(tokens: list[int], vocab: Vocab):
    """Token ids -> Tree (inverse of encode_tree, up to rare slide clamping)."""
    it = iter(tokens)

    def nxt():
        return next(it)

    def expect_seg(gid, seg):
        if vocab.seg_of(gid) != seg:
            raise ValueError(f"expected {seg}, got {vocab.seg_of(gid)}")
        return vocab.local(gid, seg)

    t0 = nxt()
    if t0 != vocab.BOS:
        raise ValueError("stream must start with BOS")

    parts: list[Part] = []
    edges: list = []
    i = 0

    def parse_brick(first_gid):
        """Parse one brick group; return (Part, edge_or_None). Raises StopIteration if the
        stream is truncated mid-brick (generated streams can be cut off)."""
        part_id = expect_seg(first_gid, "PART")
        color = vocab.idx_to_color[expect_seg(nxt(), "COLOR")]
        part = Part(part_id=part_id, color=color)
        nx = nxt()
        if nx == vocab.ROOT:
            return part, None
        ptr = expect_seg(nx, "PTR")
        # Consume the whole brick body BEFORE validating the pointer, so a bad pointer (which only
        # an untrained sampler emits) drops just this edge without desyncing the token stream.
        psub, csub = expect_seg(nxt(), "PSUB"), expect_seg(nxt(), "CSUB")
        pconn, cconn = expect_seg(nxt(), "PCONN"), expect_seg(nxt(), "CCONN")
        fam = expect_seg(nxt(), "FAMILY")
        if fam == STUD:
            make = lambda **b: StudEdge(**b, yaw=dof[0])
            dof = [expect_seg(nxt(), "ANGLE")]
        elif fam == FIXED:
            make = lambda **b: FixedEdge(**b)
            dof = []
        elif fam == HINGE:
            dof = [bool(expect_seg(nxt(), "FLIP")), expect_seg(nxt(), "ANGLE")]
            make = lambda **b: HingeEdge(**b, flip=dof[0], yaw=dof[1])
        elif fam == AXLE:
            dof = [bool(expect_seg(nxt(), "FLIP")), expect_seg(nxt(), "ANGLE"),
                   expect_seg(nxt(), "SLIDE") + SLIDE_MIN]
            make = lambda **b: AxleEdge(**b, flip=dof[0], yaw=dof[1], slide=dof[2])
        else:  # BALL
            dof = [expect_seg(nxt(), "ANGLE"), expect_seg(nxt(), "ANGLE"), expect_seg(nxt(), "ANGLE")]
            make = lambda **b: BallEdge(**b, rx=dof[0], ry=dof[1], rz=dof[2])
        if not 0 <= i - ptr < i:   # pointer at/past the start -> keep the part, drop the edge
            return part, None
        return part, make(parent=i - ptr, child=i, parent_sub=psub, child_sub=csub,
                          parent_conn=pconn, child_conn=cconn)

    for gid in it:
        if gid == vocab.EOS:
            break
        try:
            part, edge = parse_brick(gid)
        except StopIteration:
            break  # truncated trailing brick -> drop it
        parts.append(part)
        if edge is not None:
            edges.append(edge)
        i += 1

    return Tree(parts=tuple(parts), edges=tuple(edges))
