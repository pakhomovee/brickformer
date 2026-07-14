"""Native LEGO tokenizer: bricknet build-order Tree <-> integer token stream.

A bricknet graph is pose-free, so a brick along the build-order tree is fully specified by its
part + color and (for non-root bricks) the attachment edge to an already-placed brick: which
sub-parts and connectors mate, the connector family, and the family's DOF. No coordinates.

Per-brick token group (segmented vocab -> a constrained-decoding grammar can mask each position
to one field):

    root brick :  PART COLOR ROOT
    non-root   :  PART COLOR PTR PCONN CCONN <dof...>
      stud  dof: ANGLE(yaw)
      fixed dof: (none)
      hinge dof: FLIP ANGLE(yaw)
      axle  dof: FLIP ANGLE(yaw) SLIDE
      ball  dof: ANGLE(rx) ANGLE(ry) ANGLE(rz)

PCONN/CCONN are compact **flat connector indices** into each part's connector list (see
`bnet/connectors.py`): a small, per-part learnable target instead of the old wide sub/conn integers.
The edge family is *derived* from the two connectors' kinds (drops the FAMILY token). The grammar
masks PTR/PCONN/CCONN to real, mutually-compatible connectors -> generation is connector-valid by
construction. Angles are integer degrees (360 exact bins); the stream is losslessly reversible:
tokens -> Tree -> bricknet graph -> scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields

from bricknet.core import Tree, Part, StudEdge, AxleEdge, HingeEdge, BallEdge, FixedEdge

from lego_tf.bnet.trees import catalog, coerce_colors
from lego_tf.bnet import connectors as K

# families (stable order = token id)
STUD, AXLE, HINGE, BALL, FIXED = range(5)
_FAMILY_OF = {"StudEdge": STUD, "AxleEdge": AXLE, "HingeEdge": HINGE, "BallEdge": BALL, "FixedEdge": FIXED}
_EDGE_CLS = {STUD: StudEdge, AXLE: AxleEdge, HINGE: HingeEdge, BALL: BallEdge, FIXED: FixedEdge}
_FAM_STR_TO_INT = {"stud": STUD, "axle": AXLE, "hinge": HINGE, "ball": BALL, "fixed": FIXED}

SLIDE_MIN, SLIDE_MAX = -512, 511  # inclusive; token index = slide - SLIDE_MIN (observed |slide|<=322)

# segment name -> size (order here defines vocab layout). Specials first. PCONN/CCONN are flat
# connector indices into a part's connector list; the global max is 2500 (a large baseplate), so
# CONN spans 2560 with headroom. prepare_data skips (and counts) any structure that exceeds a cap.
CONN_MAX = 2560
_SEGMENTS = [
    ("PAD", 1), ("BOS", 1), ("EOS", 1), ("ROOT", 1),
    ("PART", 14583),
    ("COLOR", 219),
    ("PTR", 256),
    ("PCONN", CONN_MAX),
    ("CCONN", CONN_MAX),
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
    """Per-position grammar for constrained decoding. Tracks the per-brick template AND the parts
    placed so far, so it can mask each position to grammatically- AND geometrically-valid ids:
    PTR to an attachable earlier brick, PCONN to a real parent connector, CCONN to a connector on
    the child that mates it. Every generated stream therefore decodes to a connector-valid build.
    (Masks fall back to the raw field range if no compatible option exists, so generation is never
    stuck; that fallback is rare and matches decode's FIXED-family default.)
    """

    def __init__(self, vocab: Vocab):
        self.v = vocab
        self.reset()

    def reset(self):
        self.exp = ["PART"]        # segments still required for the current brick
        self.done = False
        self.n_bricks = 0          # bricks completed so far (brick 0 is the tree root)
        self.part_ids: list[int] = []   # part_id of each brick, in order
        self._child = None         # current brick's part_id
        self._ppid = None          # current edge's parent part_id
        self._kp = None            # chosen parent flat connector index

    # -- masked id sets --------------------------------------------------------------------------
    def _ptr_ids(self) -> list[int]:
        child_idx = len(self.part_ids) - 1
        ok = [child_idx - p for p in range(child_idx)
              if K.valid_parent_conns(self.part_ids[p], self._child)]
        rng = ok or list(range(1, child_idx + 1))          # fallback: any earlier brick
        return [self.v.gid("PTR", d) for d in rng]

    def _pconn_ids(self) -> list[int]:
        ks = K.valid_parent_conns(self._ppid, self._child) or range(K.n_conn(self._ppid))
        return [self.v.gid("PCONN", k) for k in ks]

    def _cconn_ids(self) -> list[int]:
        ks = K.compatible_child_conns(self._ppid, self._kp, self._child) or range(K.n_conn(self._child))
        return [self.v.gid("CCONN", k) for k in ks]

    def allowed_ids(self) -> list[int]:
        exp = self.exp[0]
        if exp == "PART":          # brick boundary: start another brick or stop
            return list(self.v.seg_range("PART")) + [self.v.EOS]
        if exp == "ATTACH":        # exactly one root (brick 0); every later brick attaches via PTR
            return [self.v.ROOT] if self.n_bricks == 0 else self._ptr_ids()
        if exp == "PCONN":
            return self._pconn_ids()
        if exp == "CCONN":
            return self._cconn_ids()
        return list(self.v.seg_range(exp))                 # COLOR / FLIP / ANGLE / SLIDE

    def step(self, gid: int) -> None:
        exp = self.exp[0]
        if exp == "PART":
            if gid == self.v.EOS:
                self.done = True
                return
            self._child = self.v.local(gid, "PART")
            self.part_ids.append(self._child)
            self.exp = ["COLOR", "ATTACH"]
        elif exp == "COLOR":
            self.exp = ["ATTACH"]
        elif exp == "ATTACH":
            if gid == self.v.ROOT:
                self.exp = ["PART"]
                self.n_bricks += 1
            else:  # PTR
                parent_idx = (len(self.part_ids) - 1) - self.v.local(gid, "PTR")
                self._ppid = self.part_ids[parent_idx] if 0 <= parent_idx < len(self.part_ids) - 1 \
                    else self.part_ids[0]
                self.exp = ["PCONN", "CCONN"]
        elif exp == "PCONN":
            self._kp = self.v.local(gid, "PCONN")
            self.exp = ["CCONN"]
        elif exp == "CCONN":
            kc = self.v.local(gid, "CCONN")
            try:
                fam = _FAM_STR_TO_INT[K.family_from_flat(self._ppid, self._kp, self._child, kc)]
            except Exception:
                fam = FIXED
            dof = _DOF_SEGS[fam]
            self.exp = list(dof) if dof else ["PART"]
            if not dof:
                self.n_bricks += 1
        else:  # FLIP / ANGLE / SLIDE
            self.exp = self.exp[1:] or ["PART"]
            if self.exp == ["PART"]:
                self.n_bricks += 1


def _edge_fields(e) -> dict:
    return {f.name: getattr(e, f.name) for f in dc_fields(e)}


def encode_tree(tree, vocab: Vocab) -> list[int]:
    """Build-order Tree -> token ids. Colors are coerced to known codes first."""
    tree = coerce_colors(tree)
    edge_by_child = {e.child: e for e in tree.edges}
    pids = [p.part_id for p in tree.parts]
    toks = [vocab.BOS]
    for i, p in enumerate(tree.parts):
        toks.append(vocab.gid("PART", p.part_id))
        toks.append(vocab.gid("COLOR", vocab.color_to_idx[p.color]))
        e = edge_by_child.get(i)
        if e is None:  # root of the (single-component) tree
            toks.append(vocab.ROOT)
            continue
        fam = _FAMILY_OF[type(e).__name__]
        kp, kc = K.flat_indices(e, e.parent, i, pids[e.parent], pids[i])
        toks.append(vocab.gid("PTR", i - e.parent))
        toks.append(vocab.gid("PCONN", kp))
        toks.append(vocab.gid("CCONN", kc))
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
        """Parse one brick group; return (Part, edge_or_None, ok). `ok` is False when this brick
        cannot form a valid aligned edge (a second root, a bad pointer, or an incompatible connector
        pair) -- the caller then truncates the build here. Truncating (instead of dropping just the
        edge) is required because bricknet's tree_to_graph assumes edge i connects part i+1, so a
        single missing edge would misalign every later edge. Raises StopIteration if the stream is
        truncated mid-brick (generated streams can be cut off)."""
        part_id = expect_seg(first_gid, "PART")
        color = vocab.idx_to_color[expect_seg(nxt(), "COLOR")]
        part = Part(part_id=part_id, color=color)
        nx = nxt()
        if nx == vocab.ROOT:
            return part, None, (i == 0)   # only brick 0 may be a root; a later root truncates
        ptr = expect_seg(nx, "PTR")
        kp = expect_seg(nxt(), "PCONN")
        kc = expect_seg(nxt(), "CCONN")
        parent = i - ptr
        # Family + sub/conn fields are DERIVED from the flat connector indices. Resolve them first
        # (family drives how many DOF tokens follow); fall back to FIXED (no DOF) if the connectors
        # don't form a family -- keeps the stream in sync for malformed/truncated input.
        parent_pid = parts[parent].part_id if 0 <= parent < i else (parts[0].part_id if parts else part_id)
        try:
            fam_str, psub, csub, pconn, cconn = K.edge_fields_from_flat(parent_pid, kp, part_id, kc)
            fam = _FAM_STR_TO_INT[fam_str]
            fields_ok = True
        except Exception:
            fam, fields_ok = FIXED, False
        if fam == STUD:
            dof = [expect_seg(nxt(), "ANGLE")]
        elif fam == FIXED:
            dof = []
        elif fam == HINGE:
            dof = [bool(expect_seg(nxt(), "FLIP")), expect_seg(nxt(), "ANGLE")]
        elif fam == AXLE:
            dof = [bool(expect_seg(nxt(), "FLIP")), expect_seg(nxt(), "ANGLE"),
                   expect_seg(nxt(), "SLIDE") + SLIDE_MIN]
        else:  # BALL
            dof = [expect_seg(nxt(), "ANGLE"), expect_seg(nxt(), "ANGLE"), expect_seg(nxt(), "ANGLE")]
        if not (0 <= parent < i) or not fields_ok:   # bad pointer/connectors -> truncate the build here
            return part, None, False
        base = dict(parent=parent, child=i, parent_sub=psub, child_sub=csub,
                    parent_conn=pconn, child_conn=cconn)
        if fam == STUD:
            return part, StudEdge(**base, yaw=dof[0]), True
        if fam == HINGE:
            return part, HingeEdge(**base, flip=dof[0], yaw=dof[1]), True
        if fam == AXLE:
            return part, AxleEdge(**base, flip=dof[0], yaw=dof[1], slide=dof[2]), True
        if fam == BALL:
            return part, BallEdge(**base, rx=dof[0], ry=dof[1], rz=dof[2]), True
        return part, FixedEdge(**base), True

    for gid in it:
        if gid == vocab.EOS:
            break
        try:
            part, edge, ok = parse_brick(gid)
        except StopIteration:
            break  # truncated trailing brick -> drop it
        if not ok:
            break  # brick can't form an aligned edge -> truncate here (keeps edge<->part alignment)
        parts.append(part)
        if edge is not None:
            edges.append(edge)
        i += 1

    return Tree(parts=tuple(parts), edges=tuple(edges))
