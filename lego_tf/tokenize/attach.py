"""Parent-relative stud-grid attachment tokenizer (representation v1).

Each brick is expressed *relative to an already-placed brick* -- "my port c mates parent brick
p's port q" -- rather than by absolute pose (v0). Translation-invariant, off-grid placements
unrepresentable, coherent sequences (plan sec 3; cf. BrickAnything's attachment tree).

**Port model.** From measured geometry (studs and underside tubes don't coincide; see memory
lego-connection-geometry) each brick exposes two kinds of *port*:
  * TOP    -- a stud (local position of the stud connector);
  * BOTTOM -- the underside cell directly below each stud, i.e. (stud.x, bottom_plane, stud.z).
A stud plugging into an underside means a TOP port of one brick **coincides in world space**
with a BOTTOM port of the other. So connectivity = world coincidence of opposite-kind ports,
and a single coinciding port pair + the child's orientation pins the child's pose exactly ->
reversible, and works whether the child sits above OR below its parent.

**Ordering.** We build the port-coincidence graph over all bricks, then BFS from the lowest
brick of each connected component so every non-root brick has an already-placed parent. Bricks
with no stud port (tiles, Technic pin/axle parts, SNOT until later) are their own roots and fall
back to an absolute anchor, so every canonical-rotation model stays losslessly round-trippable.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from lego_tf.data.ldraw import Brick
from lego_tf.data.parts import ConnectorExtractor, ConnType
from lego_tf.tokenize.absolute import PartVocab
from lego_tf.tokenize.rotations import RotationCodebook

TOL = 1e-2
_QUANT = 1.0  # LDU rounding for world-coincidence hashing (on-grid domain)

# A port has a mating *group* and a *polarity* (+1 male / -1 female). Two ports connect when
# they share a group, have opposite polarity, and coincide in world space. Groups:
#   STUDGRID -- stud (male) into an underside footprint cell (female)
#   PIN      -- Technic pin (male) into pin hole (female)
#   AXLE     -- Technic axle (male) into axle hole (female)
#   BALL     -- ball (male) into socket (female)
#   CLIPBAR  -- bar/handle (male) into clip (female)
STUDGRID, PIN, AXLE, BALL, CLIPBAR = "studgrid", "pin", "axle", "ball", "clipbar"

# maps ConnType -> (group, polarity) for the coincidence-matched connector families
_CONN_GROUP = {
    ConnType.PIN: (PIN, +1), ConnType.PIN_HOLE: (PIN, -1),
    ConnType.AXLE: (AXLE, +1), ConnType.AXLE_HOLE: (AXLE, -1),
    ConnType.BALL: (BALL, +1), ConnType.SOCKET: (BALL, -1),
    ConnType.BAR: (CLIPBAR, +1), ConnType.CLIP: (CLIPBAR, -1),
}


@dataclass
class Port:
    group: str
    pol: int          # +1 male, -1 female
    pos: np.ndarray   # local position (encode) or world position (after _world_ports)


@dataclass
class AttachToken:
    part_id: int
    color: int
    rot_id: int
    parent: int              # index of parent in the emitted sequence, or -1 (absolute root)
    parent_port: int         # port index on the parent   (unused when parent == -1)
    child_port: int          # port index on this brick    (unused when parent == -1)
    pos: tuple = (0, 0, 0)   # absolute translation, only when parent == -1


@dataclass
class AttachSequence:
    tokens: list[AttachToken] = field(default_factory=list)

    def __len__(self):
        return len(self.tokens)

    def parent_fraction(self) -> float:
        if not self.tokens:
            return 0.0
        return sum(1 for t in self.tokens if t.parent >= 0) / len(self.tokens)

    def num_roots(self) -> int:
        return sum(1 for t in self.tokens if t.parent < 0)


class AttachTokenizer:
    def __init__(self, part_vocab: PartVocab, extractor: ConnectorExtractor,
                 rotations: RotationCodebook | None = None):
        self.parts = part_vocab
        self.ext = extractor
        self.rotations = rotations or RotationCodebook()

    # -- ports ----------------------------------------------------------------
    def _ports(self, part: str) -> list[Port]:
        """Deterministic ports for a part (local coords). Order is fixed so indices are stable."""
        conns = self.ext.connectors(part)
        ports: list[Port] = []
        # stud grid, male side: each top stud
        studs = [np.asarray(c.pos, float) for c in conns if c.type == ConnType.STUD]
        for s in studs:
            ports.append(Port(STUDGRID, +1, s))
        # stud grid, female side: every footprint cell on the underside plane -- so tiles and
        # smooth-top parts (no studs) still expose receptacles, and jumpers cover >1 cell.
        bottom = self.ext.y_extent(part)[1]
        for (cx, cz) in self.ext.footprint_cells(part):
            ports.append(Port(STUDGRID, -1, np.array([cx, bottom, cz])))
        # Technic pins/axles, ball joints, clips/bars: the connector position is the mating
        # point directly (coincidence-matched by group + polarity).
        for c in conns:
            gp = _CONN_GROUP.get(c.type)
            if gp is not None:
                group, pol = gp
                ports.append(Port(group, pol, np.asarray(c.pos, float)))
        return ports

    def _world_ports(self, brick: Brick) -> list[Port]:
        return [Port(p.group, p.pol, brick.rot @ p.pos + brick.pos) for p in self._ports(brick.part)]

    # -- encode ---------------------------------------------------------------
    def encode(self, bricks: list[Brick]) -> AttachSequence:
        n = len(bricks)
        rot_ids = [self.rotations.id_of(b.rot) for b in bricks]
        part_ids = [self.parts.id_of(b.part) for b in bricks]
        wports = [self._world_ports(b) for b in bricks]

        # spatial hash: rounded world pos -> list of (brick, port_idx, group, polarity)
        buckets: dict[tuple, list] = defaultdict(list)
        for bi, ports in enumerate(wports):
            for pi, p in enumerate(ports):
                buckets[_key(p.pos)].append((bi, pi, p.group, p.pol))

        # adjacency: edge if two coincident ports share a group but have opposite polarity.
        adj: dict[int, list[tuple[int, int, int]]] = defaultdict(list)  # a -> (b, port_a, port_b)
        seen = set()
        for cell in buckets.values():
            for a_bi, a_pi, a_g, a_pol in cell:
                for b_bi, b_pi, b_g, b_pol in cell:
                    if a_bi == b_bi or a_g != b_g or a_pol == b_pol:
                        continue
                    if (a_bi, a_pi, b_bi, b_pi) in seen:
                        continue
                    seen.add((a_bi, a_pi, b_bi, b_pi))
                    adj[a_bi].append((b_bi, a_pi, b_pi))

        # BFS ordering: roots = lowest brick (max world y, Y points down) per component.
        order, parent_of = self._bfs_order(n, adj, bricks)

        seq = AttachSequence()
        emitted_index = {}  # brick index -> position in emitted sequence
        for bi in order:
            emitted_index[bi] = len(seq.tokens)
            par = parent_of.get(bi)
            if par is None:
                tok = AttachToken(part_ids[bi], int(bricks[bi].color), rot_ids[bi],
                                  parent=-1, parent_port=0, child_port=0,
                                  pos=tuple(int(round(v)) for v in bricks[bi].pos))
            else:
                p_bi, child_port, parent_port = par
                tok = AttachToken(part_ids[bi], int(bricks[bi].color), rot_ids[bi],
                                  parent=emitted_index[p_bi],
                                  parent_port=parent_port, child_port=child_port)
            seq.tokens.append(tok)
        return seq

    def _bfs_order(self, n, adj, bricks):
        """Return (order, parent_of) where parent_of[b] = (parent_brick, child_port, parent_port)."""
        visited = [False] * n
        order: list[int] = []
        parent_of: dict[int, tuple[int, int, int]] = {}
        # seed roots by lowest brick first (largest y), stable by index
        roots = sorted(range(n), key=lambda i: (-bricks[i].pos[1], i))
        for r in roots:
            if visited[r]:
                continue
            visited[r] = True
            order.append(r)
            q = deque([r])
            while q:
                cur = q.popleft()
                for (nb, cur_port, nb_port) in adj.get(cur, []):
                    if not visited[nb]:
                        visited[nb] = True
                        # cur is parent (already placed); nb is child.
                        parent_of[nb] = (cur, nb_port, cur_port)
                        order.append(nb)
                        q.append(nb)
        return order, parent_of

    # -- decode ---------------------------------------------------------------
    def decode(self, seq: AttachSequence) -> list[Brick]:
        bricks: list[Brick] = []
        wports: list[list[tuple[int, np.ndarray]]] = []
        for tok in seq.tokens:
            part = self.parts.part_of(tok.part_id)
            R = self.rotations.matrix_of(tok.rot_id)
            if tok.parent < 0:
                pos = np.array(tok.pos, dtype=float)
            else:
                parent_world_port = wports[tok.parent][tok.parent_port].pos
                child_local_port = self._ports(part)[tok.child_port].pos
                pos = parent_world_port - R @ child_local_port
            m = np.eye(4)
            m[:3, :3] = R
            m[:3, 3] = pos
            b = Brick(part=part, color=tok.color, matrix=m)
            bricks.append(b)
            wports.append(self._world_ports(b))
        return bricks


def _key(wp: np.ndarray) -> tuple:
    return (round(float(wp[0]) / _QUANT), round(float(wp[1]) / _QUANT), round(float(wp[2]) / _QUANT))
