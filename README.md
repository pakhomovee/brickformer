# LEGO Assembly Transformer

Implementation of the plan in [`lego transformer plan.md`](./lego%20transformer%20plan.md): a
LEGO-native autoregressive transformer for physically-valid, finished vehicle assemblies with
interactive prefix-completion.

This repo is the **Stage-0 pipeline** (CPU-only dev work). Training/inference (torch, DINOv2,
Blender rendering) run on a rented GPU box and are deliberately not set up here.

## Direction (2026-07): build on `bricknet`

BrickNet ships its own Python package (`pip install bricknet`) that provides the **entire
representation layer** — a connector-annotated catalog of 14,583 parts, a typed connector graph
with articulation DOF per edge (`StudEdge` yaw · `HingeEdge` flip/yaw · `AxleEdge` flip/yaw/slide
· `BallEdge` rx/ry/rz · `FixedEdge`), spanning-tree serialization + a collision-free sampler
(`sample_tree`, `serialize_tree`), LDR round-trip (`parse_ldr`, `graph_to_ldr`), and a collision
scorer (`score_text`). This **supersedes** the from-scratch tokenizer stack below: it is annotated
(not geometry-inferred) and its hinge/ball/axle edges already cover the ~24% articulated tail we
had deferred. **Go-forward work builds on `bricknet`.** The `lego_tf/tokenize/` + connector code
is **kept as reference / cross-check only** — not the critical path — and is not being extended.

Go-forward code lives in **`lego_tf/bnet/`**. The model trains directly on a **native LEGO token
stream** (not text): a bricknet graph is pose-free, so the build-order tree + typed connectors +
their DOF fully determine geometry via the catalog — no coordinate tokens. `trees.py` provides the
structural layer (collision-free build-order sampling, brick-boundary truncation for the
interactive-completion signal, unknown-colour coercion); `tokenizer.py` turns a tree into a
**segmented integer token stream** and back (`Tree ↔ tokens`, reversible → `score_text`). Per
brick: `PART COLOR` then, for non-root bricks, `PTR PSUB CSUB PCONN CCONN FAMILY <dof>` where DOF
is family-specific (stud yaw · hinge flip+yaw · axle flip+yaw+slide · ball rx/ry/rz). Angles are
integer degrees → 360 one-degree bins are exact, so the stream is discrete **and** losslessly
reversible. **Verified on all 512 val graphs: structural round-trip exact, score-exact; vocab
25,157; ~9 tokens/brick.** We train on **all** BrickNet data (no vehicle subset).

`model.py` is a decoder-only transformer (LLaMA-style: RoPE, RMSNorm, SwiGLU) with
grammar-constrained generation; `dataset.py` tokenizes a split into training sequences;
`train_overfit.py` is the **Stage-0 gate** — overfit a tiny model on a few structures to ~0 loss,
then generate a constrained sample and confirm it decodes to a valid build. Verified on CPU:

```bash
python -m lego_tf.bnet.train_overfit --n 16 --steps 800 --d-model 128 --batch 4 --max-len 160
# loss 9.76 -> 0.08; constrained generation -> decoded build, collisions=0
```

Design locked: discrete-bin DOF · native tokens (not text) · v0 pointer-only first (v1 resolved
poses is the first ablation) · CPU-smoke-test-then-GPU. Next: tokenize the full corpus + rent a
GPU for the 25M ablations.

## Status (legacy — reference-only tokenizer stack)

| Component | State |
|---|---|
| LDraw `.ldr`/`.mpd` parser + flatten (transform + colour composition) | ✅ |
| Absolute-pose tokenizer (v0, lossless) | ✅ |
| 48 canonical rotation codebook | ✅ |
| Bit-exact on-grid round-trip (Stage-0 gate) | ✅ |
| Stud connector extraction from LDraw parts library | ✅ |
| Parent-relative stud-grid attachment tokenizer (v1, plan §3) | ✅ lossless round-trip |
| Bidirectional ports + BFS build-order | ✅ |
| Typed connectors: stud + Technic pin/axle/hole | ✅ |
| Flat token stream + shared vocabulary (grammar-conditional) | ✅ 6.5 tok/brick |
| Footprint female ports (tiles attach) + ball/clip/bar families | ✅ |
| Fine-geom channel (hinges/angled ~24%) + hinge connectors | ⛔ dropped — `bricknet` edges cover this |
| Occupancy features, model, training | ⬜ GPU stage (build on `bricknet`) |

46 passing tests. Connector families: stud/antistud, Technic pin & axle (+holes), ball/socket,
clip/bar. Attach fraction: x-wing 56%, tractor 57%, **ATV 77%**, Technic loader 49% (round-trip
EXACT throughout). Remaining roots are mostly (a) parts whose only neighbour is a non-canonical
(hinged/angled) brick excluded until the fine-geom channel lands, and (b) Technic pins with a
systematic 10-LDU `connect.dat` origin offset. Hinges use generic knuckle geometry (no named
connector primitive) and a rotational DOF, so they wait for the fine-geom channel. Flat stream (`python -m lego_tf.stream_stats`): **6.5 tokens/brick** (plan
target 6–8), so a 1000-brick model ≈ 6.5k tokens (fits 8k context). Segmented shared vocab
(BOS/EOS · PART · COLOR · ROT · PTR · PORT · COORD) with a grammar that masks each position to
one valid segment — the hook for constrained decoding. v1 attachment on real vehicles (canonical-rotation subset,
`python -m lego_tf.attach_coverage`): round-trip EXACT everywhere; attach fraction 42% (x-wing),
52% (tractor), **74% (ATV)**, **48% (Technic loader)**. Connectivity uses a group/polarity port
model — a male port (stud / pin / axle) mates a coincident female port (underside cell / pin hole
/ axle hole). Remaining roots need hinge/clip/ball families + finer SNOT handling.

## Layout

```
lego_tf/
  data/ldraw.py         # parse / flatten / write LDR, Brick dataclass
  tokenize/rotations.py # 48-orientation codebook
  tokenize/absolute.py  # v0 tokenizer + PartVocab
  inspect_model.py      # CLI: on-grid coverage + round-trip report for real MPDs
  tests/                # pytest (synthetic + real-sample round-trip)
data/samples/           # real OMR vehicle MPDs (x-wing, tractor, ATV, Technic loader)
```

## Train on a GPU (two commands)

We tokenize natively from the graph `.npz` splits, so only the compact graphs are downloaded —
never the 11 GB `paths_*.jsonl` text.

```bash
# once: point the AWS CLI at Yandex Object Storage
aws configure --profile brickformer                                   # key id + secret
aws configure set profile.brickformer.region ru-central1
aws configure set profile.brickformer.endpoint_url https://storage.yandexcloud.net

bash scripts/prepare.sh      # install deps, download pt.npz/val.npz, tokenize -> data/*.bin
bash scripts/train.sh        # train the 25M twin on GPU (auto-detects CUDA)
# bigger run:  bash scripts/train.sh --size 250M --batch 16 --grad-accum 4 --max-iters 60000
```

**Rough time** (default 25M, ~650M tokens): **~1–1.5 h on one RTX 4090 / A10G**, ~30–40 min on one
A100. The 250M main run (`--size 250M`, ~1 B tokens): ~8–12 h on one 4090, ~3–5 h on one A100,
~1–2 days on 8×A100 (LegoACE scale). Estimates are ±2×, dominated by GPU MFU.

## Setup (CPU dev box)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # on a CPU box: pip install torch --index-url .../whl/cpu
python -m pytest lego_tf/ -q
python -m lego_tf.bnet.train_overfit --n 16 --steps 800   # Stage-0 gate
```

## Key finding (real vehicle data)

Across 4 real OMR vehicles (842 bricks): the lossless on-grid round-trip is **exact**, but
only **~76%** of bricks are fully on-grid (integer LDU position + one of the 48 axis-aligned
orientations). The Technic loader is ~71%. The off-grid ~24% are hinges / angled panels /
articulated parts — precisely the tail that pure absolute-pose grids drop and that the plan's
typed-connector + fine-geom channel (representation v1) is designed to capture. This quantifies
the representation gap before any training spend.
