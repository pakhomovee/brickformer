# Specialized LEGO Assembly Transformer — Full Technical Plan (v2)

**Goal.** Train a LEGO-native autoregressive transformer that (a) supports a large part vocabulary with plate-resolution and sideways (SNOT) building, (b) produces physically valid, *finished* structures under text / image / geometry conditioning, and (c) supports **prefix-completion for an interactive builder copilot** — condition on a human's partial build and suggest the next bricks. Target scale: structures up to ~1,500 parts, vehicle domain (planes, ships, tanks, cars).

**What changed in v2.** After a deeper literature review, the field turns out to be more crowded and faster-moving than v1 assumed. Four concurrent 2025–2026 systems (LegoACE, BrickNet, BrickAnything, PointCloud2Brick/LEGO-Maker) have independently converged on the autoregressive-transformer + native-tokenizer + constrained-decoding + preference-tuning recipe. This is good news for feasibility (the core is de-risked and partly open-sourced) and bad news for naive novelty (a plain text→brick generator is no longer publishable or differentiated). v2 therefore (1) picks a concrete representation stance informed by the papers' head-to-head tradeoffs, (2) adds DPO and structure-aware tokenization as first-class components, and (3) **re-centers the contribution on interactive completion + hierarchy + finish-calibration + full-vocabulary vehicles**, which no released system does well.

---

## 1. Prior-art landscape (the map we are building on)

| System | Venue | Data | Representation | Conditioning | Released | Key result / lesson |
|---|---|---|---|---|---|---|
| **BrickGPT / LegoGPT** | ICCV 2025 | StableText2Brick, 47k structures, **8 brick types**, studs-up only | Absolute (x,y,z) + footprint, LLaMA-3.2-1B, text serialization | Text | Yes (code+data+weights) | Physics-aware **rollback** lifts stability 24%→98.8%. Tiny vocabulary is the ceiling. |
| **LegoACE** | SIGGRAPH Asia 2025 | **LegoVerse, 55k models, 9,314 brick types, 48 rotations** | **Absolute pose tokens** (5 tok/brick), no connectivity annotation, LLaMA backbone, 243M | Text (CLIP) + **multi-view normal maps** (DINOv2) | Code+models page (xh38.github.io/LegoACE) | Beats BrickGPT 87%–13% user preference. **Subsequence sampling + DPO** both improve fidelity. Connectivity only *implicit* → 82% connected rate, some floating parts. |
| **BrickNet** | CVPR 2026 | **320k structures, 9.7k parts, 40M bricks**, connector-annotated LDraw library | **Typed connector graph** (stud/hinge/axle/ball/fixed), spanning-tree serialization, Qwen-3 0.6–14B | Text | "By end of May" (check repo) | Graph beats direct-pose on parse validity/survival. **~94-step parse survival but ~20-step collision-free horizon.** 14B ≈ 1.7B on quality → capacity isn't the bottleneck. |
| **BrickAnything** | arXiv May 2026 | (geometry-conditioned; challenging + stable subsets) | **Structure-aware BFS tree tokenization** (parent-relative attachment), point-cloud conditioned | **Point cloud** (unified geom interface) | Unclear | 83.4% stable, **100% valid**, IoU 0.586, rollbacks **6.75→0.42**. DPO + validity-constrained decoding + adaptive rollback. |
| **PointCloud2Brick / "Rollback-Free"** | arXiv May 2026 | — | Absolute footprint+anchor tokens | Point cloud | Unclear | First **rollback-free** stable generation — bakes stability into the model so inference needs no rejection loop. |
| **LEGO-Maker** | 2026 | limited categories | AR, multiple brick types | Image | Unclear | Image-conditioned; narrow category coverage. |
| **Budget-Aware Sequential Brick Assembly** | arXiv 2022 | RAD / MNIST-C / ModelNet-C (2×4 bricks) | voxel-occupancy, Bayesian | partial structure | — | **Does the completion task**: give partial structure, complete it, under a brick budget. Small vocabulary, but the closest prior to the copilot idea. |

**Reading of the map.** Two representation camps: *absolute-pose* (LegoACE, BrickGPT, PointCloud2Brick — cheap, annotation-free, but connectivity is emergent and off-grid/hinged parts are awkward) vs *connectivity-structured* (BrickNet's connector graph, BrickAnything's attachment tree — exact legality and better sequence coherence, at the cost of annotation or graph construction). The two critiques don't fully collide: BrickNet's "direct pose loses precision" targets *chained relative* transforms and *continuous* regression, while LegoACE's binned *absolute* poses sidestep accumulation — but binned absolute poses genuinely cannot represent hinge angles and off-grid SNOT cleanly, which is exactly the vehicle-domain need. The universal agreements across all recent work: **native per-brick tokenization** (~5 tok/brick), **constrained decoding** (mask to valid token types), **preference tuning (DPO)** using a geometry-fidelity reward, and **subsequence/partial-structure augmentation**.

---

## 2. Dataset decision

**Primary: BrickNet (320k / 9.7k parts / 40M bricks) + its connector-annotated LDraw library.** This is the most valuable single asset in the field: the largest corpus, the widest part vocabulary paired with *actual connector annotations* (studs/holes/axles/hinges/balls with precise positions), and it preserves LDraw subfile structure for hierarchy supervision. The connector annotation is the expensive thing we most want and least want to redo.

**Secondary / fusible: LegoVerse (55k / 9,314 parts / 48 rotations).** Independent, vehicle-inclusive (explicitly spans vehicles, spaceships, etc.), and comes with a working native tokenizer + DPO recipe we can start from. Because its representation is absolute-pose, it complements rather than duplicates BrickNet. Use it as (a) the fastest path to a working Stage-0 pipeline (their code likely runs today), and (b) a fallback if BrickNet access stalls.

**Tertiary / bootstrap: StableText2Brick (47k, 8 types).** Only useful for reproducing BrickGPT's rollback baseline and sanity-checking the stability checker. Too small a vocabulary for the real target.

**Raw fallback: LDraw OMR + Rebrickable/Eurobricks MOCs.** If both curated datasets are inaccessible, rebuild from raw LDraw and bootstrap connector annotation from the LDCad snap system. Costs +4–6 weeks; resolve access in week 1 precisely to avoid discovering this late.

**Decision: build on BrickNet's representation and data as the spine, borrow LegoACE's tokenizer/DPO engineering and vehicle coverage, adopt BrickAnything's attachment-tree idea for sequence coherence.** Curate a **vehicles subset** across whichever corpora we obtain; it is the eval domain and gets upsampled during SFT.

---

## 3. Representation decision (the central technical bet)

**Hybrid connectivity-first tokenization.** Per placed brick, target 6–8 tokens:

```
[PART] [COLOR] [ANCHOR_ptr] [CONN_ptr] [ROT_class] [FINE_geom?] [BUDGET_bucket]
```

- **Attachment is parent-relative (BrickAnything-style), not absolute (LegoACE-style).** `ANCHOR_ptr`/`CONN_ptr` are pointers into the set of open connectors on already-placed bricks. This makes "attach to a nonexistent/occupied connector" *unrepresentable*, gives translation invariance for free, and yields the compact, coherence-preserving sequences BrickAnything showed cut rollbacks by ~16×. The BFS spanning-tree order + end-of-parent (EOP) tokens come straight from BrickAnything.
- **Connector types are typed (BrickNet-style):** stud, anti-stud/tube, axle, pin, clip/bar, hinge, ball, fixed. This is what buys hinges and off-grid SNOT that absolute-pose grids can't express — the vehicle-domain requirement.
- **`ROT_class`:** the 48 canonical axis-aligned orientations (24 + mirrors), following LegoACE — covers the overwhelming majority including all standard SNOT.
- **`FINE_geom`:** emitted *only* for connector types with continuous DOFs — hinge/ball angle (5° bins + optional 1° residual), axle slide (1 LDU). Grammar-conditional, so absent for rigid studs. This is the clean way to handle articulation that pure-grid methods drop.
- **`BUDGET_bucket`:** bucketed remaining-part count (log-spaced), the finish-calibration mechanism (Sec. 6).

**Why not pure absolute pose (LegoACE):** simpler and annotation-free, but hinges/articulation are lost and connectivity is only emergent (their own 82% connected rate + floating-part limitation). For static micro-buildings that's fine; for articulated vehicles it isn't.

**Why not pure BrickNet graph:** excellent legality, but their ~20-step collision horizon shows connectivity annotation alone doesn't give *spatial* common sense — the model still needs occupancy awareness (Sec. 5.3). We keep their typed connectors, add BrickAnything's compact attachment encoding, and add geometry-in-state.

**Resolution.** Geometry stays continuous (SE(3) from connector transforms); nothing about *legality* is grid-based, so plate offsets, half-stud jumpers, and SNOT are exact. Grids appear **only** as a perceptual occupancy *input* (Sec. 5.3) at 4 LDU default, with an optional 2 LDU fine window (2 LDU is the true GCD of plate=8, half-stud=10, stud=20, brick=24).

---

## 4. Model

- **Backbone:** decoder-only transformer (LLaMA-style: RoPE, SwiGLU, RMSNorm), **~150–250M** for the main run (LegoACE's 243M is the proven reference point), plus a **~25M twin** for all architecture ablations. bf16, FlashAttention, context 8,192 tokens (~1,000+ bricks at 6–8 tok/brick); RoPE-scale to 16k late if the 1,500-part target needs it.
- **Vocabulary:** `V = V_part ∪ V_color ∪ V_connector-type ∪ V_rot ∪ V_fine ∪ V_pointer ∪ V_budget ∪ {BOS,EOS,EOP,SUBMODEL_*}`. Part vocab = top-N covering ≥99% of vehicle-domain instances (start N≈4–6k; rare tail → `PART_UNK`, excluded from generation).
- **Dynamic vocabulary masking at decode** (all four papers do this): at each position mask logits to only the grammatically-valid token class. Zero-cost elimination of all parse failures.

### Geometry mechanisms (ablation ladder — this is the core research question)
- **v0 — pointer attachment only.** Parent-relative tokens already encode local structure. Baseline.
- **v1 — geometric input embeddings.** Add each placed brick's *resolved* world pose (Fourier-featured position, rotation-class embedding + continuous residual, bbox extents) to its token group. Cheap, fully parallel in training, makes proximity/overlap directly attention-visible. Likely most of the win.
- **v2 — occupancy cross-attention.** A local occupancy crop (48³ @ 4 LDU, centered on the current attachment anchor) encoded by a small 3D CNN into ~32 memory tokens, cross-attended by the decoder. Directly targets BrickNet's ~20-step collision wall. Breaks teacher-forcing parallelism → use **chunked snapshots every K=8 bricks** (parallel within chunk; refresh every brick at inference). Gated on the ablation metric below.
- **v3 (stretch) — rollback-free stability head.** Following PointCloud2Brick, add an auxiliary stability predictor so the model internalizes stability and inference needs little/no rejection. Defer until v1/v2 land.

### Conditioning encoders (classifier-free-guidance-ready)
- **Text:** frozen CLIP or T5-small + projection → prefix tokens (LegoACE uses CLIP).
- **Image/geometry:** **multi-view normal maps via DINOv2** (LegoACE's proven choice) and/or **point cloud** (BrickAnything's unified interface). Normal maps chosen as primary: shape-over-texture, and they bridge cleanly from the original photos→mesh idea (photogrammetry mesh → render normals → condition).
- **Target-voxel (optional):** 32³ occupancy of a target mesh as prefix tokens — the direct reconnection to "replica of a real ship/plane."
- **Condition dropout** 10–15% per modality independently → enables CFG, unconditional sampling, and any-subset conditioning from one model.

### Auxiliary heads
- **Completion-fraction head** (regression): fights unfinished generations, doubles as a progress signal.
- **Stability head** (v3).

---

## 5. Data pipeline

- **Parse → graph → execute to world poses → BFS attachment tree → tokenize.** Round-trip must be bit-exact (Stage 0 gate).
- **Dedup at the *design* level** (multiset of (part, quantized pose) after canonical alignment + voxel-signature fuzzy match). MOC re-uploads/recolors are rampant; file-level splits leak and inflate completion metrics. Treat suspiciously good completion numbers as a dedup bug.
- **Augmentation (the data multiplier):**
  - *Spanning-tree / traversal resampling*: 5–20 serializations per structure (different roots, BFS/DFS/random-frontier). Primary defense against human build-order distribution shift.
  - *Truncation for completion*: random prefix cuts 10–90% — the copilot training signal **and** an anti-unfinished signal. (LegoACE's subsequence sampling and the 2022 budget-aware paper both validate this.)
  - *Mirror* augmentation (swap chiral part IDs); *color jitter*.
- **Conditioning data:** render 8 views (RGB + **normal maps**) per structure; captions via a VLM in the verbose "This is a LEGO model of…" register LegoACE/BrickGPT used; extract **submodel labels** from LDraw subfiles + filenames, normalize via LLM into a label vocabulary (free hierarchy supervision).
- **Dataloader engineering:** precompute per-part coarse voxelizations once; composite per-step occupancy crops on the fly by transforming + OR-ing cached volumes. Profile early — geometry features are the likely bottleneck and a starved GPU doubles cost.

---

## 6. The three differentiators (where the actual contribution lives)

Because plain generation is now crowded, the project must deliver these — each is under-explored or absent in released work.

### 6.1 Interactive prefix-completion (the copilot)
Only the 2022 budget-aware paper (tiny 2×4 vocabulary) really does completion; no full-vocabulary interactive system exists. Train explicitly on "arbitrary human prefix → next bricks":
- Heavy truncation + traversal randomization so real (non-canonical) prefixes are in-distribution.
- **Locality signal:** encode the last-placed brick / a user-selected region as the attachment anchor so suggestions appear *where the builder is working*.
- Serve top-5 collision-filtered ghost-brick suggestions; short 1–5-brick proposals (the ~20-step horizon is a non-issue when proposing a handful at a time with a human in the loop).
- **Cheap heuristic co-suggesters that carry day-one value:** symmetry completion (mirror recent placements across a detected plane — near-deterministic for vehicles) and pattern continuation (extend a detected repeated run/stack). These need no model and set a usefulness floor.

### 6.2 Hierarchy conditioning (F-16 → wing → weapons)
Absent from all LEGO literature (closest analogues are PartNet/StructureNet in general 3D). Two-stage sequence: a **plan block** — list of (submodel label, part-budget, anchor pose) — then per-submodel build segments bracketed by `SUBMODEL_START/END`, trained from extracted subfile structure. Enables "generate just the left wing," per-part budgets, semantically-targeted copilot suggestions, and better global coherence. Ship flat first; add once Stage-2 works.

### 6.3 Finish-calibration
The observed "unfinished model" failure of the Qwen baseline, attacked from five sides: budget-countdown token (primary), completion-fraction head, completion-mixture training, CFG (stronger condition adherence per step), and an EOS floor (mask EOS until ≥70% of budget consumed; log how often it binds). Segment-wise re-anchoring bounds exposure-bias drift on 1,000+-part builds.

---

## 7. Training + alignment recipe

- **Stage 0 — pipeline validation (gate everything).** Bit-exact LDR→tokens→LDR on 1k structures; verify pointer candidate sets, pose execution, collision checker; overfit 25M on 100 structures to ~0 loss.
- **Stage 1 — pretraining.** 25M then 150–250M on the full corpus. Mixture: 60% full sequences / 30% truncated-completion / 10% mid-sequence windows. ~1B effective (augmentation-inflated) tokens. CE on placement tokens + 0.1× completion-head loss. AdamW, peak LR 1e-4 (LegoACE's value), cosine to 1e-5, ~2% warmup.
- **Stage 2 — conditional SFT.** Captioned + normal-map-rendered subset, vehicles upsampled 3–5×, condition dropout on, budget tokens on. Text-conditioned trained on complete pairs; normal-map-conditioned uses subsequence augmentation (per LegoACE).
- **Stage 3 — DPO alignment.** Now standard (LegoACE + BrickAnything both). Build preference pairs by generating 2 candidates per condition and preferring lower Chamfer/EMD-to-GT (geometry) and/or lower collision count + higher stability (buildability, per BrickAnything's buildability-aware reward). ~5k pairs, cheap (LegoACE: 15 min on 8×A100). Big, well-validated quality lift.
- **Stage 4 — copilot fine-tune.** Heaviest truncation mixture with "messy" prefixes + hierarchy conditioning.

**Compute.** 25M ablations: hours each on one 4090/A100 (tens of $). 150–250M main run: LegoACE trained 243M in **2 days on 8×A100**; a single-node rental for one full run ≈ few hundred to ~$1–2k. DPO: minutes. Budget 2–3 full runs → **project GPU estimate ~$1–3k**. Single-node throughout.

---

## 8. Evaluation protocol (freeze before training)

- **Headline: raw (unfiltered) per-step collision rate & connector-validity rate vs sequence position.** Measures spatial common sense *before* the sampler cleans up. Architecture changes that don't move this aren't working. (This is the metric BrickNet's ~20-step horizon implies room to beat.)
- **Buildability:** % valid, % stable, mean/min brick stability (BrickGPT's physics analysis), **rollback count** (BrickAnything: 6.75→0.42 is the bar), connected rate (LegoACE: 82% is the bar).
- **Fidelity:** MMD/1-NNA/COV on point clouds; Chamfer/EMD to GT for geometry-conditioned; CLIP/DINOv2/VQAScore for text/image conditioning. Compare directly against LegoACE and BrickNet on shared prompts.
- **Finishing:** |generated − requested budget| distribution; unforced-EOS rate; completion-head calibration.
- **Copilot:** top-5 next-brick hit rate (part+connection, color-agnostic), split repetitive vs novel regions; end-to-end suggestion latency (<1 s; LegoACE infers a whole model in 3 s, so a few-brick suggestion is easily sub-second).
- **Human eval:** 20 vehicle prompts, blind preference vs LegoACE/BrickNet; plus your own hands-on Studio sessions for the copilot loop.

---

## 9. Milestones + decision gates

- **M0 (wk 1–2):** data access resolved (BrickNet + LegoACE), round-trip + collision + stability infra working, LegoACE code reproduced. *Gate: bit-exact round-trip; if both curated datasets denied → raw-LDraw fallback, +4–6 wks.*
- **M1 (wk 3–4):** 25M baseline (v0+v1, grammar+pointer masking). *Gate: raw collision curve beats BrickNet's implied ~20-step horizon; filtered survival >100 steps.*
- **M2 (wk 5–6):** geometry ablations (v0 vs v1 vs +v2 occupancy). Pick architecture. *Gate: chosen config's raw collision curve flat-ish to 200+ steps.*
- **M3 (wk 7–9):** 150–250M pretrain + conditional SFT + DPO. *Gate: matches/beats LegoACE on CLIP/fidelity for vehicles; ≤10% unfinished generations; rollback count near BrickAnything's.*
- **M4 (wk 10–12):** copilot fine-tune + minimal Studio file-watch UI + hands-on trial. *Gate: you personally accept ≥1 suggestion per few minutes of real building.*
- **M5 (month 4+):** hierarchy plan tokens; real-photo→normal-map conditioning; 16k context / 1,500-part builds; rollback-free stability head.

---

## 10. Risks + mitigations

- **Curated data unavailable** → LegoACE code/models are the hedge for BrickNet; raw-LDraw rebuild is the deeper fallback. Resolve week 1. *(Highest-variance risk.)*
- **Novelty erosion (field moving fast)** → contribution is explicitly completion + hierarchy + finish-calibration + full-vocabulary vehicles, none of which released systems cover; not "another text→brick model." Re-scan arXiv monthly; the May 2026 cluster shows ~monthly releases.
- **Connector-annotation gaps on exotic/Technic parts** (vehicles use many pins/panels) → verify ≥95% coverage on the vehicle subset early; truncate generation vocab to annotated frequent parts.
- **Occupancy conditioning too slow / non-parallel** → chunked snapshots; fall back to v1 embeddings (may suffice).
- **Duplicate leakage inflating completion metrics** → design-level dedup + split audits.
- **Real-photo domain gap** → normal-map conditioning + photo-style render augmentation + DINOv2 texture-robustness; if weak, insert photo→mesh→normal-map bridge (photogrammetry) rather than conditioning on raw photos.
- **Articulation (hinges) under-learned** (rare in data, high vehicle impact) → oversample hinge-containing structures; track FINE_geom perplexity separately.
- **Exposure bias at 1,000+ parts** → segment-wise re-anchoring, completion-mixture training; set expectations: first release targets ≤500-part quality, 1,500 is the stretch.

---

## 11. Immediate next actions

1. Request BrickNet data; clone + run **LegoACE** (xh38.github.io/LegoACE) as the fastest working reference; verify what code/weights/meshes each has actually released.
2. Stand up an LDraw→graph→execute→(normal-map + point-cloud) render pipeline on 100 OMR **vehicle** files — de-risks the raw fallback in parallel.
3. Write the hybrid tokenizer (Sec. 3) as code + bit-exact round-trip tests. Start from LegoACE's tokenizer, swap absolute-pose tokens for BrickAnything-style parent-relative attachment + BrickNet typed connectors.
4. Freeze the vehicle eval set: 200 held-out structures + 50 prompts + 50 target images/normal-map sets, before any training.
5. Rent one A100, run the Stage-0 overfit test on the 25M model.

---

### One-paragraph summary
Build on **BrickNet's data + typed-connector representation** as the spine, adopt **BrickAnything's parent-relative attachment-tree tokenization** for coherence, borrow **LegoACE's native-tokenizer + DPO + normal-map conditioning** engineering, and add **occupancy-in-state** to break the ~20-step collision horizon. The plain generator is table stakes; the real, defensible contribution is **interactive prefix-completion + hierarchical (object→part) conditioning + finish-calibration on the full vehicle vocabulary** — validated cheaply on a 25M model before any expensive run.
