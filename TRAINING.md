# Training runbook

Sequential commands to train the native LEGO transformer on a GPU box (rented VM or Colab).
Run top to bottom. We train on the **`pt`** (pretrain) split via `pt.npz` — the compact graphs —
and tokenize them natively, so the 11 GB `paths_pt.jsonl` text is **never downloaded**.

---

## 0. Prerequisites

- A GPU with ≥ 16 GB VRAM (L4 / A10 / A100). L4 is fine for the 25M model.
- NVIDIA driver + CUDA (preinstalled on Colab and most cloud GPU images).
- Python 3.10+.
- Your Yandex Object Storage **key id + secret** (for the dataset bucket).
- ~10 GB free disk (dataset + tokenized `.bin` + checkpoints; the big text file is not used).

---

## 1. Get the code

```bash
git clone <YOUR_REPO_URL> lego && cd lego          # or scp the repo onto the box
python3 -m venv .venv && . .venv/bin/activate       # skip on Colab (use the system env)
```

## 2. Install dependencies

```bash
pip install -r requirements.txt                     # pulls the CUDA torch wheel on a GPU box
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi                                           # confirm the GPU is visible
```

`torch.cuda.is_available()` must print `True`. (On Colab torch is preinstalled; this step just
adds `bricknet` + `numpy`.)

## 3. Configure dataset access (once per box)

```bash
aws configure --profile brickformer                 # enter Access Key ID + Secret (region/output: just press Enter)
aws configure set profile.brickformer.region ru-central1
aws configure set profile.brickformer.endpoint_url https://storage.yandexcloud.net

aws --profile brickformer s3 ls s3://brickformer/bricknet/   # sanity: should list pt.npz.xz, val.npz.xz, ...
```

> Enter the secret in the box's own shell, not anywhere it gets logged/shared. On Colab, use a
> **private** runtime and do **not** mount Google Drive for the dataset — keep it on the ephemeral
> `/content` disk so it's wiped when the runtime ends.

## 4. Prepare the data (download + tokenize)

```bash
bash scripts/prepare.sh
```

This downloads `pt.npz` + `val.npz`, then tokenizes them to `data/pretrain.bin` and `data/val.bin`.
Tokenization memory-maps each split and streams one graph at a time, so **peak RAM stays ~flat
regardless of split size** (the full `pt` split preps fine on a ~13 GB Colab box), and it shards
across **all CPU cores** for near-linear speedup. Override the core count with `WORKERS=4 bash
scripts/prepare.sh` if you want to cap it.

Check the printed summary:

- `n_tokens` — the real pretrain token count (prints when it finishes).
- `skipped` — should be **0**. If non-zero, a field exceeded a vocab cap (see `field_max` vs
  `field_caps` in `data/pretrain.bin.meta.json`) — tell me and I'll bump the cap.

## 5. Train

```bash
# 25M ablation model, sensible defaults (ctx 1024, batch 32, 20k iters ≈ 650M tokens)
bash scripts/train.sh

# larger / custom runs — extra flags pass straight through:
bash scripts/train.sh --size 250M --batch 16 --grad-accum 4 --max-iters 60000
```

Live log shows `loss | lr | tok/s | ETA`, and a **val loss every 1000 iters** — so you'll see the
real ETA within a minute of starting. Sizes: `tiny`, `5M`, `25M`, `125M`, `250M`.

## 6. Outputs

```
runs/pretrain-<size>/best.pt     # lowest val-loss checkpoint (model + config)
runs/pretrain-<size>/last.pt     # final checkpoint
runs/pretrain-<size>/train.json  # params, tokens, best val loss, wall-clock minutes
data/pretrain.bin.meta.json      # corpus stats + observed field maxima
```

## 7. Evaluate the trained model

Val loss (in `train.json`) tells you the model is learning, but the real quality metrics are about
the *builds it generates*. The eval harness samples builds and scores them:

```bash
# collision scoring needs the inset meshes once per box (from the dataset's inset.tar.xz):
mkdir -p data/bricknet_data/inset && tar -xJf data/inset.tar.xz -C data/bricknet_data/inset
export BRICKNET_DATA="$PWD/data/bricknet_data"     # or: python -m bricknet fetch-meshes

python -m lego_tf.bnet.evaluate --ckpt runs/pretrain-25M/best.pt --n 256 --export runs/eval
```

Reports (and writes `runs/eval/eval.json` + `collision_curve.csv` + sample `.ldr` files):

- **validity** — fraction that decode to a build (grammar-constrained, so ~100%).
- **connector-valid** — fraction whose (part, connector) pairs physically realize. The grammar
  masks connectors to compatible ones, so this is **~100% by construction** (a sanity check on the
  masking, not a learning signal); a build that names a non-attachable part is truncated there.
- **collision-free rate** and **collision-free horizon** — longest collision-free placement prefix
  per build. **BrickNet's implied bar is ~20 steps** — beating it is the M1 gate.
- **per-step collision curve** — collision rate vs sequence position (the architecture signal).
- **unforced-EOS rate** — did it learn to stop on its own.

Open the exported `runs/eval/sample_*.ldr` in any LDraw viewer (Studio, LeoCAD, or an online
viewer) to eyeball the builds. Drop `--export`/set `--no-collision` for a quick parse-only check
without meshes.

### Collision-free decoding (the M2 upper bound)

Pure next-token sampling collides constantly (horizon ~2–5) because the loss never penalizes
overlap. `--collision-free` adds a per-build collision scene (bricknet's mesh checker): when a
sampled brick would interpenetrate the structure, its tokens are rolled back and the brick is
**resampled** (up to `--max-retries`, default 8); on exhaustion the build ends. Every build is then
**collision-free by construction** — `collision-free builds` and `horizon ≥N` should read ~100%.

```bash
python -m lego_tf.bnet.evaluate --ckpt runs/pretrain-25M/best.pt --n 256 \
    --collision-free --temperature 1.0 --export runs/eval-cf
```

This answers the key question directly: **once we force collision-free, do the structures actually
look like vehicles?** Compare `runs/eval-cf/sample_*.ldr` (constrained) against the plain
`runs/eval/sample_*.ldr`. Needs the inset meshes; sampling only (greedy can't escape a rejection),
and it's slower than plain decoding (a geometry check per brick, plus retries). If the forced
builds still look wrong, the fix is a training signal (DPO on the collision/realism scorer), not
just decoding — this eval sets that ceiling.

## 8. v1 resolved-pose ablation (optional)

v1 feeds each token the resolved world pose of the *previously placed* brick (translation + 6D
rotation, Fourier-featured) so overlap is directly visible to attention — targeting the collision
horizon. Tokens are unchanged, so it's a clean v0-vs-v1 comparison.

```bash
# 1. add the aligned pose stream to the prepared data (writes data/*.bin.pose.f16):
POSES=1 bash scripts/prepare.sh                 # or: prepare_data ... --poses

# 2. train v1 with the SAME recipe as v0 (only variable = pose). Output: runs/pretrain-25M-v1/
USE_POSE=1 bash scripts/train.sh                 # add the same --batch/--ctx/--max-iters/--lr you used for v0

# 3. evaluate v1 (generation auto-resolves poses) and compare curves to v0:
python -m lego_tf.bnet.evaluate --ckpt runs/pretrain-25M-v1/best.pt --n 256 --export runs/eval-v1
```

**M2 gate:** v1's per-step collision curve is flatter and its collision-free horizon moves
meaningfully past v0 / the ~20-step bar. Note: v1 generation is slower (it resolves each brick's
pose incrementally); lower `--n` or `--max-new` if needed.

---

## Reference: time & GPU

| Run | Tokens | L4 / A10 | A100 | 8×A100 |
|---|---|---|---|---|
| **25M** (default) | ~650M | ~1.5–2.5 h | ~30–40 min | ~10 min |
| **250M** (`--size 250M`) | ~1 B | ~12–18 h | ~3–5 h | ~1–2 days |

Estimates ±2×; the printed `tok/s` gives the real number. **Machine:** any GPU ≥16 GB works;
~8 GB **RAM** and ~10 GB **disk** are enough (prep memory-maps the split and streams one graph at
a time, so RAM stays flat even on the full `pt` split). More CPU cores = faster prep — tokenization
is ~780 graphs/s **per core** and runs on all of them.

## Tuning knobs

| Flag | Effect |
|---|---|
| `--size` | model preset (VRAM + quality) |
| `--batch` | per-step batch (lower if OOM) |
| `--grad-accum N` | keep effective batch when `--batch` is small |
| `--ctx` | context length (default 1024; raise for very large builds) |
| `--max-iters` | training length; tokens = `batch × grad_accum × ctx × iters` |
| `--lr` | peak LR (default 3e-4, cosine to 3e-5) |

**Out of memory?** Lower `--batch` (e.g. 8) and/or `--ctx` (e.g. 512); add `--grad-accum 4` to
keep the effective batch size.

## Smoke test (optional, no GPU needed)

Proves the whole pipeline end-to-end on a laptop/CPU before renting a GPU (needs `val.npz` — a
1.7 MB download):

```bash
aws --profile brickformer s3 cp s3://brickformer/bricknet/val.npz.xz data/ && unxz -kf data/val.npz.xz
python -m lego_tf.bnet.train_overfit --n 16 --steps 800     # loss 9.8 -> ~0.1, generates a valid build
```
