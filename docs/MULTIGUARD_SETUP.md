# multiGuard PC — Setup & Transfer Recipe

How to get the full MultiGuard environment running on the dual-4090 PC
(the "multiGuard" machine), so it's ready for:

1. **Verifying the forensic ship** (reproduce the val_AP 0.9848 result).
2. **Approach 1** (RGB + Fourier mask) — Linux-only repo, runs fine on Windows via Git Bash.
3. **V4 retrain** — use `forensic_dct_model.pth` as Stage 1 init weights.

**Key design:** code + small artifacts come from git, the 282 MB checkpoint
comes from HuggingFace Hub (`FerasMad/forensic-dct-v1`), large datasets are
either regenerated from scratch (~2 min) or already on the multiGuard PC
from prior V3 work.

---

## Pre-flight (one-time, on multiGuard PC)

```powershell
# 1. Python 3.12 (or use existing if you already have one)
python --version   # expect 3.12.x

# 2. NVIDIA CUDA 12.4+ driver — should already be there for the 4090s
nvidia-smi

# 3. Free disk (~30 GB is comfortable)
Get-PSDrive C
```

```bash
# 4. Tools: git, HuggingFace CLI
git --version

# 5. HuggingFace login (read token is enough for downloads;
#    only needed if you want to upload too)
#    -> https://huggingface.co/settings/tokens
hf auth login
```

---

## Step 1 — Clone repo (~30 sec)

```bash
cd C:\Desktop     # or your preferred parent dir
git clone https://github.com/FerasMad/MultiGuard.git
cd MultiGuard
```

This gives you:
- All code (`phases/forensic/`, `phases/v4/`, `app/`, `scripts/`)
- All small artifacts (`REPORT.md`, `REPORT.docx`, `eval_table.md`,
  `dct_stats.json`, `splits_summary.json`, `training_history.csv`,
  `train_summary.json`, `eval_dct.json`, `training_curves.png`)
- All docs (`docs/MASTER_CHECKLIST.md`, `docs/DECISIONS.md`,
  `phases/forensic/FOLLOWUP.md`)

---

## Step 2 — Create Python venv + install deps (~5 min)

```bash
# In MultiGuard/ root
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1
# or in Git Bash:
# source .venv/Scripts/activate

# Upgrade pip
python -m pip install --upgrade pip wheel

# Install the project + GPU extras (torch, transformers, etc.)
pip install -e ".[gpu,server,dev]"

# Extras the forensic pipeline needs that aren't in the core deps:
pip install pyarrow matplotlib pypandoc-binary
```

Quick sanity check:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count(), 'devices')"
# Expect: CUDA: True 2 devices  (because dual 4090)

# Forensic + v4 imports
python -c "from forensic.models.dct_resnet50 import build_dct_resnet50; print('forensic OK')"
python -c "import v4; print('v4 OK')"

# Run the unit tests (no data required)
pytest phases/forensic/tests/ phases/v4/tests/smoke/ -v
# Expect: 17 passed
```

---

## Step 3 — Download the trained checkpoint from HF Hub (~30 sec)

```bash
# Download just the .pth + dct_stats.json
hf download FerasMad/forensic-dct-v1 forensic_dct_model.pth \
    --local-dir phases/forensic/outputs/dct/

# dct_stats.json is already in git, but if you want the HF mirror:
hf download FerasMad/forensic-dct-v1 dct_stats.json \
    --local-dir phases/forensic/data/

# Verify checkpoint loads
python -c "
import torch
ckpt = torch.load('phases/forensic/outputs/dct/forensic_dct_model.pth',
                  map_location='cpu', weights_only=False)
print('epoch:', ckpt['epoch'])
print('val_ap:', ckpt['val_ap'])
print('phase:', ckpt['phase'])
"
# Expect: epoch=13, val_ap=0.984812..., phase=2
```

You now have a fully working inference-ready checkpoint. If your only
goal is showing the doctor a working model, you can stop here.

---

## Step 4 — Acquire data (only if you want to retrain / reproduce)

The detector trains on **6 of 8 generators** (sdv1_4 / sdv1_5 missing —
see `phases/forensic/FOLLOWUP.md` section A). Per-generator targets:
1750 AI + 1750 nature.

### 4a. Re-run the prep script (works automatically for 5 gens)

```bash
# Downloads ~5 bitmind parquets from HF Hub + samples VisualNews nature
python phases/forensic/scripts/prepare_genimage_v2.py --target-per-gen 1750
```

This works if **VisualNews is on disk** at `data/raw/visualnews/origin/`.
If it's not, see step 4c.

### 4b. The wildcard: midjourney 10K (1.4 GB)

The midjourney row of the eval table comes from a local 10K JPG cache
at `data/raw/GenImage/ai/midjourney/` (256x256). Two paths:

- **If multiGuard PC already has it** (from V3 work) — done, the prep
  script will pick it up automatically.
- **If not** — either TeamViewer-transfer it from FSOS PC (~5 min via
  zip-first), or skip midjourney entirely:
  ```bash
  python phases/forensic/scripts/prepare_genimage_v2.py \
      --target-per-gen 1750 \
      --only-gens wukong vqdm biggan adm glide
  ```
  Eval table will show midjourney as `_skipped_` in that case (5/8 gens).

### 4c. VisualNews not on disk

Per V4 plan Appendix A.5.5, VisualNews was scheduled to be transferred
to the multiGuard PC. If it's not there yet, follow that recipe (zip on
FSOS, TeamViewer transfer, extract on multiGuard). Or grab from
HF community mirrors of the news-images subset.

### 4d. Build splits + precompute DCT cache (~3 min on a 4090)

```bash
python phases/forensic/scripts/build_splits.py \
    --train-per-gen 1250 --test-per-gen 500 --val-frac 0.20

python phases/forensic/scripts/precompute_dct.py --workers 4

python phases/forensic/scripts/compute_dct_stats.py
```

At this point you have everything you need to retrain from scratch.

---

## Step 5 — (Optional) Full retrain to verify

```bash
# Should take ~5-10 min on a single 4090 (3-4x faster than FSOS's 4070).
python phases/forensic/scripts/train_dct.py \
    --out-dir phases/forensic/outputs/dct_multiguard

# Compare best val_AP to the FSOS run (0.9848); should be within +/- 0.005.
python phases/forensic/scripts/eval_dct.py \
    --ckpt phases/forensic/outputs/dct_multiguard/forensic_dct_model.pth \
    --out-json phases/forensic/outputs/eval_dct_multiguard.json \
    --out-table phases/forensic/outputs/eval_table_multiguard.md
```

---

## Step 6 — Approach 1 (RGB + Fourier mask)

See `phases/forensic/FOLLOWUP.md` section B for the full recipe.
On multiGuard PC (Windows with Git Bash) the relevant steps are:

```bash
# 1. Clone the external repo
git clone https://github.com/chandlerbing65nm/FakeImageDetection \
    phases/forensic/external/FakeImageDetection

# 2. Manually download `mask_15/rn50ft_fouriermask.pth` from the Google
#    Drive link in their README, save to:
#    phases/forensic/external/FakeImageDetection/checkpoints/mask_15/

# 3. Patch the repo (idempotent):
python phases/forensic/scripts/adapt_fakeimagedetection.py
# (script doesn't exist yet — see FOLLOWUP.md B step 3)

# 4. Train:
bash phases/forensic/scripts/train_rgb_fourier.sh

# 5. Eval:
python phases/forensic/scripts/eval_rgb.py \
    --ckpt phases/forensic/outputs/rgb/best_rgb.pth
```

---

## Step 7 — V4 retrain (the dual-4090's real purpose)

See `phases/forensic/FOLLOWUP.md` section D for the full plan. High level:

1. Wrap `forensic_dct_model.pth` as a V4 `EncoderBase` (~50 LOC).
2. Update `phases/v4/configs/v4_pipeline_qwen.yaml`.
3. Run `python -m v4 precompute --feature v_imgfor` (regenerates V4
   image cache with the new encoder, ~30 min).
4. Reuse the existing `cache/v3/v_textfor_qwen/` (3584-dim, matches V4's
   text projection).
5. Train Stage 0 FND-CLIP (~1.5 h on dual 4090).
6. Train Stage 2 with 3 seeds {42, 1337, 2024} — two in parallel on the
   dual 4090 + one sequential. Total ~7 h.
7. V4 section 7 eval: P/R/F1-macro/CM + MMFakeBench probe.

---

## What you do NOT need to transfer from FSOS

- `.venv/` — rebuild on multiGuard (different OS / paths).
- DCT cache (`phases/forensic/data/dct_cache/`) — regenerated in 43 sec.
- Train/val/test split (`phases/forensic/data/genimage_{train,test}/`) —
  regenerated by `build_splits.py`.
- Logs (`phases/forensic/outputs/*.log`) — already inspected; not useful
  to re-host.
- Intermediate checkpoints (`best.pt`, `latest.pt`) — `forensic_dct_model.pth`
  is the same as `best.pt` (renamed); `latest.pt` is for resuming a
  training run, irrelevant once training is done.

---

## Sanity check: how to know the transfer worked

After step 3, the val AP from the loaded checkpoint should match exactly:

```bash
python -c "
import torch
ckpt = torch.load('phases/forensic/outputs/dct/forensic_dct_model.pth',
                  map_location='cpu', weights_only=False)
assert abs(ckpt['val_ap'] - 0.984812) < 1e-4
print('Checkpoint matches FSOS-side training.')
"
```

If you also re-eval on the test set (step 4 + step 5's eval_dct.py), the
per-generator numbers should be **identical** because the test data
comes from the same bitmind parquets and the inference is deterministic.

---

## TL;DR — the absolute minimum to demo to the doctor

```bash
# On multiGuard PC, fresh shell:
git clone https://github.com/FerasMad/MultiGuard.git
cd MultiGuard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gpu]" pyarrow matplotlib
hf download FerasMad/forensic-dct-v1 forensic_dct_model.pth \
    --local-dir phases/forensic/outputs/dct/

# Doctor sees: phases/forensic/REPORT.md / REPORT.docx + trained .pth
# Five commands, ~7 minutes total.
```
