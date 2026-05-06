# Phase 2 — Multimodal Forgery Detection: Handoff Notes

This commit adds the Phase 2 implementation following the `Implementation
Guidelines (V2)` PDF: a forensic-only baseline (Step 1) and a full
fusion pipeline that combines frozen FND-CLIP semantic features with a
ResNet18 forensic encoder via cross-attention (Step 2).

---

## ⚠ Read this before trusting any numbers

The current Step 2 results are inflated by a **frozen-feature-extractor leak**.
The pipeline uses a previously trained FND-CLIP checkpoint
(`outputs/v1_ooc/best.pt`) as the source of `v_semantic`. That checkpoint
was a binary "OOC vs not-OOC" model trained on `balanced_dataset.csv`,
and its training set overlaps heavily with the forensic 3-class splits
we built on top:

|                     | overlap with v1_ooc training |
|---------------------|------------------------------|
| our **test** OOC    | 757 / 900  (84 %) |
| our **train** OOC   | 2,908 / 4,200 (69 %) |

So `v_semantic` for those samples literally encodes the OOC label that
v1_ooc was trained to predict. The OOC F1 of 0.98 we measured is mostly
that leak, not generalization.

There is also a smaller secondary issue: NewsCLIPpings OOC images are
stored at ~22 % lower JPEG quality than DGM4 origin/manipulated images
(36.7 KB vs 47.5 / 45.8 KB on average), which DCT can shortcut on.

**Action items before relying on these numbers:**
1. Re-run Step 2 with FND-CLIP **uninitialized from any task checkpoint**
   (i.e. drop `model.fnd_clip_ckpt` from `config/full_pipeline.yaml` and
   pass a fresh `FNDCLIP(...)` so `v_semantic` comes only from
   pretrained ImageNet/BERT/CLIP encoders, not a model that already saw
   our test labels).
2. Re-encode all images at a uniform JPEG quality before DCT precompute,
   to remove the source-distribution shortcut.

---

## What's in this commit

### New files

| Path | Purpose |
|---|---|
| `scripts/build_forensic_3class.py` | Builds the 3-class CSV (Real / Manipulated / OOC) from DGM4 origin + DGM4 face_* + NewsCLIPpings OOC. Balanced 6 000/class, 70/15/15 per-class split. |
| `scripts/precompute_dct.py` | Precomputes DCT maps (BGR→YCbCr→Y, 2D DCT, log scale, per-image min-max) and caches them as `.pt` files keyed by image-path md5. |
| `scripts/precompute_fnd_features.py` | Precomputes frozen FND-CLIP `v_semantic` vectors so training doesn't pay the 300 M-param forward-pass cost every epoch. Critical optimization for V2 §global ("computational cost"). |
| `src/models/forensic_baseline.py` | ResNet18 forensic encoder (1-channel input from DCT, 512-dim output) plus a linear classifier head — the Step 1 baseline. |
| `src/models/full_pipeline.py` | Full Step 2 pipeline: frozen FND-CLIP + trainable ResNet18 forensic + bidirectional cross-attention fusion (FCINet 2024 style) + 3-layer MLP classifier. |
| `src/train_forensic.py` | Training script for Step 1 (V2 §10–11). |
| `src/train_full_pipeline.py` | Training script for Step 2; loads cached `v_semantic` so each epoch is ~25 s on MPS instead of ~50 min. |
| `src/evaluate_forensic.py` | Test-split + MMFakeBench transfer eval for Step 1. |
| `src/evaluate_full_pipeline.py` | Same for Step 2. |
| `config/forensic_baseline.yaml` | Step 1 hyperparams. |
| `config/full_pipeline.yaml` | Step 2 hyperparams. |

### Modified files

- `src/models/fnd_clip.py` — added `forward_semantic()` returning the modality-attention output without going through the classifier head. Used by Step 2.

### Things excluded from the repo (regenerate locally)

These are gitignored and never get pushed:

| Path | Size | How to regenerate |
|---|---|---|
| `data/raw/DGM4/` | many GB | download from rshaojimmy/DGM4 (HF) |
| `data/raw/NewsCLIPpings/` | many GB | NewsCLIPpings test split |
| `data/raw/MMFakeBench/` | several GB | liuxuannan/MMFakeBench (HF) |
| `data/processed/forensic_3class.csv` | ~3 MB | `python scripts/build_forensic_3class.py` |
| `data/processed/dct_cache/` | 3.4 GB | `python scripts/precompute_dct.py` |
| `data/processed/fnd_features/` | 70 MB | `python scripts/precompute_fnd_features.py` |
| `outputs/*/best.pt` | ~50 MB each | retrain with `src/train_*.py` |

---

## How to reproduce Step 1 (forensic baseline)

```bash
# 1. Build the 3-class dataset
python scripts/build_forensic_3class.py

# 2. Precompute DCT maps (~1 minute, 18,000 images)
python scripts/precompute_dct.py

# 3. Train (50 epochs max, early-stop patience 10 on val F1-macro)
python src/train_forensic.py

# 4. Evaluate on test split + MMFakeBench transfer
python src/evaluate_forensic.py --checkpoint outputs/forensic_baseline/best.pt
```

Expected behaviour: F1-macro ~0.47 in-distribution, drops to ~0.17 on
MMFakeBench transfer. Validates the project hypothesis that pure
forensic features don't transfer to AI-generated content.

## How to reproduce Step 2 (full pipeline)

```bash
# Steps 1-2 from above must already be done.

# 3. Precompute FND-CLIP semantic vectors (one-time)
python scripts/precompute_fnd_features.py

# 4. Train (cached features → ~5 minutes total on MPS)
python src/train_full_pipeline.py

# 5. Evaluate
python src/evaluate_full_pipeline.py --checkpoint outputs/full_pipeline/best.pt
```

Current numbers: F1-macro 0.71 in-distribution, OOC F1 0.98 — but as
flagged at the top, **the OOC number is mostly leakage from v1_ooc**.

---

## Architecture deviations from V2 PDF (intentional, documented)

1. **`feat_dim = 512` instead of 768.** The V2 spec says 768 for
   `v_semantic` and `v_forensic`. We use 512 because the existing
   pretrained FND-CLIP checkpoint is at 512. The cross-attention block
   projects both inputs to a common 512 anyway, and the concatenated
   fused output is 1024 either way.

2. **Auxiliary classifier on the forensic stream is binary, not 3-class.**
   The PDF (§6) specifies `Linear(768, 3)` aux. We use `Linear(512, 2)`:
   real-image vs fake-image. Reason: the forensic encoder sees only DCT
   pixel artifacts and *cannot* distinguish OOC from Real (both have
   real images). A 3-class aux supervises an impossible task and was the
   cause of Step 1 plateauing at F1 0.49. Binary supervision aligns the
   aux loss with what the encoder can actually learn; the 3-class job
   lives at the fusion head where all signals are present.

---

## Open work

- Fix the v1_ooc feature leak (use a non-task-tuned FND-CLIP)
- Remove the JPEG-quality source shortcut (uniform re-encode before DCT)
- Implement and benchmark the **FakeInversion alternative** for the
  forensic vector (DCT+ResNet ⊕ FakeInversion → concat). Per the
  guidelines, this carries high compute cost — feasibility study still
  needed.
- Domain-adaptive training including MMFakeBench fake samples to close
  the OOD generalization gap (Step 2 transfer accuracy still ~19 %).
