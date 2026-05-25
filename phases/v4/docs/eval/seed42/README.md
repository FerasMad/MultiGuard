V4 Stage 2 retrain — seed 42 (FSOS RTX 4070)
============================================

Frozen copies of the eval artifacts from `outputs/v4/eval/` and the
training history from `outputs/v4/stage2_fusion_dctforensic/` (both
gitignored at the repo level — these copies are the version-controlled
record).

## Setup

| Item | Value |
|------|-------|
| Config | `phases/v4/configs/v4_pipeline_dctforensic_v3caches.yaml` |
| Encoder (image) | `dct_forensic_v1` (loaded from `phases/forensic/outputs/dct/forensic_dct_model.pth`) |
| Encoder (semantic) | V3 raw FND-CLIP 512-d cache → projection 512→768 |
| Encoder (text) | V3 raw Qwen2-7B-Instruct 3584-d cache → projection 3584→768 |
| Fusion | `V3PairwiseFusion` with `proj_dims={v_semantic: 512, v_textfor: 3584}` (P5.10a) |
| Manifest | `data/processed/forensic_5class_unified.csv` (16500 rows, V3 c4fix repointed to MultiGuard layout) |
| Seed | 42 |
| Best epoch | 4 (early-stopped at ep 14, patience=10) |

## Headline numbers

| Metric | Value | Notes |
|---|---|---|
| Val F1-macro | **0.7334** | V3 baseline 0.7215 → +1.19 pp |
| Test F1-macro | **0.7267** | First V3.1 §7 5-class test for any of our checkpoints |
| MMFakeBench transfer F1-macro | **0.4805** | V3 transfer 0.3832 → +9.7 pp |

## Per-class test F1

| Class | Name | F1 | Support | Notes |
|---|---|---|---|---|
| 0 | Real | 0.41 | 495 | Confused with OOC (column 1: 246 of 495) |
| 1 | Out-of-Context | 0.47 | 495 | Confused with Real (178 of 495) |
| 2 | Manipulated | 0.76 | 495 | Decent — DGM4 + MMFakeBench |
| 3 | AI-Text | **0.997** | 495 | V3 caption-shortcut persists via reused V3 text cache |
| 4 | Fully-Fabricated | **0.998** | 495 | Same shortcut |

## Files

| File | What it is |
|---|---|
| `test_metrics.json` | sklearn precision/recall/F1 per class + macro |
| `test_classification_report.txt` | human-readable sklearn report |
| `test_confusion_matrix.png` | 5x5 heatmap |
| `test_per_source.json` | per-source breakdown (NewsCLIPpings, DGM4, MMFakeBench_*) |
| `transfer_*` | MMFakeBench transfer probe (502 samples) |
| `training_history.csv` | per-epoch train_loss, val_loss, val_F1, lr, seconds |

See `STATUS.md` (root) for the live narrative and `docs/MASTER_CHECKLIST.md` for
V3.1 spec compliance.
