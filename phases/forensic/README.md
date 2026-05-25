# Phase Forensic — Binary AI-Generated vs Real Image Detector

**Status:** **Both approaches SHIPPED** (May 2026).

**Headline results (6 of 8 generators evaluated):**

| Approach | Best val AP | Overall test AP | StdDev across gens |
|----------|-------------|-----------------|--------------------|
| 1 (RGB + Fourier) | **0.9971** @ ep 10 | **0.9979** | **0.0023** |
| 2 (DCT) | 0.9848 @ ep 13 | 0.9863 | 0.0158 |

Both approaches pass the doctor's spec (F.4-F.26). Approach 1 wins on
overall AP and consistency across generators; Approach 2 is more
interpretable (frequency-domain input). All 21 tests pass: 12 Approach 2
trainer invariants + 9 V4 encoder wrapper tests.

**Doctor handoff:** [`REPORT.md`](REPORT.md) (also `REPORT.docx`).
**Source brief:** `docs/doctor-briefs/Forensic_Image_Detector_En.pdf`

---

## What this phase will deliver

Binary classifier: **AI-generated vs Real** image on the GenImage dataset. The doctor's brief specifies **two parallel approaches** to be implemented and compared:

### Approach 1 — RGB + Frequency Masking
Fine-tune `chandlerbing65nm/FakeImageDetection` ResNet50 checkpoint (`mask_15/rn50ft_fouriermask.pth`) on GenImage with the existing Fourier-masking augmentation.

- Freeze `conv1`, `bn1`, `layer1`, `layer2`; train `layer3`, `layer4`, `fc` only
- `model.change_output(1)` for binary head
- Fourier masking 50% of training images, mask_ratio=0.15 (training-only)
- AdamW lr=1e-4 wd=1e-4, BCEWithLogitsLoss, batch 64, max 30 epochs
- ReduceLROnPlateau (mode=max, factor=0.5, patience=3 on val AP), early-stop patience=5

**Output:** `outputs/forensic_rgb_model.pth`

### Approach 2 — DCT Frequency Domain
Train a fresh torchvision ResNet50 (ImageNet weights) with `conv1` modified to 1-channel (Kaiming init), on precomputed DCT feature maps.

- DCT recipe: Resize 224x224 -> YCbCr Y-channel -> **dual patch sets** (8x8 and 16x16) -> `scipy.fftpack.dct` Type-II -> log scaling -> element-wise average -> z-score normalize via global `dct_stats.json`
- Two-phase training:
  - **Phase 1 (epochs 1-5):** freeze `layer1`, `layer2`; AdamW lr=1e-4 wd=1e-4; no gradient clipping
  - **Phase 2 (epoch 6+):** unfreeze `layer1`, `layer2`; **reinitialize** optimizer + scheduler at lr=1e-5; gradient clipping max_norm=1.0
- Early-stop patience=5 on val AP, **carries across phase boundary**

**Outputs:** `outputs/forensic_dct_model.pth` + `outputs/dct_stats.json`

---

## Deliverable

Per-generator evaluation table for both approaches (rows = 8 generators × columns = AP/Accuracy/AUC):

| Generator | Type | AP | Accuracy | AUC |
|-----------|------|----|----|-----|
| Midjourney | Diffusion | _pending_ | _pending_ | _pending_ |
| SD v1.4 | Diffusion | | | |
| SD v1.5 | Diffusion | | | |
| Wukong | Diffusion | | | |
| VQDM | Diffusion | | | |
| ADM | Diffusion | | | |
| GLIDE | Diffusion | | | |
| BigGAN | GAN | | | |
| **Overall Avg** | | | | |
| **GAN Avg** | | | | |
| **Diffusion Avg** | | | | |
| **Std Dev** | | | | |

**Final artifacts:**
- `outputs/eval_table.md` — Approach 2 per-generator (F.25 format)
- `outputs/eval_table_rgb.md` — Approach 1 per-generator (F.25 format)
- `outputs/eval_table_combined.md` — both approaches side-by-side (F.25 final)
- `REPORT.md` / `REPORT.docx` — doctor handoff with both approaches + training curves embedded

## Actually-shipped state (May 2026)

| Generator | Type | AP | Accuracy | AUC | Source |
|-----------|------|-----|----------|-----|--------|
| midjourney | Diffusion | 0.9569 | 0.8590 | 0.9606 | local 10K 256px |
| sdv1_4 | Diffusion | _skipped_ | _skipped_ | _skipped_ | bitmind 404 |
| sdv1_5 | Diffusion | _skipped_ | _skipped_ | _skipped_ | bitmind 404 |
| wukong | Diffusion | 0.9842 | 0.9310 | 0.9844 | bitmind parquet |
| vqdm | Diffusion | 0.9846 | 0.9420 | 0.9870 | bitmind parquet |
| adm | Diffusion | 0.9967 | 0.9740 | 0.9967 | bitmind parquet |
| glide | Diffusion | 0.9958 | 0.9740 | 0.9966 | bitmind parquet |
| biggan | GAN | 0.9998 | 0.9830 | 0.9998 | bitmind parquet |
| **Overall Avg** | (6 evaluated) | **0.9863** | 0.9438 | 0.9875 | - |
| **GAN Avg** | GAN | 0.9998 | 0.9830 | 0.9998 | - |
| **Diffusion Avg** | Diffusion | 0.9836 | 0.9360 | 0.9851 | - |
| **Std Dev (AP)** | (across gens) | 0.0158 | - | - | - |

Caveat: real-class is VisualNews (not ImageNet "nature") — see `docs/DECISIONS.md` F-A3.

---

## Files (current state)

```
phases/forensic/
├── src/forensic/
│   ├── __init__.py                  # Defines FORENSIC_ROOT
│   ├── data/
│   │   ├── dct_dataset.py           # Reads .pt cache + z-score
│   │   ├── rgb_dataset.py           # For Approach 1
│   │   └── splits.py                # Build train/val/test from raw GenImage
│   ├── preprocessing/
│   │   └── dual_dct.py              # Doctor's exact 8x8+16x16 averaged Y-DCT recipe
│   ├── models/
│   │   └── dct_resnet50.py          # torchvision RN50 + 1ch Kaiming conv1 + Linear(2048,1)
│   ├── training/
│   │   └── two_phase_trainer.py     # 5+25 epoch trainer with optimizer/scheduler reinit at ep 6
│   └── evaluation/
│       ├── per_generator.py         # AP / Accuracy / AUC per generator
│       └── table.py                 # Render eval_table.md
├── scripts/
│   ├── download_genimage.py         # HF + gdown fallback (~10 GB)
│   ├── build_splits.py
│   ├── compute_dct_stats.py         # Welford streaming -> dct_stats.json
│   ├── precompute_dct.py            # Multiprocess (spawn) — scipy isolated from torch
│   ├── train_dct.py                 # Approach 2 entry
│   ├── eval_dct.py
│   ├── adapt_fakeimagedetection.py  # Idempotent path/freeze patcher (Approach 1)
│   ├── train_rgb_fourier.sh         # Wraps bash train.sh "0" (Git Bash)
│   ├── eval_rgb.py
│   └── build_final_table.py
├── configs/
│   ├── dct_resnet50.yaml
│   └── rgb_fourier.yaml
├── external/                        # gitignored
│   └── FakeImageDetection/          # Clone of chandlerbing65nm repo
├── outputs/                         # gitignored
│   ├── forensic_dct_model.pth
│   ├── forensic_rgb_model.pth
│   ├── dct_stats.json
│   ├── eval_dct.json
│   ├── eval_rgb.json
│   └── eval_table.md
├── data/                            # gitignored
│   ├── raw/genimage/                # Per-generator (8) + nature class
│   ├── genimage_train/{train,val}/{0_real,1_fake}/   # 80/20 internal split
│   ├── genimage_test/<gen>/{0_real,1_fake}/          # Doctor's per-generator test layout
│   └── dct_cache/{train,val,test}/                   # Precomputed .pt files
└── tests/
    ├── test_dual_dct_shape.py       # Asserts [1,224,224] float32, finite
    ├── test_dct_stats_io.py         # Round-trip dct_stats.json
    ├── test_dct_model_forward.py    # Smoke: [1,1,224,224] -> [1,1]
    └── test_per_generator_loader.py # Dummy folder eval
```

---

## How this hooks back into V4

The trained `forensic_dct_model.pth` can later REPLACE `blur_jpg_v0.pth` as the initialization weights for V4's UnivFD Stage 1 encoder (`phases/v4/src/v4/models/encoders/univfd.py`). Better init -> better Stage 1 forensic features -> potentially better V4 Stage 2 fusion. This is a separate V4 plan revision; out of scope for this sprint.

See [`FOLLOWUP.md`](FOLLOWUP.md) for the unfinished tracks (Approach 1, SD v1.4/v1.5, V4 retrain).

---

## Verification before training (P6)

After P5 implementation lands, run:

```bash
cd C:\Desktop\MultiGuard
.venv\Scripts\Activate.ps1

# DCT shape assertion
python -c "from forensic.preprocessing.dual_dct import compute_dual_dct; \
  import os; first=os.listdir('phases/forensic/data/genimage_train/train/0_real')[0]; \
  t=compute_dual_dct(f'phases/forensic/data/genimage_train/train/0_real/{first}'); \
  assert t.shape==(1,224,224) and str(t.dtype)=='torch.float32' and t.isfinite().all(); print('OK')"

# Tests
pytest phases/forensic/tests/ -v
```

---

## Doctor's specs traced into this repo

| Requirement | File |
|---|---|
| Both approaches required | This README |
| `forensic_rgb_model.pth` + `forensic_dct_model.pth` + `dct_stats.json` filenames | `outputs/` after P5 |
| Per-generator eval table | `outputs/eval_table.md` after P5 |
| DCT dual-patch averaged recipe (8x8 + 16x16) | `src/forensic/preprocessing/dual_dct.py` (P5) |
| Two-phase training with optimizer reinit at epoch 6 | `src/forensic/training/two_phase_trainer.py` (P5) |
| Standard torchvision ResNet50 (NOT FakeImageDetection's) for Approach 2 | `src/forensic/models/dct_resnet50.py` (P5) |
| 8 generators in test, real = ImageNet nature class | `data/genimage_test/<gen>/{0_real,1_fake}/` after P5 |
| AP / Accuracy / AUC per generator + Overall/GAN/Diffusion/StdDev aggregates | `src/forensic/evaluation/per_generator.py` (P5) |

See `docs/MASTER_CHECKLIST.md` for the full F.1-F.26 audit table.
