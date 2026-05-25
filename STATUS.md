# MultiGuard — Status (live, top-level)

> Single entry-point for the doctor + future contributors. Mirrors current
> on-disk state; manually refreshed as work lands.

---

## TL;DR

**Forensic image detector** (current sprint deliverable, doctor's brief
`docs/doctor-briefs/Forensic_Image_Detector_En.pdf`):

| Approach | Status | Best val AP | Overall test AP | Doc |
|----------|--------|-------------|-----------------|-----|
| **2 (DCT)** | OK Shipped | 0.9848 @ ep 13 | **0.9863** (6/8 gens) | `phases/forensic/REPORT.md` + `REPORT.docx` |
| **1 (RGB + Fourier)** | Training | 0.9968+ @ ep 5 | TBD | `phases/forensic/REPORT_v2.md` (TBD) |

**V4 multimodal pipeline** (5-class detector, V3.1 spec):
- Code + tests + V4 docs all on disk; CI green.
- `dct_forensic_v1` encoder wrapper added - `forensic_dct_model.pth` is now
  a drop-in Stage-1 init for V4 (replaces `blur_jpg_v0.pth`).
- Stage 0 / Stage 2 retrain BLOCKED on getting the 155 GB datasets onto
  the multiGuard dual-4090 PC (see `docs/MULTIGUARD_SETUP.md` step 7).

---

## What's where

| Artifact | Location |
|----------|----------|
| Doctor handoff (Approach 2) | `phases/forensic/REPORT.md`, `phases/forensic/REPORT.docx` |
| Per-generator eval table | `phases/forensic/outputs/eval_table.md` (Approach 2) |
| Training curves PNG | `phases/forensic/outputs/training_curves.png` (Approach 2) |
| Approach 2 model | `phases/forensic/outputs/dct/forensic_dct_model.pth` (282 MB, local only) |
| Approach 2 model (mirror) | https://huggingface.co/FerasMad/forensic-dct-v1 |
| Approach 1 model | `phases/forensic/outputs/rgb/forensic_rgb_model.pth` (training in progress) |
| Spec compliance table | `docs/MASTER_CHECKLIST.md` (F.1-F.26 status per doctor's brief) |
| Design decisions | `docs/DECISIONS.md` (U, D, F-A series) |
| Followup plan | `phases/forensic/FOLLOWUP.md` |
| multiGuard PC setup | `docs/MULTIGUARD_SETUP.md` |

---

## Live progress (current 12-hour autonomous window)

| Task | Status | Notes |
|------|--------|-------|
| Approach 1 trainer + eval scripts | OK Committed `ce46614` | `train_rgb_fourier.py`, `eval_rgb.py`. Smoke clean. |
| Approach 1 training | Running | ep 1-6 done, val_ap 0.992-0.997 |
| V4 forensic encoder wrapper | OK Committed `ce46614` | `dct_forensic_v1` registered + 9 tests pass |
| FakeImageDetection checkpoint | OK Downloaded via gdown | `rn50ft_spectralmask.pth` (renamed from spec's `fouriermask`; F-A8) |
| SD v1.4/v1.5 mirror hunt | OK Exhausted | No HF mirrors with actual data; only Drive remains |
| Combined eval table builder | OK Written | `build_combined_eval_table.py` |
| Approach 1 eval | Pending training finish | |
| Approach 1 REPORT update | Pending eval | |
| V4 Stage 0 / Stage 2 retrain on FSOS | Deferred | Single-4070 too slow; dual-4090 needs 155 GB data transfer |

---

## Spec compliance summary (F.1-F.26)

| ID | Requirement | Status |
|----|-------------|--------|
| F.1-F.3 | Data structure, 8 generators, real class | Partial (6/8 gens, VisualNews-as-nature substitute) |
| F.4-F.11 | Approach 1 (RGB + Fourier) | Training in progress |
| F.12-F.22 | Approach 2 (DCT) | OK Done |
| F.23-F.26 | Evaluation (per-gen + aggregates) | OK Done for Approach 2; pending A1 |

See `docs/MASTER_CHECKLIST.md` for the per-row detail with deviation notes.

---

## How to run any of this

See `docs/MULTIGUARD_SETUP.md` for a copy-paste recipe. Quick local re-run:

```bash
source .venv/Scripts/activate  # Python 3.12, torch 2.6.0+cu124

# Approach 2 (already done; verify by loading the ckpt)
python -c "import torch; print(torch.load('phases/forensic/outputs/dct/forensic_dct_model.pth', weights_only=False)['val_ap'])"
# -> 0.9848...

# Approach 1 (after training finishes)
python phases/forensic/scripts/eval_rgb.py \
    --ckpt phases/forensic/outputs/rgb/forensic_rgb_model.pth

# Combined F.25 table
python phases/forensic/scripts/build_combined_eval_table.py
```

---

## Open follow-ups

1. SD v1.4 / SD v1.5 generators (`phases/forensic/FOLLOWUP.md` A) - needs official
   GenImage Drive (browser-side).
2. V4 retrain on dual-4090 (`docs/MULTIGUARD_SETUP.md` step 7) - needs 1 TB USB
   SSD for 155 GB dataset transfer.
3. ImageNet "nature" validation (FOLLOWUP C) - only if domain bias is a concern.

---

_For the latest commit: `git log -1 --oneline`._
