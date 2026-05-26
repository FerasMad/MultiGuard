# MultiGuard — Status (live, top-level)

> Single entry-point for the doctor + future contributors. Mirrors current
> on-disk state; manually refreshed as work lands.

## Live demo (HF Spaces)

🔗 **https://huggingface.co/spaces/FerasMad/multiguard-demo**

Three tabs in one Gradio app: Forensic A1 (RGB+Fourier), Forensic A2 (Dual-DCT), and
V4 5-class fake-news. Forensic tabs work on CPU. The V4 tab requires GPU; on CPU-basic
it gracefully degrades to the static eval numbers from `phases/v4/docs/eval/seed42/`.

For the box-to-file mapping of the doctor's architecture diagram see
[`docs/PIPELINE.md`](docs/PIPELINE.md).

---

## TL;DR

**Forensic image detector** (current sprint deliverable, doctor's brief
`docs/doctor-briefs/Forensic_Image_Detector_En.pdf`):

| Approach | Status | Best val AP | Overall test AP | StdDev AP | Doc |
|----------|--------|-------------|-----------------|-----------|-----|
| **2 (DCT)** | OK Shipped | 0.9848 @ ep 13 | 0.9863 (6/8 gens) | 0.0158 | unified `phases/forensic/REPORT.md` |
| **1 (RGB + Fourier)** | OK Shipped | **0.9971 @ ep 10** | **0.9979 (6/8 gens)** | **0.0023** | (same) |

Approach 1 beats Approach 2 by ~1.2 pp on Overall AP and ~7x lower variance
across generators (StdDev 0.0023 vs 0.0158). Both approaches pass the
doctor's spec; A1 is more consistent across diffusion generators, A2 is more
interpretable (frequency-domain input).

### External validation — MMFakeBench transfer probe

Both detectors evaluated on `MMFakeBench_val` (700 fake + 300 real images):

| Detector | Overall AP | Overall AUC | Notes |
|----------|-----------|-------------|-------|
| Approach 1 (RGB+Fourier) | 0.6975 | 0.499 | Domain bias visible |
| Approach 2 (DCT)         | 0.7166 | 0.514 | Same pattern |

**Honest finding:** Both detectors struggle to transfer out of distribution.
Mean p(fake) on truly AI-generated subcategories (`antifact_image_generation`,
`fever_AI`, `llm_*_generation`) is 0.00–0.17 — they don't fire strongly on
AI images from different generation pipelines than GenImage_v2 trained on.

This matches the F-A3 caveat about VisualNews-as-nature: the detectors
likely learned a partial "VisualNews real photo vs GenImage AI image"
boundary instead of universal forensic features. For per-generator AP
inside the GenImage_v2 distribution they're excellent; for novel AI
generators the score collapses. See
`phases/forensic/outputs/mmfakebench_transfer{,_dct}.json` for per-category
breakdowns.

**Mitigation path:** retrain with ImageNet "nature" + a wider AI-source
mix (FOLLOWUP C). Recommended for next iteration before the model is
considered for live deployment.

**V4 multimodal pipeline** (5-class detector, V3.1 spec):
- Code + tests + V4 docs all on disk; CI green.
- `dct_forensic_v1` encoder wrapper added - `forensic_dct_model.pth` is now
  a drop-in Stage-1 init for V4 (replaces `blur_jpg_v0.pth`).
- **V4 Stage 2 retrained on FSOS** using:
  - new `v_imgfor_dctforensic` cache (16500 shards, built from `forensic_dct_model.pth`),
  - V3 raw `v_semantic` (512) and `v_textfor_qwen` (3584) caches reused via inline
    `proj_dims` projection heads in `V3PairwiseFusion` (P5.10a).

  | Metric | V4 (this run) | V3 baseline |
  |---|---|---|
  | Val F1-macro | **0.7334** @ ep 4 | 0.7215 |
  | Test F1-macro | **0.7267** | (V3 5-class test was never re-evaluated under §7) |
  | MMFakeBench transfer F1-macro | **0.4805** | 0.3832 |

  Trained from `phases/v4/configs/v4_pipeline_dctforensic_v3caches.yaml`,
  output `outputs/v4/stage2_fusion_dctforensic/best.pt`.
  Early-stopped at epoch 14 (patience=10 from ep 4). +1.19 pp val vs V3,
  +0.4805 vs 0.3832 (+9.7 pp) on transfer. Class 4 caption shortcut
  persists via reused V3 text cache (per-class F1: cls3=0.997, cls4=0.998).
- 3-seed runs (1337, 2024) and Stage 0 fresh fine-tune still deferred —
  single-4070 + cache regen cost; not needed for current ship.

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

## Live progress (12-hour autonomous window)

| Task | Status | Notes |
|------|--------|-------|
| Approach 1 trainer + eval scripts | OK Committed `ce46614` | `train_rgb_fourier.py`, `eval_rgb.py`. Smoke clean. |
| Approach 1 training | OK Done | Best val_AP 0.9971 @ ep 10, early-stopped @ ep 15 (1271 s) |
| Approach 1 per-generator eval | OK Done | Overall AP 0.9979, StdDev 0.0023 |
| V4 forensic encoder wrapper | OK Committed `ce46614` | `dct_forensic_v1` registered + 9 tests pass |
| FakeImageDetection checkpoint | OK Downloaded via gdown | `rn50ft_spectralmask.pth` (renamed from spec's `fouriermask`; F-A8) |
| SD v1.4/v1.5 mirror hunt | OK Exhausted | No HF mirrors with actual data; only Drive remains |
| Combined eval table builder | OK Written | `build_combined_eval_table.py` |
| Approach 1 REPORT update | OK Done | REPORT.md / REPORT.docx now cover both approaches |
| Approach 1 .pth on HF Hub | OK Uploaded | https://huggingface.co/FerasMad/forensic-rgb-v1 |
| V4 cache regen (v_imgfor with dct_forensic_v1) | OK Done | 16500 shards @ ~155s on RTX 4070 (P5.10b) |
| V4 Stage 2 retrain (seed=42) | OK Done | val_F1=0.7334 / test_F1=0.7267 / transfer_F1=0.4805 — P5.11a |
| V4 Stage 0 fresh + seeds 1337/2024 | Deferred | Single-4070; not needed for current ship — see open follow-ups |

---

## Spec compliance summary (F.1-F.26)

| ID | Requirement | Status |
|----|-------------|--------|
| F.1-F.3 | Data structure, 8 generators, real class | Partial (6/8 gens, VisualNews-as-nature substitute) |
| F.4-F.11 | Approach 1 (RGB + Fourier) | OK Done (spectralmask rename per F-A8) |
| F.12-F.22 | Approach 2 (DCT) | OK Done |
| F.23-F.26 | Evaluation (per-gen + aggregates) | OK Done for both approaches |

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
2. V4 multi-seed (1337, 2024) + fresh Stage 0 (FND-CLIP fine-tune) on dual-4090
   - needs the 155 GB dataset transfer; current single-seed FSOS run is the ship.
3. ImageNet "nature" validation (FOLLOWUP C) - only if domain bias is a concern.
4. V4 Class 4 / 3 caption-shortcut: dropped by reusing V3 text cache as-is.
   Re-cache `v_textfor` over `forensic_5class_unified.csv` after BLIP-2 caption
   rewrite once we re-stage VisualNews on a real GPU (deferred to next sprint).

---

_For the latest commit: `git log -1 --oneline`._
