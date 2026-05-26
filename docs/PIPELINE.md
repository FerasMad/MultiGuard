# MultiGuard — pipeline (doctor handoff)

This page is the single map between the architecture diagram the doctor
sketched and the code in this repository. **The fact-checking module shown
in the broader `Algorithms.html` figures is intentionally out of scope for
this sprint** — the deliverable is the 3-branch pipeline from the V3.1
Implementation Guidelines.

---

## Architecture (3 branches → fusion → 5 classes)

```
                News text                       News image
                  │  │                             │  │
          ┌───────┘  └────────┐           ┌────────┘  └────────┐
          │                   │           │                    │
          ▼                   ▼           ▼                    ▼
   ┌─────────────┐    ┌────────────────────┐         ┌─────────────────┐
   │ Text        │    │ Cross-Modal        │         │ Image           │
   │ Credibility │    │ Consistency        │         │ Authenticity    │
   │ Branch      │    │ (FND-CLIP V1)      │         │ Branch          │
   │             │    │                    │         │                 │
   │ Qwen2-7B    │    │ ResNet50 + BERT +  │         │ Dual-DCT prepr. │
   │ masked-mean │    │ CLIP +             │         │ + 1-ch ResNet50 │
   │ → 3584-d    │    │ modality attention │         │ → 768-d         │
   └──────┬──────┘    │ → 512-d            │         └────────┬────────┘
          │           └─────────┬──────────┘                  │
          │                     │                             │
          └─────────┬───────────┴──────────────────┬──────────┘
                    ▼                              ▼
            ┌─────────────────────────────────────────────┐
            │ Attention Fusion Layer (V3PairwiseFusion)   │
            │   per-stream LayerNorm                      │
            │   3 × bidirectional MHA (8 heads)           │
            │   element-wise SUM (per V3.1 §5.3)          │
            │   Conv1d(768→768, k=3) + GELU               │
            │   Conv1d(768→1024, k=1) + GELU              │
            │   AdaptiveAvgPool1d(1) → [B, 1024]          │
            └──────────────────────┬──────────────────────┘
                                   ▼
            ┌─────────────────────────────────────────────┐
            │ MLP classifier (V3.1 §6 — exact spec)       │
            │   Linear(1024, 512) + BN + GELU + Drop(0.5) │
            │   Linear(512, 256) + GELU                   │
            │   Linear(256, 5)                            │
            └──────────────────────┬──────────────────────┘
                                   ▼
                              softmax → 5 probabilities
                                   │
   ┌─────────────────────────────────────────────────────────────────┐
   │  Real news │ Out-of-context │ Real-text +  │ Fake-text + │ Fully │
   │            │                │ Fake-image   │ Real-image  │ fake  │
   └─────────────────────────────────────────────────────────────────┘
```

---

## Box-to-file mapping

| Doctor's box | V3.1 spec | Code path | What it does |
|---|---|---|---|
| **Text Credibility Branch** | §1 / §4 | `phases/v4/src/v4/models/encoders/qwen_text.py` (`Qwen2TextEncoder`, registry key `qwen2_7b`) | Qwen2-7B-Instruct → last-layer hidden state → masked-mean pool → 3584-d. The fusion's `text_proj` (Linear 3584→768 + GELU) projects to 768-d. |
| **Image Authenticity Branch** | §1 / §3 (UnivFD → replaced by Forensic Approach 2) | `phases/v4/src/v4/models/encoders/dct_forensic.py` (`DctForensicEncoder`, registry key `dct_forensic_v1`) | Dual-patch DCT (`phases/forensic/src/forensic/preprocessing/dual_dct.py`) → 1-channel ResNet50 (`phases/forensic/src/forensic/models/dct_resnet50.py`) → 2048-d → Linear(2048, 768)+GELU. Weights from `phases/forensic/outputs/dct/forensic_dct_model.pth`. |
| **Cross-Modal Consistency (FND-CLIP)** | §1 / §2 | `phases/v4/src/v4/models/encoders/fnd_clip.py` (`FNDCLIPSemanticEncoder`, registry key `fnd_clip`) | ResNet50 visual + BERT-base text + CLIP-ViT-base CLIP-cross-modal + modality-wise attention → 512-d. The fusion's `sem_proj` (Linear 512→768 + GELU) projects to 768-d. Weights from V1 `outputs/v1/leakfree/best.pt`. |
| **Attention Fusion Layer** | §5 | `phases/v4/src/v4/models/fusion/v3_pairwise.py` (`V3PairwiseFusion`, registry key `v3_pairwise`) | Per-stream LayerNorm → 3 PairwiseCrossAttention blocks (8-head MHA, SUM not concat) → stack→permute→Conv1d→AdaptiveAvgPool → 1024-d. Plus §5.5 aux head (`Linear(768, 2)` on `v_imgfor.detach()`) and per-P5.10a optional `proj_dims` for raw-cache compatibility. |
| **MLP classifier** | §6 (exact) | `phases/v4/src/v4/models/classifier/mlp_head.py` (`MLPClassifier`) | Linear(1024, 512) + BatchNorm + GELU + Dropout(0.5) → Linear(512, 256) + GELU → Linear(256, 5). Strict V3.1 §6, no flexibility hooks. |
| **5 output labels** | §2 | `phases/v4/src/v4/core/class_map.py` | `{0: "Real", 1: "Out-of-Context", 2: "Manipulated", 3: "AI-Text", 4: "Fully-Fabricated"}` — semantically matches the doctor's `Real news / OOC / Real-text+Fake-image / Fake-text+Real-image / Fully-fake`. |

### Out of scope (explicitly)

The `Algorithms.html` Figure 4 shows a **Fact-Checking Module** (Google Fact Check API → Sentence-Transformer → NLI verification → veracity score). Per the user directive on 2026-05-26, this branch is **not** in the current ship. The V3.1 Implementation Guidelines specify only the three branches above, and that spec is the deliverable.

---

## Trained checkpoint

| Item | Value |
|---|---|
| Config | `phases/v4/configs/v4_pipeline_dctforensic_v3caches.yaml` |
| Seed | 42 (single-seed; multi-seed deferred to dual-4090 PC) |
| Best epoch | 4 (early-stopped at epoch 14, patience=10) |
| Checkpoint on disk | `outputs/v4/stage2_fusion_dctforensic/best.pt` |
| Checkpoint on HF Hub | [`FerasMad/multiguard-v4-fusion`](https://huggingface.co/FerasMad/multiguard-v4-fusion) |

### Headline numbers (FSOS RTX 4070)

| Split | F1-macro | vs V3 baseline |
|---|---|---|
| Validation (in-distribution) | **0.7334** | V3: 0.7215 (+1.19 pp) |
| Test (in-distribution) | **0.7267** | First V3.1 §7 5-class test for any of our checkpoints |
| MMFakeBench transfer | **0.4805** | V3: 0.3832 (+9.7 pp) |

### Per-class test F1

| Class | Display | F1 | Notes |
|---|---|---|---|
| 0 | Real news | 0.41 | Confused with Out-of-context (column 1: 246 of 495) |
| 1 | Out-of-context | 0.47 | Confused with Real news (178 of 495) |
| 2 | Real text + Fake image | 0.76 | DGM4 + MMFakeBench manipulated subsets |
| 3 | Fake text + Real image | **0.997** | V3 caption-shortcut inherited via reused V3 text cache |
| 4 | Fully fake | **0.998** | Same shortcut — flagged as honest finding |

The near-perfect F1 on classes 3 and 4 reflects a known caption-fingerprint shortcut from V3's text cache that we did not regenerate; removing it requires BLIP-2 caption rewrite + a fresh `v_textfor` cache (logged as a follow-up in `STATUS.md`).

---

## Forensic image detectors (companion deliverable)

The Forensic brief (`docs/doctor-briefs/Forensic_Image_Detector_En.pdf`) asks for two binary AI-image detectors. Both are shipped:

| Approach | Code path | Eval (6 generators, AP) | HF Hub |
|---|---|---|---|
| **Approach 1 — RGB + Fourier mask** | `phases/forensic/scripts/train_rgb_fourier.py` + `phases/forensic/external/FakeImageDetection/` | Overall **0.9979**, StdDev 0.0023 | [`FerasMad/forensic-rgb-v1`](https://huggingface.co/FerasMad/forensic-rgb-v1) |
| **Approach 2 — Dual-DCT** | `phases/forensic/src/forensic/models/dct_resnet50.py` + `dual_dct.py` | Overall **0.9863**, StdDev 0.0158 | [`FerasMad/forensic-dct-v1`](https://huggingface.co/FerasMad/forensic-dct-v1) |

Full per-generator + transfer probe numbers in `phases/forensic/REPORT.md` and `phases/forensic/REPORT.docx`. Trained on 6 of 8 generators (SD v1.4 / SD v1.5 mirrors deferred — only browser-only Drive remains).

---

## Live demo

| Link | What's there |
|---|---|
| HF Space (Gradio UI) | https://huggingface.co/spaces/FerasMad/multiguard-demo |
| Source code | https://github.com/FerasMad/MultiGuard |
| Models | https://huggingface.co/FerasMad |

The Space has three tabs:
- **Forensic A1 (RGB + Fourier)** — works on CPU, instant.
- **Forensic A2 (Dual-DCT)** — works on CPU, instant.
- **5-class fake-news (V4)** — requires GPU (ZeroGPU or A10G). On CPU-basic Spaces (default free tier) this tab gracefully degrades to a static readout of the offline eval numbers. To enable live inference, upgrade the Space hardware in its Settings page.

---

## Reproducing the eval

```bash
# Activate venv (Python 3.12, torch 2.6.0+cu124)
source .venv/Scripts/activate

# V4 test split — reproduces F1-macro 0.7267
python -m v4 eval \
    --config phases/v4/configs/v4_pipeline_dctforensic_v3caches.yaml \
    --split test

# V4 MMFakeBench transfer probe — reproduces F1-macro 0.4805
python -m v4 eval \
    --config phases/v4/configs/v4_pipeline_dctforensic_v3caches.yaml \
    --split mmfakebench-transfer

# Forensic A2 per-generator eval — reproduces Overall AP 0.9863
python phases/forensic/scripts/eval_dct.py

# Forensic A1 per-generator eval — reproduces Overall AP 0.9979
python phases/forensic/scripts/eval_rgb.py
```

Frozen artifact copies of the V4 eval (metrics JSON + classification reports + confusion-matrix PNGs + per-source breakdowns + training history CSV) live in `phases/v4/docs/eval/seed42/`.

---

## Spec compliance summary

| ID | Requirement (V3.1) | Status |
|---|---|---|
| V3.1 5-class output | `class_map.py` matches | OK |
| §1 three encoders (semantic, image-forensic, text-forensic) | all implemented | OK |
| §5 fusion (LayerNorm per signal + 3-pair bidirectional MHA-SUM + Conv1d stack + AdaptiveAvgPool) | implemented in `v3_pairwise.py` + spec test | OK |
| §5.5 aux head | `Linear(768, 2)` on `v_imgfor.detach()` + 0.1 × BCE | OK |
| §6 classifier (exact: 1024→512+BN+GELU+Drop(0.5)→256+GELU→5) | `mlp_head.py` | OK |
| §5.5 training (AdamW lr=1e-4 wd=1e-4, batch=64, StepLR ×0.1 @ 30, ES patience=10, grad clip 1.0) | trained config in `v4_pipeline_dctforensic_v3caches.yaml` | OK |
| §7 eval (P/R/F1-macro/CM + MMFakeBench transfer) | implemented + reproduced | OK |
| §7 aux head deactivated at inference | softmax on `main_logits` only | OK |

See `docs/MASTER_CHECKLIST.md` for the full F.1–F.26 forensic + V1/V2/V3.1 audit, and `docs/DECISIONS.md` for every locked design decision (D-series + F-A deviation register).
