# Phase V2 — 3-Class Forensic Fusion (Real / Manipulated / OOC)

**Status:** Historical / superseded by V3.1. Preserved here for reproducibility + reference.

**Source brief:** `docs/doctor-briefs/Implementation Guidelines (V2).pdf`

---

## What V2 was

3-class classifier fusing semantic context (FND-CLIP) + image forensics (DCT + ResNet18) using bidirectional cross-attention. First version to combine multimodal semantic and forensic streams.

**Architecture:**
- Semantic stream: FND-CLIP (frozen, from V1) -> v_semantic [768]
- Forensic stream: ResNet18 on patch-DCT [1,224,224] -> v_forensic [768]
- Auxiliary head (training only): Linear(768, 3) BCE on v_forensic
- Fusion: bidirectional cross-attention (8 heads, 512-d projection, residual+LN both directions, **concatenate** -> [1024])
- Classifier: Linear(1024, 512)+ReLU+Drop(0.5) -> Linear(512, 256)+ReLU+Drop(0.3) -> Linear(256, 3)
- Loss: CE(main) + 0.1*BCE(aux)

**Three trained variants:**
1. `forensic_baseline/` — DCT + ResNet18 only (Step 1)
2. `full_pipeline/` + `full_pipeline_clean/` + `full_pipeline_strict/` — FND-CLIP + forensic cross-attention (Step 2)
3. `clip_forensic/` — CLIP features + forensic cross-attention (Step 3)

---

## Final V2 metrics

From the canonical "gold standard" full pipeline (clean variant, 128k samples):

| Model | Accuracy | F1-macro | AUC-ROC |
|---|---|---|---|
| Forensic baseline (DCT only) | 0.479 | 0.475 | 0.677 |
| CLIP-Forensic | 0.693 | 0.699 | 0.862 |
| **Full pipeline (clean)** | **0.863** | **0.864** | **0.971** |

---

## Files

```
phases/v2/
├── src/
│   ├── train_forensic.py            # Step 1: DCT + ResNet18
│   ├── train_full_pipeline.py       # Step 2: FND-CLIP + forensic + cross-attention
│   ├── train_clip_forensic.py       # Step 3: CLIP variant
│   ├── evaluate_{forensic,full_pipeline,clip_forensic}.py
│   ├── dataset.py                   # V2 dataset loader (3-class)
│   └── models/
│       ├── forensic_baseline.py
│       ├── full_pipeline.py
│       ├── clip_forensic_pipeline.py
│       └── fnd_clip.py              # Mirrored from V1 for V2 self-containment
├── configs/
│   ├── forensic_baseline.yaml
│   ├── full_pipeline.yaml           # Canonical "gold standard"
│   └── clip_forensic.yaml
├── scripts/
│   ├── build_forensic_3class*.py    # 4 variants of dataset builder
│   ├── precompute_dct.py            # V2 DCT preprocessing
│   ├── precompute_fnd_features.py
│   ├── precompute_clip_features.py
│   ├── compile_results.py
│   └── run_phase2_pipeline.sh
└── release_archive/                 # V2 release tag snapshot (scripts/, src/, tests/)

outputs/v2/
├── forensic_baseline/               # Step 1 best.pt + metrics + ROC
├── full_pipeline/, full_pipeline_clean/, full_pipeline_strict/   # Step 2 variants
└── clip_forensic/                   # Step 3 best.pt + metrics + ROC
```

---

## How to re-run

V2 used `scripts/run_phase2_pipeline.sh` to chain dataset build -> precompute -> train -> eval. After the unified mega-repo restructure:

```bash
cd C:\Desktop\MultiGuard
# Activate env: .venv\Scripts\Activate.ps1
bash phases/v2/scripts/run_phase2_pipeline.sh
```

(Expect import path errors from the V2 scripts referencing top-level `src.` from pre-unification layout. Patch as needed; this is archived code.)

---

## Why V2 was superseded

The doctor's V3.1 brief expanded the task to 5 classes (added AI-Text and Fully-Fabricated) and introduced a third stream (Qwen2-7B text forensics). V3.1 also changed the fusion from "concatenate" to "element-wise SUM" of bidirectional cross-attention outputs (V3.1 section 5.3, a deliberate spec change).

For new work, see `phases/v4/` (the modular registry-driven rebuild that implements V3.1 with full reusability).

---

## Cross-phase notes

- **FND-CLIP encoder:** V1's `fnd_clip.py` is mirrored here as a self-contained dependency. The encoder is **frozen** (no training) in V2; only the forensic stream + fusion + classifier are trained.
- **Image data:** V2 used NewsCLIPpings + DGM4 (no MMFakeBench yet — that came in V3 for the new 5-class extension).
- **Caches:** V2's DCT and FND-CLIP feature caches lived at `cache/v2/` originally; they were rebuildable from scripts and are not preserved in git. Forensic baseline's per-image DCT shape was the older `[1, 224, 224]` single-set (V3 introduced the dual 8x8 + 16x16 averaged variant).
