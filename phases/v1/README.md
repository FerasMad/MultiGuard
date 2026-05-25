# Phase V1 — FND-CLIP Binary OOC Detector

**Status:** Historical / superseded by V2 and V3.1. Preserved here for reproducibility + reference.

**Reference paper:** Zhou et al., "Multimodal Fake News Detection via CLIP-Guided Learning", IEEE ICME 2023.

---

## What V1 was

Binary classifier: **Out-of-Context (OOC) vs not-OOC** on NewsCLIPpings. Used the published FND-CLIP architecture as a reference baseline.

**Architecture:**
- Visual encoder: ResNet-50 (ImageNet pretrained, fine-tuned) → 2048-d
- Text encoder: BERT-base-uncased [CLS] → 768-d
- CLIP cross-modal: paired image + text encoders → 512-d shared semantic space
- Modality attention: 3 scalar scores (text, image, CLIP-fused) for adaptive fusion
- Classifier: 2-layer MLP with sigmoid → binary

**Source brief:** `docs/doctor-briefs/Dataset + V1 (instructions).pdf`

---

## Final V1 metrics (canonical)

From `outputs/v1/results/V1_OOC_FINAL.md`:

| Metric | Value |
|---|---|
| Accuracy | 0.946 |
| F1 (macro) | 0.945 |
| AUC | 0.987 |

Per-class breakdown + leakage-free retrain variants live alongside in `outputs/v1/results/`.

---

## Files

```
phases/v1/
├── src/
│   ├── train.py                 # Binary OOC training loop
│   ├── evaluate.py              # Binary OOC evaluation
│   ├── evaluate_transfer.py     # Transfer eval (held-out OOC sources)
│   ├── visualnews_resolver.py   # image_id -> on-disk JPG path resolver
│   └── models/fnd_clip.py       # FND-CLIP encoder (also used as frozen module in V2/V3)
└── configs/v1_ooc.yaml          # V1 hyperparameters + dataset config

outputs/v1/
└── results/                      # V1_OOC_FINAL.md + per-fold metrics + diagrams
```

---

## How to re-run (historical reproduction)

V1 was originally run from the old repo root using `src.train`. After the mega-repo restructure, the import paths inside these scripts may need updating to match the new locations.

If you need to reproduce V1:
```bash
cd C:\Desktop\MultiGuard
# Activate env: .venv\Scripts\Activate.ps1
python -m phases.v1.src.train --config phases/v1/configs/v1_ooc.yaml
```

(Expect import path errors — the V1 scripts reference top-level `src.` paths from the pre-unification layout. Fix as needed; this is archived code, not the active codebase.)

---

## Why V1 was superseded

- V1 was binary only (OOC vs not-OOC). The doctor's V2 brief expanded to 3 classes (Real / Manipulated / OOC) and added a forensic image stream.
- V3.1 expanded further to 5 classes (Real / OOC / Manipulated / AI-Text / Fully-Fabricated) with a third text-forensic stream (Qwen2-7B).
- The **active codebase** is `phases/v4/` (the modular registry-driven rebuild). For new work, start there — V1/V2/V3 are reference only.

---

## V1 in the unified mega-repo

V1's `fnd_clip.py` is also a runtime dependency for V2's full pipeline and V3's fusion module (both consume FND-CLIP semantic embeddings as a frozen encoder). The canonical copy of the encoder in V1's directory is mirrored to `phases/v2/` and `phases/v3/` during their respective phase migrations so each phase is self-contained.

V4 ships its own `fnd_clip.py` at `phases/v4/src/v4/models/encoders/fnd_clip.py` (with the `sem_proj` 512->768 adapter, deviation F1) and does not depend on V1's copy.
