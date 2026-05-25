# MultiGuard — Unified Mega-Repo

Multimodal fake-news detection research project. This repo consolidates **every phase** of the work (V1 -> V2 -> V3.1 -> V4) into one tree plus the active **Forensic Image Detector** sprint, all under a single canonical structure organized by phase.

Owner: Feras Madkhali · KSU graduation project · Doctor-supervised.

---

## Repo layout

```
MultiGuard/
├── README.md                 # ← you are here
├── pyproject.toml            # Canonical Python config (V4-derived)
├── Makefile                  # `make data | precompute | train | eval | server`
├── .pre-commit-config.yaml   # ruff + standard hooks
├── .github/workflows/ci.yml  # GitHub Actions CI
├── CHANGELOG.md              # V1 -> V2 -> V3 -> V4 timeline
│
├── docs/                     # All cross-phase documentation
│   ├── doctor-briefs/        # 5 doctor PDFs (V1, V2, V3.1, Forensic, IMAGE_ONLY_METRICS)
│   ├── MASTER_CHECKLIST.md   # Audit checklist synthesized from all PDFs (P3 deliverable)
│   ├── ARCHITECTURE.md / SETUP.md / CONTRIBUTING.md
│   ├── ADD_ENCODER.md / ADD_DATASET.md / ADD_FUSION.md   # V4 flexibility guides
│   ├── EVAL_PROTOCOL.md / SPEC_COMPLIANCE_MAP.md / DECISIONS.md
│   ├── PHASE3_ARCHITECTURE.md   # V3 dataflow (preserved from old repo)
│   └── V4_PLAN.md           # V4 rebuild plan
│
├── phases/                   # Phase subfolders (each is self-contained)
│   ├── v1/  ← FND-CLIP binary OOC (Zhou et al. ICME 2023). README + src + configs
│   ├── v2/  ← 3-class forensic fusion. README + src + configs + scripts + release_archive
│   ├── v3/  ← 5-class Qwen pipeline (V3.1 spec, revs 7-12). README + src + scripts + .venv-qwen
│   ├── v4/  ← Active modular rebuild. README + src/v4/ + configs + tests
│   └── forensic/  ← Current sprint: binary AI-image detector (DCT + RGB+Fourier). README + src + scripts + configs + tests
│
├── shared/                   # Cross-phase utilities (legacy code preserved here)
│   └── legacy_utils/         # corruption/, generation/, preprocess/, scraping/, utils/ from old src/
│
├── data/                     # Raw + processed datasets (gitignored)
│   ├── raw/                  # DGM4, NewsCLIPpings, VisualNews, MMFakeBench, GenImage
│   └── processed/            # Built manifests
│
├── cache/                    # Feature caches (gitignored)
│   ├── v3/_features/         # V3 precomputed v_semantic, v_imgfor, v_textfor (6.9 GB)
│   ├── v4/                   # V4 per-modality .pt shards
│   └── forensic/             # Forensic DCT cache
│
├── outputs/                  # Training outputs per phase (mostly gitignored)
│   ├── v1/  v2/  v3/  v4/  forensic/  archive/
│
├── app/                      # FastAPI server
│   ├── server.py             # V4 modular server (canonical, port 8081)
│   ├── server_v3_legacy.py   # V3.1 5-class server (port 8080, preserved)
│   ├── server_config.yaml    # V4 config
│   ├── samples/              # Demo images for UI
│   └── static/               # Bilingual frontend
│
├── scripts/                  # Top-level utility scripts (cross-phase)
│   ├── download_data.py      # MMFakeBench via HF
│   ├── download_pretrained.py # blur_jpg_prob0.pth via CNNDetection
│   ├── prepare_image_subset.py
│   └── precompute.py         # V4 feature cache builder
│
├── tests/                    # Cross-phase integration tests
└── news_clippings/           # NewsCLIPpings README + download script (annotations live in data/raw/)
```

---

## Quick start (active V4 + forensic work)

```bash
git clone https://github.com/FerasMad/MultiGuard.git
cd MultiGuard
# Set up Python 3.12 venv:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[gpu,server,dev]"

# Stage raw datasets at data/raw/ (see docs/SETUP.md step 6)
make data         # build manifests + leakage audit
make precompute   # feature caches
make train        # Stage 0 -> Stage 1 -> Stage 2 (3 seeds)
make eval         # V3.1 section 7 + MMFakeBench transfer
make server       # FastAPI on port 8081
```

For the forensic detector sprint, see `phases/forensic/README.md`.

---

## Phase status

| Phase | Status | Canonical metrics | Where to look |
|-------|--------|-------------------|---------------|
| **V1** | Frozen / superseded | Val acc 0.946, F1 0.945, AUC 0.987 | `phases/v1/README.md` |
| **V2** | Frozen / superseded | Test acc 0.863, F1 0.864, AUC 0.971 (full pipeline clean) | `phases/v2/README.md` |
| **V3.1** | Reference baseline | Val F1 0.7215, AUC 0.9305; transfer F1 0.3832 | `phases/v3/README.md` |
| **V4** | Active / canonical for new work | Pending Stage 0 + Stage 1 + Stage 2 trains | `phases/v4/README.md` |
| **Forensic** | Current sprint | Pending — see plan + doctor brief | `phases/forensic/README.md` |

---

## Documentation entry points

- **What is V4 and how to extend it:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/ADD_ENCODER.md`](docs/ADD_ENCODER.md), [`docs/ADD_DATASET.md`](docs/ADD_DATASET.md), [`docs/ADD_FUSION.md`](docs/ADD_FUSION.md)
- **How to install + run on Windows:** [`docs/SETUP.md`](docs/SETUP.md), [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) (MANDATORY Windows workarounds)
- **V4 plan + V3.1 spec compliance:** [`docs/V4_PLAN.md`](docs/V4_PLAN.md), [`docs/SPEC_COMPLIANCE_MAP.md`](docs/SPEC_COMPLIANCE_MAP.md)
- **Evaluation protocol:** [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)
- **Doctor's briefs:** [`docs/doctor-briefs/`](docs/doctor-briefs/) — 5 PDFs (V1, V2, V3.1, Forensic Image Detector, IMAGE_ONLY_METRICS)
- **Master requirements checklist:** [`docs/MASTER_CHECKLIST.md`](docs/MASTER_CHECKLIST.md) — audit table synthesized from all 5 PDFs

---

## Reusability mandate (the doctor's brief)

The doctor's reusability mandate: *"build the general structure for the next people who are going to work on this project"*. Concretely:

1. **Swap an encoder in ~80 LOC** — see `docs/ADD_ENCODER.md`
2. **Swap a dataset source in ~100 LOC** — see `docs/ADD_DATASET.md`
3. **Swap a fusion module** — see `docs/ADD_FUSION.md`
4. **Reproduce any past run from saved provenance** — see `docs/DECISIONS.md` D14
5. **Change num_classes** — edit `phases/v4/src/v4/core/class_map.py` + `mlp_head.py`
6. **YAML-driven training** — every encoder/fusion/loss/scheduler parametrized in `phases/v4/configs/*.yaml`
7. **Registry pattern** — `phases/v4/src/v4/core/registry.py` for ENCODER/FUSION/DATASET registries

V1/V2/V3 phases are preserved as **frozen historical reference**. New work happens in V4 and in `phases/forensic/`.

---

## Git history

- Pre-unification snapshot: tag `pre-unification-old` (on old `Multimodal-fake-news-detection` remote) + `pre-unification-v4` (on FerasMad/MultiGuard).
- Legacy V3 history preserved on branch `legacy-multimodal` (after P4 git unification).
- Canonical going-forward remote: `github.com/FerasMad/MultiGuard`.

---

## License

MIT (see `pyproject.toml`).

**3-class task:** Real | Manipulated (face-swap/attribute) | Out-of-Context (OOC)

## Project structure

```
config/             YAML configs for each experiment
scripts/            Data building, feature precomputation, evaluation helpers
src/
  models/           Model definitions (FND-CLIP, forensic baseline, full pipeline, CLIP-forensic)
  train*.py         Training scripts
  evaluate*.py      Evaluation scripts
  dataset.py        Dataset / dataloader
tests/              Smoke and invariant tests
outputs/            Experiment results (metrics, ROC curves, training history)
```

## Phases

### Phase 1 -- FND-CLIP binary OOC detector

Binary classifier: OOC vs not-OOC.  
Model: FND-CLIP (Zhou et al., ICME 2023) -- ResNet-50 + BERT + frozen CLIP with modality attention.

| Metric | Value |
|--------|-------|
| Accuracy | 0.946 |
| F1 | 0.945 |
| AUC | 0.987 |

Config: `config/v1_ooc.yaml`  
Results: `outputs/v1_results/`

### Phase 2 -- 3-class forensic fusion

Extends Phase 1 with pixel-level forensic features (DCT + ResNet18) fused
with semantic features via bidirectional cross-attention.

**Step 1: Forensic baseline** (DCT + ResNet18 alone)  
Config: `config/forensic_baseline.yaml`

**Step 2: Full fusion pipeline** (FND-CLIP semantic + forensic cross-attention)  
Config: `config/full_pipeline.yaml`

**CLIP-Forensic variant** (CLIP features + forensic cross-attention)  
Config: `config/clip_forensic.yaml`

#### Results (in-distribution test set, n=2,700)

| Model | Accuracy | F1-macro | AUC-ROC |
|-------|----------|----------|---------|
| Forensic baseline | 0.479 | 0.475 | 0.677 |
| CLIP-Forensic | 0.693 | 0.699 | 0.862 |
| **Full pipeline (clean)** | **0.863** | **0.864** | **0.971** |

#### Transfer (MMFakeBench, zero-shot)

| Model | Accuracy | F1-macro |
|-------|----------|---------|
| Forensic baseline | 0.175 | 0.173 |
| CLIP-Forensic | 0.543 | 0.289 |
| **Full pipeline (clean)** | **0.643** | **0.308** |

The "clean" pipeline uses leak-free FND-CLIP features and uniform JPEG
normalization. See `HANDOFF.md` for details on the data-leak fix.

## Quick start

```bash
pip install -r requirements.txt

# Phase 1
python scripts/build_balanced_dataset.py
python -m src.train --config config/v1_ooc.yaml
python scripts/fndclip_ooc_test_eval.py

# Phase 2 (full pipeline, end-to-end)
bash scripts/run_phase2_pipeline.sh
# or step by step:
python scripts/build_forensic_3class_clean.py
python scripts/precompute_dct.py
python scripts/precompute_fnd_features.py
python src/train_full_pipeline.py
python src/evaluate_full_pipeline.py --checkpoint outputs/full_pipeline_clean/best.pt
```

## Dataset build variants

Four dataset construction scripts exist for different experimental conditions:

| Script | Real source | Notes |
|--------|-------------|-------|
| `build_forensic_3class.py` | DGM4 origin | Original; same-source Real/Manipulated |
| `build_forensic_3class_v2.py` | NewsCLIPpings genuine | Matches OOC image distribution |
| `build_forensic_3class_clean.py` | NewsCLIPpings bank (excl. v1 overlap) | Leak-free Phase 1/2 split |
| `build_forensic_3class_strict.py` | NewsCLIPpings bank (random) | Follows PDF literally (has JPEG shortcut) |

The **clean** variant is the recommended default.

## Data and checkpoints

Not in the repo (too large). Download locally:

| Data | Source |
|------|--------|
| DGM4 | `rshaojimmy/DGM4` on HuggingFace |
| NewsCLIPpings | NewsCLIPpings test split |
| MMFakeBench | `liuxuannan/MMFakeBench` on HuggingFace |

Checkpoints (`.pt`) and precomputed caches are regenerated by the scripts above.

## Next: Phase 3 (V3) — 5-class three-signal fusion

V3 extends the pipeline to 5 classes (Real, OOC, Manipulated, AI-Text,
Double Fake) by adding a third signal: text forensics via Qwen2-7B hidden
states. The full V3 implementation guide, execution checklist, and code
mapping are in **`HANDOFF.md`**.
