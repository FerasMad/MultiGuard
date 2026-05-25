# Changelog

All notable changes to MultiGuard are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; dates are ISO-8601.

---

## [2026-05-25 P5] - Forensic detector (both approaches) + V4 integration

### Added — Forensic phase

- **Approach 2 (DCT)** detector trained end-to-end per F.12-F.22:
  torchvision ResNet50 + 1-ch Kaiming conv1 + Linear(2048,1) head, two-phase
  trainer (freeze layer1/2 in Phase 1, reinit optimizer+scheduler at epoch 6),
  BCEWithLogitsLoss, ReduceLROnPlateau on val AP. Best val_AP **0.9848** @ ep13,
  Overall test AP **0.9863** across 6/8 generators.
- **Approach 1 (RGB + Fourier mask)** detector adapted from
  `chandlerbing65nm/FakeImageDetection` per F.4-F.11: single-GPU clean wrapper
  using their resnet50 + `change_output(1)` + FrequencyMaskGenerator, loading
  `mask_15/rn50ft_spectralmask.pth` (upstream rename per F-A8). Best val_AP
  **0.9971** @ ep10, Overall test AP **0.9979** across 6/8 generators.
  StdDev across gens **0.0023** vs Approach 2's 0.0158 — 7x more consistent.
- **Data pipeline:** `prepare_genimage_v2.py` (HF bitmind parquet extraction
  for 5 gens + local midjourney + VisualNews-as-nature substitute),
  `build_splits.py`, `precompute_dct.py` (multiprocess spawn for scipy.fft
  isolation), `compute_dct_stats.py` (Welford streaming).
- **Eval:** `eval_dct.py`, `eval_rgb.py`, `build_combined_eval_table.py` (F.25
  side-by-side format), `plot_training.py` (training curves PNG), `infer.py`
  (single-image demo).
- **Doctor handoff:** unified `phases/forensic/REPORT.md` + `REPORT.docx`
  covering both approaches with embedded training-curve plots.
- **HuggingFace Hub mirrors:**
  https://huggingface.co/FerasMad/forensic-dct-v1,
  https://huggingface.co/FerasMad/forensic-rgb-v1,
  https://huggingface.co/datasets/FerasMad/genimage-midjourney-10k.

### Added — V4 integration

- `phases/v4/src/v4/models/encoders/dct_forensic.py` — `DctForensicEncoder`
  registered as `"dct_forensic_v1"` (drop-in Stage-1 init replacement for
  V4's `UnivFDEncoder` blur_jpg_v0 path).
- 9 unit tests (`test_dct_forensic_encoder.py`) + 4 integration tests
  (`test_dct_forensic_in_v4_pipeline.py`) covering forward shape, freeze
  flag, missing-ckpt fallback, real-ckpt load, fusion-pipeline plumbing,
  and the V3.1 sec 5.5 aux-head `detach()` invariant.

### Fixed — CI / lint

- CI workflow paths updated for the post-unification layout
  (`phases/v4/src/`, `phases/forensic/src/` etc.). Ruff ignore list expanded
  to whitelist intentional patterns (lazy imports in workers per Risk R2,
  NaN self-comparison, etc.). 30 tests now pass on every push.

### Documented deviations (F-series in `docs/DECISIONS.md`)

- F-A1: Approach 1 ships using FakeImageDetection's `resnet50` (not their
  full DDP train.py).
- F-A2: SD v1.4 / SD v1.5 generators unavailable on bitmind HF
  (search exhausted); only the official GenImage Drive remains as a path.
- F-A3: VisualNews used as the "real" class substitute for ImageNet "nature".
- F-A4: 256x256 source JPGs, 224x224 model input.
- F-A7: Approach 1 uses `change_output(1)` per F.7 instead of manual nn.Linear.
- F-A8: upstream renamed `rn50ft_fouriermask.pth` -> `rn50ft_spectralmask.pth`;
  we accept the rename as the spec-intended successor.

### Deferred

- 2 of 8 generators (SD v1.4 / SD v1.5) — needs browser-side GenImage Drive
  download. See `phases/forensic/FOLLOWUP.md` A.
- V4 retrain (Stage 0 FND-CLIP + Stage 2 fusion x 3 seeds) — blocked on
  155 GB dataset transfer to multiGuard dual-4090 PC. See
  `docs/MULTIGUARD_SETUP.md` step 7 and `phases/forensic/FOLLOWUP.md` D.

---

## [Unreleased] - V4 Day 0-7 build

### Added

- Full V4 package layout under `src/v4/` (core, models, data, training, evaluation, cli).
- `EncoderBase` and `FusionBase` abstract base classes for the registry pattern.
- `V3PairwiseFusion` matching V3.1 section 5 (LayerNorm + 3-pair MHA-SUM + Conv1d + AdaptiveAvgPool).
- `MLPClassifier` matching V3.1 section 6 exactly (1024 -> 512+BN+GELU+Dropout(0.5) -> 256+GELU -> 5).
- Three encoders: `FNDCLIPSemanticEncoder` (with `sem_proj` 512->768 adapter, deviation F1), `UnivFDEncoder` (1-ch conv1 + Kaiming + blur_jpg_v0 strict=False), `Qwen2TextEncoder` (last layer + masked-mean + Linear(3584, 768) + GELU, deviation F2).
- Data builders: NewsCLIPpings (classes 0/1), DGM4 (class 2 + class 4 augmentation), MMFakeBench (classes 2/3/4 via `SUBSET_LABEL_MAP`), GenImage (Stage 1 only, 8 generators), Stage 1 binary pretrain.
- `merge.py` with stratified 70/15/15 split by (label, source).
- `leakage_audit.py` (duplicate sample_id + path/label collision checks + path verification).
- `BaseTrainer` with bf16 autocast, latest.pt resume, best.pt on val F1-macro improvement.
- `BaseEvaluator` writing metrics.json + confusion_matrix.png + classification_report.txt + per_source_metrics.json.
- Transfer probe `run_transfer_probe` for MMFakeBench OOD evaluation.
- FastAPI server `app/server.py` with V3-compatible JSON shape (verdict, verdict_ar, confidence, probabilities, modules, explanation, explanation_ar).
- Helper scripts: precompute caches, download pretrained blur_jpg_prob0.pth, MMFakeBench HF download, prepare VisualNews image subset.
- Three pipeline configs: `v4_pipeline_stage0.yaml`, `v4_pipeline_stage1.yaml`, `v4_pipeline_qwen.yaml`.
- Six tests: spec-compliance (V3.1 section 5.3 pairwise SUM + section 6 classifier arch), unit (class_map, manifest schema, registry), smoke (imports).
- GitHub Actions CI at `.github/workflows/ci.yml`: ruff lint + ruff format check + smoke imports.
- `.pre-commit-config.yaml` with ruff hooks.
- Docs: `ARCHITECTURE.md`, `SETUP.md`, `CONTRIBUTING.md`, `ADD_ENCODER.md`, `ADD_DATASET.md`, `ADD_FUSION.md`, `EVAL_PROTOCOL.md`, `SPEC_COMPLIANCE_MAP.md`, `DECISIONS.md`.
- Provenance metadata in every checkpoint: config_hash, data_hash, git_sha, seed, torch_version, cuda_version, spec_version, stage.

### Pending (Day 1-7)

- Day 1: Build `forensic_5class_v4.csv` (15k rows, 3k per class) + `univfd_stage1.csv` (~30k binary).
- Day 2: Precompute caches + Stage 0 FND-CLIP fine-tune.
- Day 3: Stage 1 UnivFD binary pretrain.
- Day 4: Stage 2 fusion training across 3 seeds (42, 1337, 2024).
- Day 5: V3.1 section 7 evaluation + MMFakeBench transfer probe.
- Day 6: Server cutover (V4 on 8081 alongside V3 on 8080) + parity check.
- Day 7: Doctor handoff prep.

---

## V3 - canonical numbers (for reference)

V3 final canonical Qwen Stage-2 checkpoint (`v3/outputs/v3_pipeline_qwen/best.pt`):

| Surface | F1-macro | Accuracy | AUC-macro |
|---------|----------|----------|-----------|
| In-distribution val | 0.7215 | 0.7208 | 0.9305 |
| MMFakeBench transfer | 0.3832 | - | 0.8168 |

In-distribution test F1-macro number requires verification on Day 0 of V4 (0.5377 vs 0.7215 ambiguity flagged in V4 plan section 19).

---

## V2 (Squashed history pre-V3)

V2 introduced:
- 4-class baseline (no Class 4 Double-Fake).
- 5-class extension with RoBERTa text encoder.
- Initial PairwiseFusion + aux head + bf16 unfrozen Stage 2 retraining.
- MMFakeBench zero-shot probe.

V2 issues forwarded to V3:
- No source stratification -> source-fingerprint shortcut on class 4.
- No leakage audit -> path overlap between classes.
- Hardcoded paths everywhere -> port to V4 required rewriting builders.

---

## V1 (Initial)

- FND-CLIP V1 reference implementation (Zhou et al. ICME 2023).
- Binary OOC detection on NewsCLIPpings.
- Frozen encoder, MLP head, BCE loss.

---

## Versioning

Pre-1.0: `v4.0.0-alpha.<N>` for internal milestones during the Day 0-7 build.
`v4.0.0` reserved for doctor sign-off.
