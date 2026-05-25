# Changelog

All notable changes to MultiGuard are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; dates are ISO-8601.

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
