# V3.1 Spec Compliance Map

This is the doctor-audit table. Each row maps a V3.1 spec section to the V4 file and (where applicable) the test that asserts compliance. Update this whenever you change a referenced file.

V3.1 spec PDF: `Implementation Guidelines V3_1.pdf` (the version frozen on Day 0).

---

## Section 1 - Modalities

| Spec point | V4 file | Notes |
|------------|---------|-------|
| 3 modalities: `v_semantic`, `v_imgfor`, `v_textfor` | `src/v4/core/class_map.py`, `configs/v4_pipeline_qwen.yaml` | feature_keys list |
| Each is 768-dim at fusion input | `src/v4/models/fusion/v3_pairwise.py` | `feat_dim=768` |

---

## Section 2 - Datasets

| Spec point | V4 file | Notes |
|------------|---------|-------|
| Class 0 = NewsCLIPpings Matched | `src/v4/data/builders/build_newsclippings.py` | |
| Class 1 = NewsCLIPpings Mismatched | `src/v4/data/builders/build_newsclippings.py` | |
| Class 2 = DGM4 + MMFB tampered | `src/v4/data/builders/build_dgm4.py` + `build_mmfakebench.py` | per [A2] in V4 plan |
| Class 3 = MMFB AI-text subsets | `src/v4/data/builders/build_mmfakebench.py` | `SUBSET_LABEL_MAP` |
| Class 4 = MMFB AI/AI + DGM4 multimodal | `build_mmfakebench.py` + `build_dgm4.py` | DGM4 capped at 800 [A1] |
| Stratified split 70/15/15 | `src/v4/data/builders/merge.py::_stratified_split` | per `(label, source)` tuple |
| Leakage audit | `src/v4/data/builders/leakage_audit.py` | duplicates + path/label collisions |

---

## Section 3.1 - FND-CLIP semantic encoder

| Spec point | V4 file | Test |
|------------|---------|------|
| ResNet-50 + BERT + CLIP + modality attention | `src/v4/models/encoders/fnd_clip.py` | shape test (planned) |
| Modality attention via attention-weighted sum | `src/v4/models/encoders/fnd_clip.py::ModalityAttention` | |
| Output dim = 512 (then sem_proj to 768) | `src/v4/models/encoders/fnd_clip.py::FNDCLIPSemanticEncoder.sem_proj` | deviation [F1] |

---

## Section 3.2 - UnivFD forensic encoder (Stage 1)

| Spec point | V4 file | Test |
|------------|---------|------|
| ResNet-50, conv1 modified to 1-channel | `src/v4/models/encoders/univfd.py::_make_1ch_conv1` | shape test (planned) |
| Kaiming Normal init for conv1 | `src/v4/models/encoders/univfd.py::_make_1ch_conv1` | |
| Load blur_jpg_v0.pth with strict=False | `src/v4/models/encoders/univfd.py::load_legacy_checkpoint` | |
| Output dim = 768 | `src/v4/models/encoders/univfd.py` | `out_dim=768` |
| BCEWithLogitsLoss for binary pretrain | `configs/v4_pipeline_stage1.yaml` | |
| AdamW lr=1e-4 | `configs/v4_pipeline_stage1.yaml` | |

---

## Section 4 - Qwen text-forensic encoder

| Spec point | V4 file | Test |
|------------|---------|------|
| Qwen2-7B-Instruct, layer pool | `src/v4/models/encoders/qwen_text.py` | deviation [F2]: last layer (-1) |
| Hidden 3584 (spec says 4096) | `src/v4/models/encoders/qwen_text.py` | deviation [F2] |
| Linear(3584, 768) + GELU adapter | `src/v4/models/encoders/qwen_text.py::TextForensicProjection` | |
| Masked-mean pooling | `src/v4/data/preprocessing/tokenizers.py::masked_mean_pool` | |

---

## Section 5 - Fusion

| Spec point | V4 file | Test |
|------------|---------|------|
| LayerNorm per signal | `src/v4/models/fusion/v3_pairwise.py::V3PairwiseFusion.norms` | |
| 3-pair MHA-sum (bidirectional, SUM not concat) | `src/v4/models/fusion/v3_pairwise.py::PairwiseCrossAttention` | `test_v3_1_section_5_3_pairwise_sum.py` |
| 8 attention heads | `configs/v4_pipeline_qwen.yaml` | `num_heads: 8` |
| 1D-Conv stack (768->768 k=3, then 768->1024 k=1) | `src/v4/models/fusion/v3_pairwise.py::ThreeWayFusion.conv_stack` | |
| AdaptiveAvgPool to [B, 1024] | `src/v4/models/fusion/v3_pairwise.py::ThreeWayFusion` | |

---

## Section 5.5 - Training hyperparameters

| Spec point | V4 file | Notes |
|------------|---------|-------|
| Batch size 64 | `configs/v4_pipeline_qwen.yaml::train.batch_size` | |
| AdamW lr=1e-4 (corrected from V3's 5e-5) | `configs/v4_pipeline_qwen.yaml::train.lr` | deviation A3 fixed |
| Weight decay 1e-4 | `configs/v4_pipeline_qwen.yaml::train.weight_decay` | |
| CE + 0.1 x BCE-aux with detach | `src/v4/training/losses.py::CompositeLoss` + `configs/v4_pipeline_qwen.yaml::train.aux_weight` | |
| Aux input is `v_imgfor.detach()` | `src/v4/training/losses.py` `detach_input` field | |
| StepLR x0.1 @ epoch 30 | `configs/v4_pipeline_qwen.yaml::train.lr_step` + `lr_gamma` | |
| Early-stop patience 10 on val F1-macro | `configs/v4_pipeline_qwen.yaml::train.early_stop_patience` + `src/v4/training/trainer.py` | |
| Grad clip max_norm = 1.0 | `configs/v4_pipeline_qwen.yaml::train.grad_clip` + `src/v4/training/trainer.py` | |

---

## Section 6 - Classifier head

| Spec point | V4 file | Test |
|------------|---------|------|
| Linear(1024, 512) + BatchNorm1d + GELU + Dropout(0.5) | `src/v4/models/classifier/mlp_head.py` | `test_v3_1_section_6_classifier_arch.py` |
| Linear(512, 256) + GELU | `src/v4/models/classifier/mlp_head.py` | same test |
| Linear(256, 5) | `src/v4/models/classifier/mlp_head.py` | same test |

---

## Section 7 - Evaluation

| Spec point | V4 file | Notes |
|------------|---------|-------|
| Deactivate aux head at inference | `src/v4/evaluation/evaluator.py::BaseEvaluator.predict` | uses only `main_logits` |
| Softmax on main logits | `src/v4/evaluation/evaluator.py` | |
| Precision per class | `src/v4/evaluation/reporting.py::compute_metrics` | sklearn |
| Recall per class | same | |
| F1-macro | same | primary headline |
| Confusion matrix | `src/v4/evaluation/reporting.py::write_confusion_matrix_png` | |
| MMFakeBench probe | `src/v4/evaluation/transfer.py::run_transfer_probe` | |

---

## Provenance and reproducibility

| Spec point | V4 file | Notes |
|------------|---------|-------|
| Seed everywhere | `src/v4/core/seed.py::seed_all` | Python/NumPy/torch/CUDA |
| Save provenance with checkpoint | `src/v4/core/checkpoints.py::build_provenance` | config_hash, data_hash, git_sha, etc. |
| Reproduce from saved provenance | `src/v4/cli.py::reproduce_main` (planned) | |

---

## Files with no spec correspondence (V4 infrastructure)

These are V4-specific scaffolding with no V3.1 spec line; they exist to make the spec-compliant pieces reusable:

- `src/v4/core/registry.py` - encoder/fusion/dataset registries
- `src/v4/core/config.py` - YAML loader + dataclass schema
- `src/v4/core/paths.py` - project root resolution
- `src/v4/core/logging.py` - logging setup
- `src/v4/data/manifest.py` - canonical CSV schema
- `src/v4/data/datasets/*.py` - dataset abstraction
- `app/server.py` - inference server (not in V3.1 spec)
- `scripts/*.py` - tooling

---

## How to update this map

When you change a file referenced in this table:

1. Verify the file:line citation still points to the correct construct.
2. If you split a function into multiple files, add a new row.
3. If a deviation changes (e.g. F2 resolves because the spec gets clarified), mark the row resolved and move it to a "Resolved deviations" section.
4. Run `make test` - if a spec-compliance test breaks, you've violated the spec; do not silence the test.

The doctor uses this table during audit. Keep it accurate.
