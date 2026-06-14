# Experimental Setup — MultiGuard

> 5-class multimodal fake-news detector (V3.1 spec) + two standalone binary AI-image forensic detectors. All values below are taken from the project repository (sources cited per table). Items marked **[fill in]** are machine-specific and not recorded in the repo.

## I. Experimental Setup

### A. Hardware Environment

| Item | Value | Source |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4070 (12 GB VRAM) | user-provided |
| CPU | Intel Core i5-13500 (14 cores / 20 threads) | user-provided |
| RAM (system) | 32 GB | user-provided |
| Storage | ~155 GB for the full datasets (VisualNews 99 GB, DGM4 21 GB, NewsCLIPpings 21 GB, MMFakeBench 14 GB) | docs/MULTIGUARD_SETUP.md |
| CUDA | 12.4+ | docs/V4_PLAN.md, docs/MULTIGUARD_SETUP.md |

### B. Software Stack

| Item | Value | Source |
| --- | --- | --- |
| Operating System | Windows (development machine) | docs/V4_PLAN.md |
| Programming Language | Python 3.12 | pyproject.toml |
| DL Framework | PyTorch ≥ 2.6 (CUDA 12.4 build) · torchvision ≥ 0.21 | pyproject.toml |
| LLM / Transformers | Hugging Face Transformers ≥ 4.45, < 5.0 · accelerate ≥ 0.34 · safetensors ≥ 0.4 | pyproject.toml |
| Image pre-processing | OpenCV ≥ 4.9 · Pillow ≥ 10.2 | pyproject.toml |
| Metrics / ML utils | scikit-learn ≥ 1.4 | pyproject.toml |
| Numerics / data | NumPy ≥ 1.26 (< 2.0) · pandas ≥ 2.2 | pyproject.toml |

### C. Model Hyperparameters

| Component (stage) | Backbone | Frozen? | Embed dim | LR | Optimizer | Batch | Epochs | Dropout | Scheduler | Precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic — FND-CLIP (Stage 0) | ResNet50 + BERT-base + CLIP | Yes (CLIP frozen) | 512 → 768 | 1e-4 | AdamW | 32 | 20 | — | StepLR ×0.1 @10 | bf16 |
| Image forensic — UnivFD (Stage 1) | ResNet50 (1-ch conv1) | No (fine-tuned) | → 768 | 1e-4 | AdamW | 64 | 30 | 0.3 (head) | StepLR ×0.1 @20 | bf16 |
| Image forensic — DCT ResNet50 (detector) | ResNet50 (1-ch conv1) | Partial (2-phase) | → 768 | 1e-4 → 1e-5 | AdamW | 64 | 30 | — | ReduceLROnPlateau | fp32 |
| Image forensic — RGB+Fourier (detector) | ResNet50 | Partial (L1–L2 frozen) | → 768 | 1e-4 | AdamW | 64 | 30 | — | ReduceLROnPlateau | fp32 |
| Text forensic — Qwen2-7B-Instruct | Qwen2-7B (last layer, masked-mean) | Yes (frozen) | 3584 → 768 | — | — | — | — | — | — | — |
| Fusion — V3PairwiseFusion (Stage 2) | 3-pair cross-attn (8 heads) + Conv1d | trained | 768 → 1024 | 1e-4 | AdamW | 64 | 50 | 0.5 (MLP) / 0.1 (attn) | StepLR ×0.1 @30 | bf16 |

_Common: weight decay = 1e-4, gradient clipping max-norm = 1.0, early stopping (patience 5–10 on val AUC/F1). Loss (Stage 2): CrossEntropy (5-class) + 0.1 × BCEWithLogits auxiliary head on detached v_imgfor. Source: phases/v4/configs/*.yaml, phases/v4/src/v4/training/{trainer,losses,schedulers}.py._

## II. Evaluation Results & Ablation Studies

_Final 5-class model = deployed 3-seed honest ensemble (seeds 42 / 1337 / 2024, softmax-averaged). Test split = 2,475 held-out samples, 495 per class._

### Result R1 — 5-Class Classification Report (final model)

_Source: docs/MODEL_REPORT.md §3._

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| 0 Real | 0.428 | 0.370 | 0.397 | 495 |
| 1 Out-of-Context | 0.447 | 0.438 | 0.442 | 495 |
| 2 Manipulated | 0.707 | 0.814 | 0.757 | 495 |
| 3 AI-Text | 0.984 | 0.994 | 0.989 | 495 |
| 4 Fully-Fabricated | 0.994 | 0.986 | 0.990 | 495 |
| F1-macro (overall) | — | — | 0.7149 | 2,475 |

### Result R2 — 5-Class Confusion Matrix (final model)

_Source: docs/MODEL_REPORT.md §3. Rows = true, columns = predicted._

| true \ pred | Real | OOC | Manip | AI-Text | Fab |
| --- | --- | --- | --- | --- | --- |
| Real | 183 | 227 | 85 | 0 | 0 |
| Out-of-Context | 196 | 217 | 82 | 0 | 0 |
| Manipulated | 49 | 42 | 403 | 1 | 0 |
| AI-Text | 0 | 0 | 0 | 492 | 3 |
| Fully-Fabricated | 0 | 0 | 0 | 7 | 488 |

![Confusion matrix](confusion_matrix_5class_ensemble.png)

### Result R3 — Ablation A: Training-time Branch Contribution

_Each variant trained from scratch with only the listed branches. Source: docs/IMAGE_BRANCH_ABLATION.md._

| Variant | Test F1 | C0 Real | C1 OOC | C2 Manip | C3 AI-Text | C4 FullFab |
| --- | --- | --- | --- | --- | --- | --- |
| fndclip (semantic only) | 0.6681 | 0.451 | 0.306 | 0.738 | 0.886 | 0.960 |
| fndclip + text | 0.7103 | 0.411 | 0.448 | 0.731 | 0.981 | 0.982 |
| fndclip + image | 0.6975 | 0.391 | 0.464 | 0.751 | 0.911 | 0.970 |
| all three (full) | 0.7157 | 0.430 | 0.426 | 0.752 | 0.985 | 0.985 |

### Result R4 — Ablation B: Inference-time Leave-One-Out

_One branch zeroed at inference on the shipped ensemble. Source: docs/IMAGE_BRANCH_ABLATION.md._

| Variant | Test F1 | ΔF1 | C2 Manip | Note |
| --- | --- | --- | --- | --- |
| full pipeline | 0.7149 | — | 0.757 | baseline |
| disable image (v_imgfor) | 0.6954 | -2.0 pp | 0.733 | image removal hits C2 most |
| disable text (v_textfor) | 0.4494 | -26.6 pp | 0.710 | text dominates C3/C4 |
| semantic only | 0.3146 | -40.0 pp | 0.655 | — |

### Result R5 — Per-Seed Results & External Transfer

_Source: phases/v4/docs/eval/honest_run_summary.json._

| Seed | Test F1-macro | MMFakeBench transfer F1 (raw) |
| --- | --- | --- |
| 42 | 0.7152 | 0.3516 |
| 1337 | 0.7189 | 0.3446 |
| 2024 | 0.7009 | 0.4308 |
| mean ± std | 0.7117 ± 0.0095 | 0.3757 ± 0.0479 |
| 3-seed ensemble | 0.7149 | 0.4308 (raw) / 0.7197 (bias-corrected) |

### Result R6 — Forensic Image Detectors (8/8 generators, official nature)

_Two standalone binary real-vs-AI detectors. Metric = Average Precision (AP). Source: docs/MODEL_REPORT.md §4, outputs/eval_table_combined.md._

| Generator | Type | A1 RGB+Fourier AP | A2 DCT AP |
| --- | --- | --- | --- |
| midjourney | Diffusion | 0.965 | 0.828 |
| sdv1_4 | Diffusion | 0.988 | 0.890 |
| sdv1_5 | Diffusion | 0.985 | 0.896 |
| wukong | Diffusion | 0.980 | 0.902 |
| vqdm | Diffusion | 0.976 | 0.830 |
| adm | Diffusion | 0.997 | 0.971 |
| glide | Diffusion | 0.987 | 0.975 |
| biggan | GAN | 0.995 | 0.976 |
| Overall AP |  | 0.9841 | 0.9085 |
| Overall Accuracy |  | 0.940 | 0.834 |
| Std-dev AP |  | 0.011 | 0.061 |

### Notes / Honest Limitations

- Classes 3 (AI-Text) and 4 (Fully-Fabricated) are near-perfect (F1 ≈ 0.99); the Qwen2 text branch isolates AI-written text cleanly.
- The error mass is in Real ↔ Out-of-Context (linear-probe AUC ≈ 0.51, at-chance) — a frozen-FND-CLIP / data-scale ceiling, not a fusion bug.
- A1 (RGB+Fourier) is the recommended forensic detector; A2 (DCT) is the weaker frequency-domain baseline (notably on MidJourney/VQDM).
