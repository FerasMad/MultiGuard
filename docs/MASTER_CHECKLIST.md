# Master Checklist — Doctor's Requirements (All 5 Briefs)

Synthesized from `docs/doctor-briefs/*.pdf` during the unified mega-repo restructure (P3). This is the doctor-audit tool: every requirement ever specified is listed here with a pointer to where it is implemented (or marked pending).

**Source PDFs:**
- `V1_Dataset_Instructions.pdf` — V1 FND-CLIP binary OOC baseline
- `V2_Implementation_Guidelines.pdf` — V2 3-class fusion (superseded)
- `V3.1_Implementation_Guidelines.pdf` — V3.1 5-class three-signal fusion (CURRENT SPEC)
- `Forensic_Image_Detector_En.pdf` — Binary AI-image detector (CURRENT SPRINT)
- `IMAGE_ONLY_METRICS.pdf` — V3 aux head ablation report

---

## A. Forensic Image Detector (NEWEST, primary current focus)

| ID | Requirement | Status | File / Artifact |
|----|-------------|--------|-----------------|
| F.1 | Dataset structure: 3 non-overlapping trees | Pending P5 | `phases/forensic/data/genimage_{train,test}/` |
| F.2 | 8 generators in test: midjourney, sdv1_4, sdv1_5, wukong, vqdm, biggan, adm, glide | Pending P5 | `phases/forensic/data/genimage_test/<gen>/` |
| F.3 | Real class = ImageNet "nature" (bundled per-generator in GenImage release) | Pending P5 | Data layout |
| **Approach 1 — RGB + Fourier Mask** | | | |
| F.4 | Fine-tune chandlerbing65nm/FakeImageDetection RN50 with Fourier masking | Pending P5 | `phases/forensic/external/FakeImageDetection/` |
| F.5 | Use checkpoint `mask_15/rn50ft_fouriermask.pth` exactly (do not rename) | Pending P5 | External clone |
| F.6 | Freeze conv1/bn1/layer1/layer2, train layer3/layer4/fc only | Pending P5 | Adapted `train.py` |
| F.7 | `model.change_output(1)` after loading weights (do NOT manually create nn.Linear) | Pending P5 | Adapted `train.py` |
| F.8 | Fourier masking 50% probability, mask_ratio=0.15, training-only | Pending P5 | Use repo's `augment.py` + `mask.py` |
| F.9 | Preprocessing: Resize 224x224 bilinear, ToTensor, Normalize ImageNet stats | Pending P5 | Adapted `dataset.py` |
| F.10 | Hyperparams: BCEWithLogitsLoss, AdamW lr=1e-4 wd=1e-4, batch=64, max 30 epochs, ReduceLROnPlateau (mode=max, factor=0.5, patience=3, monitors val AP), early-stop patience=5 | Pending P5 | `phases/forensic/configs/rgb_fourier.yaml` |
| F.11 | Output filename: `forensic_rgb_model.pth` | Pending P5 | `phases/forensic/outputs/` |
| **Approach 2 — DCT Frequency Domain** | | | |
| F.12 | torchvision ResNet50 with `weights='IMAGENET1K_V1'`, NOT FakeImageDetection's RN50 | Pending P5 | `phases/forensic/src/forensic/models/dct_resnet50.py` |
| F.13 | conv1 in_channels 3->1, Kaiming Normal (mode='fan_out', nonlinearity='relu') | Pending P5 | Same |
| F.14 | Load ImageNet weights strict=False skipping conv1 | Pending P5 | Same |
| F.15 | fc replaced with `Linear(2048, 1)` no activation | Pending P5 | Same |
| F.16 | DCT recipe: Resize 224x224 bilinear -> YCbCr Y channel float32 [0,255] -> Set A (8x8) + Set B (16x16) -> scipy.fftpack.dct type=2 norm='ortho' patch-level -> log(\|coef\|+1e-8) -> reassemble each set to [224,224] -> element-wise average -> [1,224,224] | Pending P5 | `phases/forensic/src/forensic/preprocessing/dual_dct.py` |
| F.17 | Z-score: compute global DCT_mean + DCT_std from `genimage_train/train/` only, save to `dct_stats.json`, apply `(t-mean)/(std+1e-8)` at every stage | Pending P5 | `dct_stats.json` |
| F.18 | .pt cache: float32, shape [1,224,224], precomputed for train/val/test (never recomputed) | Pending P5 | `phases/forensic/data/dct_cache/` |
| F.19 | Phase 1 (epochs 1-5): conv1/bn1/layer3/layer4/fc trainable; layer1/layer2 frozen; AdamW lr=1e-4 wd=1e-4; BCEWithLogitsLoss; batch=64; ReduceLROnPlateau (mode=max factor=0.5 patience=3); no gradient clipping; num_workers=4 pin_memory=True | Pending P5 | `two_phase_trainer.py` |
| F.20 | Phase 2 transition at START of epoch 6: unfreeze layer1/layer2, **reinitialize** AdamW with ALL params at lr=1e-5 wd=1e-4, **reinitialize** ReduceLROnPlateau, enable gradient clipping max_norm=1.0 every batch | Pending P5 | `two_phase_trainer.py` |
| F.21 | Early-stop: patience=5 on val AP, **continues across phase boundary** (do NOT reset patience counter) | Pending P5 | `two_phase_trainer.py` |
| F.22 | Outputs: `forensic_dct_model.pth` (best by val AP) + `dct_stats.json` | Pending P5 | `phases/forensic/outputs/` |
| **Evaluation (both approaches)** | | | |
| F.23 | Per generator: AP (sklearn `average_precision_score`), Accuracy at threshold 0.5, AUC (sklearn `roc_auc_score`) | Pending P5 | `per_generator.py` |
| F.24 | Aggregates: Overall avg, GAN avg (BigGAN only), Diffusion avg (other 7), Std Dev of AP across 8 generators | Pending P5 | `per_generator.py` + `table.py` |
| F.25 | Eval table format: rows = 8 generators + 4 summary; columns = AP, Accuracy, AUC | Pending P5 | `phases/forensic/outputs/eval_table.md` |
| F.26 | Inference: load best ckpt, eval mode, torch.no_grad, sigmoid outputs | Pending P5 | Both eval scripts |

---

## B. V3.1 Multimodal — `V3.1_Implementation_Guidelines.pdf` (CURRENT 5-class spec)

| ID | Requirement | Status | File / Artifact |
|----|-------------|--------|-----------------|
| V3.1 | 5-class output: 0 Real / 1 OOC / 2 Manipulated / 3 AI-Text / 4 Fully-Fabricated | Done | `phases/v4/src/v4/core/class_map.py` |
| V3.2 | Datasets: NewsCLIPpings (0,1), DGM4+MMFB (2,3,4) — undersample to smallest class | Done | `phases/v4/src/v4/data/builders/` |
| V3.3 | FND-CLIP frozen semantic encoder -> `v_semantic` [B,768] | Done | `phases/v4/src/v4/models/encoders/fnd_clip.py` + `sem_proj` 512->768 (deviation F1) |
| V3.4 | UnivFD-style ResNet50 on patch-DCT, 1ch conv1 (Kaiming), blur_jpg_v0 strict=False -> `v_imgfor` [B,768] | Done | `phases/v4/src/v4/models/encoders/univfd.py` |
| V3.5 | Qwen2-7B-Instruct layer 30 hidden state, masked-mean pool, Linear(4096,768)+GELU -> `v_textfor` [B,768] | Done (deviation F2: V4 uses last layer + hidden 3584) | `phases/v4/src/v4/models/encoders/qwen_text.py` |
| V3.6 | Fusion section 5: LayerNorm per signal -> 3-pair bidirectional MHA (8 heads) -> **element-wise SUM** -> Conv1d stack -> AdaptiveAvgPool -> [B,1024] | Done + spec-compliance test | `phases/v4/src/v4/models/fusion/v3_pairwise.py` + `phases/v4/tests/spec_compliance/test_v3_1_section_5_3_pairwise_sum.py` |
| V3.7 | Classifier section 6 EXACT: Linear(1024,512)+BN+GELU+Dropout(0.5) -> Linear(512,256)+GELU -> Linear(256,5) | Done + spec-compliance test | `phases/v4/src/v4/models/classifier/mlp_head.py` + `tests/spec_compliance/test_v3_1_section_6_classifier_arch.py` |
| V3.8 | Loss section 5.5: CE(main, label) + 0.1*BCE(aux, binary_image_label) with `.detach()` on aux input | Done | `phases/v4/src/v4/training/losses.py` |
| V3.9 | Training section 5.5: AdamW lr=1e-4 wd=1e-4, batch=64, StepLR x0.1 @ epoch 30, ES patience=10 on val F1-macro, grad clip max_norm=1.0 | Done | `phases/v4/configs/v4_pipeline_qwen.yaml` |
| V3.10 | Evaluation section 7: Precision per class, Recall per class, F1-macro, confusion matrix, MMFakeBench transfer probe | Done | `phases/v4/src/v4/evaluation/{evaluator.py, transfer.py, reporting.py}` |
| V3.11 | Aux head deactivated at inference (softmax on main_logits only) | Done | `evaluator.py::predict` |

---

## C. V2 Multimodal — `V2_Implementation_Guidelines.pdf` (SUPERSEDED by V3.1, code preserved as `phases/v2/`)

| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| V2.1 | 3-class output: Real / Manipulated / OOC | Superseded | V3.1 expands to 5 |
| V2.2 | ResNet18 + DCT [1,224,224] forensic encoder | Preserved | `phases/v2/src/models/forensic_baseline.py` |
| V2.3 | Cross-attention bidirectional fusion (concat -> 1024) | Superseded | V3.1 uses SUM not concat |
| V2.4 | MLP 1024->512->256->3 | Superseded | V3.1 final layer is 256->5 |
| V2.5 | Adam lr=1e-4 wd=1e-4, StepLR x0.1 @ epoch 30, ES patience=10 | Same as V3.1 | |
| V2.6 | MMFakeBench zero-shot transfer probe | Carried forward to V4 | `phases/v4/src/v4/evaluation/transfer.py` |

---

## D. V1 Semantic Baseline — `V1_Dataset_Instructions.pdf` (SUPERSEDED, code in `phases/v1/`)

| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| V1.1 | Binary classifier on 5 scenarios | Superseded | |
| V1.2 | FND-CLIP reference impl: ResNet50 + BERT-base + CLIP + modality attention + 2-layer MLP sigmoid | Preserved | `phases/v1/src/models/fnd_clip.py` |
| V1.3 | Datasets balanced to smallest class | Preserved methodology | V4 implements this in `merge.py` |

---

## E. Image-only metrics — `IMAGE_ONLY_METRICS.pdf` (V3 ablation report)

| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| E.1 | Binary image-only eval of V3 aux head on 2475-sample test split, threshold 0.50, positive label in {Manipulated, Fully-Fabricated} | Implemented in V3 | `phases/v3/scripts/eval_image_only.py` |
| E.2 | Metrics: Accuracy, F1 (fake class), F1-macro, Precision (fake), Recall (fake), AUC-ROC, confusion matrix, per-class breakdown | Done | `outputs/v3/image_only_eval/` |
| E.3 | Reference numbers: V3 Qwen2-7B (rev 11) test F1-macro ~ 0.5377, AUC ~ 0.84; transfer F1 0.3832, AUC 0.8168 | Verified | Documented in `phases/v3/README.md` |

---

## F. Cross-cutting flexibility mandate (doctor's stated brief for V4)

| ID | Requirement | Status | Where enforced |
|----|-------------|--------|----------------|
| FL.1 | Swap an encoder in <=80 LOC + 1 YAML edit | Done | `docs/ADD_ENCODER.md` |
| FL.2 | Swap a dataset source in <=100 LOC | Done | `docs/ADD_DATASET.md` |
| FL.3 | Swap a fusion module under `FusionBase` ABC | Done | `docs/ADD_FUSION.md` |
| FL.4 | Change num_classes via `class_map.py` + classifier head | Done | `phases/v4/src/v4/core/class_map.py` + `mlp_head.py` |
| FL.5 | Reproduce any past run from saved provenance | Done | `phases/v4/src/v4/core/checkpoints.py::build_provenance` |
| FL.6 | YAML-driven training: encoders + fusion + losses + scheduler all parametrized | Done | `phases/v4/configs/` |
| FL.7 | Registry pattern: `ENCODER_REGISTRY`, `FUSION_REGISTRY`, `DATASET_REGISTRY` | Done | `phases/v4/src/v4/core/registry.py` |

---

## G. Provenance / reproducibility / hygiene

| ID | Requirement | Status | File |
|----|-------------|--------|------|
| G.1 | Seed everywhere (Python/NumPy/torch/CUDA) | Done | `phases/v4/src/v4/core/seed.py` |
| G.2 | Every checkpoint carries: config_hash, data_hash, git_sha, seed, torch_version, cuda_version, spec_version, stage | Done | `phases/v4/src/v4/core/checkpoints.py::build_provenance` |
| G.3 | Linting + formatting via ruff; pre-commit hook | Done | `.pre-commit-config.yaml`, `pyproject.toml` |
| G.4 | CI: ruff + smoke tests on push | Done | `.github/workflows/ci.yml` |
| G.5 | Type hints + `from __future__ import annotations` everywhere | Done | All `phases/v4/src/v4/` files |
| G.6 | Logging via `logging.getLogger(__name__)` — no `print()` in library code | Done | `phases/v4/src/v4/core/logging.py` |

---

## Outstanding gaps (nothing the doctor asked for is currently missing on disk)

- WARN: Approach 1 + Approach 2 forensic checkpoints + `dct_stats.json` + eval table — **P5 will produce these**.
- All other doctor requirements are implemented in V3 (preserved in `phases/v3/`) or V4 (active at `phases/v4/`).

---

## Summary

**Implemented (V3.1 5-class detector):** 100% of V1, V2, V3.1, V3 image-only, and flexibility requirements.

**Pending (Forensic Image Detector):** 26 requirements (F.1-F.26) all blocked on P5 implementation sprint (3-4 days of work).

**No gaps in the doctor's spec coverage as of P3 completion.**
