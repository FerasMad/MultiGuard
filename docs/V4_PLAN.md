# V4 Multimodal Fake-News Detector — Rebuild Plan

**Status:** plan-only. Implementation happens on a separate PC (Windows + Python 3.12 + CUDA 12.4 + dual RTX 4090, 48 GB total VRAM). One-week wallclock. Same 5-class task as V3.1; clean rebuild for reusability per the doctor's brief.

This document is the source of truth for the V4 rebuild. The companion `docs/DECISIONS.md` (to be authored from this file) records *why* each decision was made.

---

## 1. Context

### Why this rebuild

The doctor instructed a from-scratch rebuild emphasizing **reusability** so future students can swap models and datasets easily ("we're going to build the general structure for the next people who are going to work on this project"). V3 produced strong numbers (val F1-macro 0.7215, val accuracy 0.7208, val AUC-macro 0.9305) but the codebase shows the cost of accreted growth — no base classes, duplicated preprocessing, no shortcut audit, no leakage discipline. V4 fixes the engineering while remaining V3.1-spec-aligned.

### Team ownership (preserved from V3)

V4 is implemented by a solo developer on the new PC, but the code structure is organized as if the original 4-student team still owns one subpackage each. This matches the doctor's mental model and makes future handoff to a 4-student team possible without restructuring.

| Subpackage | Owner | V3 deliverable | V4 deliverable |
|---|---|---|---|
| `src/models/encoders/univfd.py` + `src/data/preprocessing/patch_dct.py` | Student 1 | UnivFD encoder + patch-DCT precompute | Same, refactored into EncoderBase contract |
| `src/models/encoders/qwen_text.py` + `src/models/encoders/text_fluoroscopy.py` | Student 2 | Qwen2-7B layer-30 + TextForensicProjection | Same, refactored |
| `src/models/fusion/v3_pairwise.py` + `src/models/classifier/mlp_head.py` | Student 3 | V3FusionModule + V3Classifier | Same, refactored into FusionBase contract |
| `src/data/` + `src/training/` + `src/evaluation/` | Student 4 | Dataset builders + trainer + evaluator | Same, refactored. Also owns Stage 0 (FND-CLIP fine-tune) |
| `src/models/encoders/fnd_clip.py` (Stage 0) | Student 4 | Inherited from V1 frozen | Fresh fine-tune from V1 paper architecture |

### V3 numbers on disk (to verify on Day 0)

V3 canonical checkpoint `v3/outputs/v3_pipeline_qwen/best.pt`:
- **Val** F1-macro 0.7215, accuracy 0.7208, AUC-macro 0.9305
- **Test** F1-macro — needs verification on Day 0; the 0.5377 number quoted earlier in chat may have been from a different artifact
- **Transfer (MMFakeBench)** F1-macro 0.3832, AUC 0.8168

V4 PASS gate (implicit, since user chose strict-spec eval): produce the four V3.1 §7 metrics on test split and transfer probe; doctor reviews.

---

## 2. V3.1 Spec audit & deviation register

### Aligned (no action)
17 spec points map exactly: patch-DCT preprocessing, UnivFD architecture, fusion §5 (LayerNorm per signal, 3-pair MHA-sum, 1D-Conv stack, AdaptiveAvgPool), classifier §6 (1024→512+BN+GELU+Dropout(0.5)→256+GELU→5), training §5.5 (batch 64, AdamW wd=1e-4, CE + 0.1×BCE-aux with detach, StepLR ×0.1 @ epoch 30, ES patience 10), grad clip 1.0, eval §7 metrics.

### Fixable deviations (V3 had them; V4 fixes)

| ID | Spec | V3 did | V4 will do |
|---|---|---|---|
| **A1** | §2 datasets = NewsCLIPpings + DGM4 + MMFakeBench | Used GenImage MidJourney for class 4 in 5-class data | Class 4 = MMFakeBench AI-Image+AI-Text + DGM4 multimodal-paired-fake augmentation. **GenImage stays in Stage 1 UnivFD pretrain only** |
| **A2** | §2 Class 2 = DGM4 / MMFakeBench tampered | DGM4 only | DGM4 + MMFakeBench tampered subsets |
| **A3** | §5.5 AdamW lr=1e-4 | 5e-5 sweep result | **lr = 1e-4 per spec** |

### Forced deviations (model-architecture realities the spec didn't anticipate)

| ID | Spec | Reality | V4 will do | Reason |
|---|---|---|---|---|
| **F1** | §1 v_semantic = 768 | FND-CLIP V1 (Zhou et al. ICME 2023) outputs 512 | `sem_proj`: Linear(512, 768) + GELU as a registered learnable adapter | Preserve published FND-CLIP architecture |
| **F2** | §4 Qwen Layer 30, hidden 4096 | Qwen2-7B-Instruct has 28 layers, hidden 3584 | Last layer (-1), Linear(3584, 768) + GELU. Confirmed user choice. | Spec language matches Qwen1.5-7B; we use Qwen2-7B-Instruct (V3-proven, dual-4090 compatible) |
| **F3** | §3.2 Pretrain data unspecified | n/a | 10k GenImage + 3k DGM4 + 2k MMFakeBench tampered (fake) vs 10k VisualNews + 3k DGM4 origin + 2k MMFakeBench real (real). Total ~30k balanced binary. | Standard UnivFD recipe; user confirmed GenImage approved for Stage 1 only |

---

## 3. Datasets

### Stage 1 (UnivFD binary pretrain) — ~30k balanced binary set

| Source | Count | Side | Notes |
|---|---|---|---|
| **GenImage** | 10,000 | Fake | All 8 official GenImage generators stratified ~1,250 each: SD v1.4, SD v1.5, MidJourney, ADM, GLIDE, VQDM, BigGAN, Wukong |
| **DGM4** | 3,000 | Fake | HFGI + face_swap + face_edit, balanced |
| **MMFakeBench tampered** | 2,000 | Fake | coco_image_edit + Fakeddit_photo_edit + Newsclipings_person/scene/semantic |
| **VisualNews genuine** | 10,000 | Real | Guardian + USA Today + WSJ + BBC, no overlap with 5-class NewsCLIPpings rows |
| **DGM4 origin** | 3,000 | Real | DGM4 paired-real images (originals of manipulated samples) |
| **MMFakeBench real** | 2,000 | Real | MMFakeBench's real-image subsets |

- Loss: BCEWithLogitsLoss (binary, per spec §3.2)
- Optimizer: AdamW lr=1e-4 (per spec §3.2)
- Pre-trained init from `blur_jpg_v0.pth` with `strict=False`; conv1 modified to 1 channel + Kaiming Normal init
- Output: `outputs/v4/stage1_univfd/forensic_model.pth` per spec §3.2

### Stage 2 (5-class fusion training) — 15,000 samples total (3,000/class)

| Class | Source | Sub-sources |
|---|---|---|
| 0 Real | NewsCLIPpings Matched | Guardian + USA Today + WSJ + BBC subsets |
| 1 OOC | NewsCLIPpings Mismatched | Guardian + USA Today + WSJ + BBC subsets |
| 2 Manipulated | DGM4 + MMFakeBench tampered | DGM4 (face_attribute, face_swap, face_edit) + MMFakeBench (coco_image_edit, Newsclipings_person/scene/semantic, Fakeddit_photo_edit) |
| 3 AI-Text | MMFakeBench AI-text subsets | fever_AI + gossipcop_match + politicat_match + rumor_match + llm_rewrite + coco_text_edit (val + test partitions both used) |
| 4 Double Fake | MMFakeBench AI/AI + DGM4 multimodal | chatgpt_match + antifact_image_generation + llm_gossip_md + llm_science_md + gossipcop_midjourney (~2,200) + DGM4 multimodal-paired-fake (~800) per V1 Dataset PDF authorization |

### Sampling and splits

- **Undersample to 3,000/class** (V3.1 §2.b, target chosen via interview C1)
- **70 / 15 / 15** train / val / test per class (V2 spec)
- **Stratified by `(label, source)` tuple** so each source appears proportionally in all three splits (V3 lesson — eliminates source-fingerprint shortcuts)
- All MMFakeBench rows (val + test partitions combined) enter the 5-class dataset; OOD probe per spec §7 implemented as per-source breakdown filter `source.startswith("MMFakeBench_")` on the test split, plus a separate convenience pass over the whole MMFakeBench_test directory
- `src/data/builders/leakage_audit.py` runs after manifest construction: asserts no `sample_id` appears in two splits; no `image_path` reappears under different labels

### Dependencies (image-source only)

| Source | Role |
|---|---|
| VisualNews | Raw image files for NewsCLIPpings (loader/resolver only, no labels) |

### Data ops on the new PC

1. `scripts/download_data.py` auto-fetches NewsCLIPpings + MMFakeBench from HuggingFace
2. `scripts/download_pretrained.py` auto-fetches `blur_jpg_v0.pth` from the CNNDetection repo
3. DGM4 + GenImage are documented as manual prerequisites in `docs/SETUP.md` (requires `huggingface-cli login` for DGM4, GenImage from HuggingFace `Andyrasika/genimage_v0` or official benchmark links)
4. `scripts/prepare_image_subset.py` runs AFTER the 5-class manifest is built and copies only the VisualNews images referenced by our manifest (~3 GB) to the new PC, avoiding the full ~50 GB VisualNews download

### Dataset acquisition procedure — Day 1 critical path

#### Step 0 — Pre-flight (before Day 1)
| # | Task | Output |
|---|---|---|
| 0.1 | Verify TeamViewer File-Transfer connection from new PC to current PC | TeamViewer session tested with one small file |
| 0.2 | Disable sleep/hibernate on BOTH PCs (current PC must stay awake during overnight transfer) | Power plans confirmed |
| 0.3 | Verify new PC has stable ≥10 Mbps internet (for small HF downloads only) | Bandwidth confirmed |
| 0.4 | Confirm ≥150 GB free disk on new PC | Disk check |
| 0.5 | Obtain HF token (READ scope) for MMFakeBench + BERT/CLIP auto-download | Token saved to `~/.cache/huggingface/token` on new PC |

#### Step 1 — TeamViewer-hybrid acquisition (Day 1, see Appendix A for full step-by-step)

| Source | Method | Size | Time |
|---|---|---|---|
| **DGM4** (all manipulation types) | TeamViewer File Transfer from current PC | ~80 GB | 4-9 h |
| **VisualNews** (full image set) | TeamViewer (zip-first recommended) | ~50 GB | 3-8 h |
| **GenImage** (all 8 generators) | TeamViewer + HF/Drive fallback for missing generators | ~10 GB | 1-2 h |
| **Qwen2-7B-Instruct** (HF cache) | TeamViewer Option A or HF download Option B | ~15 GB | 30 min - 25 min |
| **blur_jpg_prob0.pth** | TeamViewer or CNNDetection release download | ~100 MB | <1 min |
| **NewsCLIPpings annotations** | TeamViewer or Google Drive | ~200 MB | <5 min |
| **MMFakeBench** | HF download (`liuxuannan/MMFakeBench`) | ~3 GB | 5 min |
| **BERT + CLIP** | Auto-download on first use | ~1 GB | minutes |

**See Appendix A** for exact source/destination paths and per-dataset commands.

#### Step 2 — Build the 5-class manifest (after Step 1 raw data is in place)
| # | Command | Output |
|---|---|---|
| 2.1 | `python -m v4 build-manifest --source newsclippings` | Class 0, 1 rows |
| 2.2 | `python -m v4 build-manifest --source dgm4` | Class 2 + class 4 augmentation rows |
| 2.3 | `python -m v4 build-manifest --source mmfakebench` | Class 2/3/4 rows |
| 2.4 | `python -m v4 merge-manifest` | `data/processed/forensic_5class_v4.csv` (15,000 rows, stratified 70/15/15) |
| 2.5 | `python -m v4 leakage-audit` | Assertion-based audit log; must pass before training |
| 2.6 | `python -m v4 verify-image-paths` | Asserts every `image_path` resolves on disk |

#### Step 3 — Stage-1 binary CSV
| # | Command | Output |
|---|---|---|
| 3.1 | `python -m v4 build-manifest --source stage1` | `data/processed/univfd_stage1.csv` (~30k balanced binary) |
| 3.2 | Leakage audit on Stage-1 CSV | Audit log |

#### Step 4 — Precompute caches (Day 2)
| # | Cache | Encoder | Time on dual 4090 |
|---|---|---|---|
| 4.1 | `cache/v4/v_imgfor_dct/` | patch-DCT (CPU) | 30 min |
| 4.2 | `cache/v4/v_semantic_fnd/` | Stage 0 FND-CLIP | 1.5 h |
| 4.3 | `cache/v4/v_textfor_qwen/` | Qwen2-7B-Instruct | ~4 h |

### Disk-budget on the new PC
| Use | Size |
|---|---|
| Raw datasets (DGM4 + MMFB + NewsCLIPpings + GenImage subset + VisualNews subset + Qwen2-7B HF cache) | ~110 GB |
| Feature caches | ~5 GB |
| Trained checkpoints (Stage 0 + Stage 1 + 3-seed Stage 2) | ~3 GB |
| Eval artifacts | <100 MB |
| Reserve | 30 GB |
| **Total recommended free disk** | **~150 GB** |

---

## 4. Architecture

### Layered package layout

```
app/
  server.py                          FastAPI; reads server_config.yaml; uses registry
  server_config.yaml                 Encoders, fusion, ckpts named by registry key

src/v4/                              Top-level package
  core/
    registry.py                      ENCODER_REGISTRY, FUSION_REGISTRY, DATASET_REGISTRY (module-level dicts + @register decorator)
    checkpoints.py                   save_with_provenance, load_legacy_checkpoint, strip_prefix, shape_compat_filter
    config.py                        YAML loader + dataclass schema
    seed.py                          seed_all() — Python, NumPy, PyTorch, CUDA
    paths.py                         Centralized path resolution
    logging.py                       logging.getLogger config; INFO default
    class_map.py                     5-class label_to_name / name_to_label
  models/
    encoders/
      base.py                        EncoderBase ABC
      fnd_clip.py                    FND-CLIP V1 wrapper; sem_proj(512->768) adapter [F1]                       [Student 4 (Stage 0)]
      univfd.py                      ResNet50 + conv1 1ch + Kaiming + blur_jpg_v0 strict=False; 768-dim output  [Student 1]
      qwen_text.py                   Qwen2-7B-Instruct, last layer, masked-mean, Linear(3584, 768)+GELU [F2]    [Student 2]
    fusion/
      base.py                        FusionBase ABC
      v3_pairwise.py                 V3.1 §5: LN per stream -> 3-pair MHA-sum -> Conv1d -> AdaptiveAvgPool -> [B,1024]  [Student 3]
    classifier/
      mlp_head.py                    V3.1 §6 EXACT: Linear(1024,512)+BN+GELU+Dropout(0.5) -> Linear(512,256)+GELU -> Linear(256,5)  [Student 3]
  data/
    manifest.py                      Canonical CSV schema (sample_id, text, image_path, label, source, split)
    datasets/
      base.py                        MultimodalManifestDataset
      cached.py                      CachedFeatureDataset
      runtime.py                     Runtime preprocessing for new images (server use)
    builders/
      newsclippings.py               -> classes 0, 1
      dgm4.py                        -> class 2 (+ class 4 augmentation)
      mmfakebench.py                 -> classes 2, 3, 4
      genimage.py                    -> Stage 1 pretrain ONLY
      visualnews.py                  Image-source resolver
      merge.py                       Combine -> forensic_5class_v4.csv
      leakage_audit.py               Post-build audit (assertion-based)
    preprocessing/
      patch_dct.py                   cv2.dct (Windows-safe), ONE copy
      tokenizers.py                  BERT + CLIP + Qwen helpers
  training/
    trainer.py                       BaseTrainer (single class, parametrized by losses)
    losses.py                        CompositeLoss spec
    schedulers.py                    StepLR builder per spec
  evaluation/
    evaluator.py                     BaseEvaluator
    transfer.py                      MMFakeBench transfer probe (V3.1 §7)
    reporting.py                     Precision, Recall, F1-macro, confusion matrix
  cli.py                             python -m v4 (subcommands)
```

### Encoder base contract

```python
class EncoderBase(nn.Module):
    name: ClassVar[str]
    required_inputs: ClassVar[tuple[str, ...]]
    output_dim: int
    modality: ClassVar[Literal["text","image","multi"]]
    def forward(self, batch: dict[str, Tensor]) -> Tensor: ...
    def load_legacy_checkpoint(self, path: str) -> None: ...
```

### Fusion base contract

```python
class FusionBase(nn.Module):
    name: ClassVar[str]
    expected_inputs: ClassVar[tuple[str, ...]]
    num_classes: int
    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        # Returns {'main_logits': [B, C], 'aux_logits': [B, 2], 'fused': [B, 1024]}
        ...
```

### Registries (no plugin discovery)

```python
ENCODER_REGISTRY: dict[str, type[EncoderBase]] = {}
FUSION_REGISTRY:  dict[str, type[FusionBase]]  = {}
DATASET_REGISTRY: dict[str, type[Dataset]]     = {}

def register(reg, name):
    def deco(cls):
        if name in reg: raise ValueError(f"duplicate: {name}")
        reg[name] = cls; cls.name = name; return cls
    return deco
```

### Canonical YAML schema (example)

```yaml
seed: 42
deterministic: false                  # opt-in cudnn determinism
data:
  type: cached_features
  csv_path: data/processed/forensic_5class_v4.csv
  cache_root: cache/v4
  feature_keys:
    - { name: v_semantic, subdir: v_semantic_fnd,  dim: 512 }
    - { name: v_imgfor,   subdir: v_imgfor_dct,    dim: null }
    - { name: v_textfor,  subdir: v_textfor_qwen,  dim: 3584 }
encoders:
  semantic: { type: fnd_clip, ckpt: outputs/v4/stage0_fndclip/best.pt, feat_dim: 512 }
  image:    { type: univfd,   ckpt: outputs/v4/stage1_univfd/forensic_model.pth, out_dim: 768 }
  text:     { type: qwen2_7b, model_name: Qwen/Qwen2-7B-Instruct, hidden_size: 3584, layer_index: -1 }
fusion:
  type: v3_pairwise
  feat_dim: 768
  fused_dim: 1024
  num_classes: 5
  num_heads: 8
  attn_dropout: 0.1
  projections:
    v_semantic: { in_dim: 512,  out_dim: 768 }
    v_textfor:  { in_dim: 3584, out_dim: 768 }
train:
  epochs: 50
  batch_size: 64
  lr: 1.0e-4                          # spec §5.5
  weight_decay: 1.0e-4
  aux_weight: 0.1
  lr_step: 30
  lr_gamma: 0.1
  grad_clip: 1.0
  early_stop_patience: 10
  precision: bf16                      # interview pick
  out_dir: outputs/v4/stage2_fusion
```

---

## 5. Training pipeline

### Single `BaseTrainer`, parametrized by `CompositeLoss`

- **Stage 1**: `losses=[{type:bce, target:main_logits, weight:1.0}]`
- **Stage 2**: `losses=[{type:ce, target:main_logits, weight:1.0}, {type:bce, target:aux_logits, weight:0.1, detach_input:v_imgfor}]`

### Stage 0 — FND-CLIP V1 (binary OOC fine-tune)

- Per Dataset+V1 PDF: ResNet-50 + BERT + CLIP + modality attention + 2-layer MLP sigmoid
- Loss: BCEWithLogitsLoss
- Output: `outputs/v4/stage0_fndclip/best.pt`
- Wall-clock: ~3 h on one 4090

### Stage 1 — UnivFD binary pretrain (V3.1 §3.2)

- Conv1 modified to 1 ch, Kaiming Normal init; load blur_jpg_v0.pth with strict=False
- Data: §3 Stage 1 binary set above (~30k samples)
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW lr=1e-4
- Output: `outputs/v4/stage1_univfd/forensic_model.pth`
- Wall-clock: ~4 h on one 4090

### Stage 2 — V3 fusion + projections (V3.1 §5.5)

- Loss: CE(main, label) + 0.1 × BCE(aux, binary_image_label) with `v_imgfor.detach()`
- Optimizer: AdamW **lr = 1e-4** (corrected from V3's 5e-5)
- Weight decay = 1e-4, batch = 64
- LR scheduler: StepLR ×0.1 @ epoch 30
- Early stopping patience = 10 on val F1-macro
- Gradient clip max_norm = 1.0
- **3 seeds: 42, 1337, 2024**. Two in parallel via `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`, third sequential.
- **Precision: bf16** mixed-precision
- Resume: `latest.pt` saved every epoch with full state
- Best ckpt: `best.pt` on every val-F1-macro improvement
- Wall-clock: ~5 h total on dual 4090

### Checkpoint payload (every save)

```python
{
  "model_state": ...,
  "sem_proj_state": ...,             # Stage 2 only
  "text_proj_state": ...,            # Stage 2 only
  "optimizer_state": ...,            # latest.pt only
  "scheduler_state": ...,            # latest.pt only
  "rng_state": {"python", "numpy", "torch", "cuda"},   # latest.pt only
  "epoch": int,
  "val_metrics": {"f1_macro", "accuracy", "loss", "per_class_f1"},
  "config": <full YAML>,
  "config_hash": "<sha256>",
  "data_hash":   "<sha256>",
  "git_sha":     "<git rev-parse HEAD>",
  "seed": int,
  "torch_version": "...",
  "cuda_version":  "...",
  "stage": "stage0|stage1|stage2",
  "spec_version": "V3.1",
}
```

---

## 6. Evaluation

**Strict V3.1 §7 only** (per interview F lock). No bootstrap CI, no ablations, no shortcut audit.

### Mandatory (V3.1 §7)

1. Deactivate aux head at inference; apply softmax to main logits
2. Metrics: **Precision per class, Recall per class, F1-macro, single confusion matrix**
3. Evaluate on **MMFakeBench test set** (implemented as `source.startswith("MMFakeBench_")` filter on the 15% test split + separate convenience pass over MMFakeBench_test directory)

### Output artifacts

```
outputs/v4/eval/
  in_distribution_test/
    metrics.json
    confusion_matrix.png
    classification_report.txt
  mmfakebench_transfer/
    metrics.json
    confusion_matrix.png
    classification_report.txt
```

### Risks accepted (logged for doctor visibility)

- **R1 — No shortcut audit.** V3 lost weeks to undetected Class-4 leak. V4 skipping shortcut detection means any re-occurrence won't be caught until post-deployment. Documented in `docs/EVAL_PROTOCOL.md`.
- **R2 — Per-source bias undetectable beyond visual inspection** of per-source breakdown.

---

## 7. Deployment

### Server (Day 6 cutover)

- Same FastAPI structure as V3's `app/server.py`
- Same JSON response shape (UI compatibility)
- Registry-driven via `app/server_config.yaml`
- No Docker

### Day-6 parity migration

1. Train + evaluate V4 fully (Days 0-5)
2. Bring V4 server up on **port 8081**, V3 stays on **port 8080**
3. `scripts/compare_servers.py` sends 50 (text, image) pairs to both — verify JSON shape valid + sanity
4. Flip Cloudflare tunnel target to 8081 only after parity passes
5. Old `app/server.py` retained as `app/server_v3.py` for one release

---

## 8. Reproducibility

- `src/v4/core/seed.py::seed_all(seed)` — Python random, NumPy random, PyTorch CPU/CUDA random
- Strict cudnn determinism OFF by default (10% speed cost), opt-in via `train.deterministic: true`
- Every checkpoint carries: `config_hash, data_hash, git_sha, seed, torch_version, cuda_version, spec_version, stage`
- Data versioning via sha256 of the manifest CSV
- `python -m v4 reproduce --run <path>` verifies a past run

---

## 9. Build & code quality (handoff hygiene)

| Item | Choice |
|---|---|
| Packaging | `pyproject.toml` (PEP 517), Python `==3.12.*`, optional extras `[gpu, dev, test]`, entry points `v4-train`, `v4-eval`, `v4-server` |
| Lint + format | `ruff` (replaces black + isort + flake8) |
| Pre-commit | `.pre-commit-config.yaml` with ruff hook |
| CI | GitHub Actions: smoke tests + ruff lint on push (~2 min) |
| Logging | `logging.getLogger(__name__)` everywhere, INFO default. Replace 100% of V3's `print()` calls. |
| Type hints | `from __future__ import annotations` + type all public APIs; no mypy strict mode |

---

## 10. Testing (~3.5 h)

| Layer | Files | Purpose |
|---|---|---|
| Spec-compliance | `tests/spec_compliance/test_v3_1_section_*.py` (~10) | One per §-section: e.g. `test_spec_5_3_pairwise_sum.py` asserts fusion = dir1+dir2 element-wise |
| Shape/dtype | `tests/unit/test_*_shapes.py` | Encoder + fusion forward shape/dtype |
| Smoke | `tests/smoke/test_stage_smoke.py` | `python -m v4 train --max-epochs 1 --limit 128` per stage, <5 min |
| Integration | `tests/integration/test_full_pipeline.py` | manifest → trainer → eval on fixture data |
| Fixtures | `tests/fixtures/` | 5 synthetic samples + tiny random tensors |

---

## 11. Documentation (~5 h)

| File | Purpose |
|---|---|
| `README.md` (rewrite) | What is V4, how to install, how to run |
| `docs/ARCHITECTURE.md` | Layer diagram + responsibilities + V3.1 spec-section map |
| `docs/SETUP.md` | Windows + Python 3.12 + CUDA 12.4 install + manual DGM4/GenImage download steps |
| `docs/CONTRIBUTING.md` | Git workflow + branch naming + ruff/pre-commit + MANDATORY Windows workarounds (torch-before-cv2, cv2.dct, Start-Process detach, full python.exe path) |
| `docs/ADD_ENCODER.md` | Template walkthrough using UnivFD |
| `docs/ADD_DATASET.md` | Write a builder, produce canonical CSV, run precompute |
| `docs/ADD_FUSION.md` | Extend FusionBase, declare expected_inputs |
| `docs/EVAL_PROTOCOL.md` | How to run the suite, how to read metrics + R1/R2 risks |
| `docs/SPEC_COMPLIANCE_MAP.md` | V3.1 §-section → file:line table (doctor audit tool) |
| `docs/DECISIONS.md` | Record of every major design decision (this interview as source) |
| `CHANGELOG.md` | V1 → V2 → V3 → V4 timeline |

Plus inline Google-style docstrings on all public APIs.

---

## 12. Migration

### V3 ckpt cleanup

- **Keep all 14 ckpts during V4 build** (safety net)
- **After V4 ships and doctor approves**: `git tag v3-archive`, then prune orphans (~1.3 GB):
  - `outputs/v1_leakfree_v2`
  - `v3/outputs/stage2_5class*`
  - `v3/outputs/_smoke*`
  - `v3/outputs/v3_pipeline` (RoBERTa)
  - `v3/outputs/v3_pipeline_4class`
  - `v3/outputs/v3_pipeline_c4fix`
  - `v3/outputs/v3_pipeline_lr5e5`
  - `v3/outputs/v3_pipeline_qwen_l14`
  - `v3/outputs/v3_pipeline_qwen_c4fix`
- **Never delete**: `outputs/v1_leakfree/best.pt`, `outputs/univfd_genimage/best.pt`, `v3/outputs/v3_pipeline_qwen/best.pt`

### New-PC handoff

1. `git clone … && git checkout v4`
2. `make setup` → `pip install -e ".[gpu,dev]"` + `huggingface-cli login`
3. Manual: download DGM4 + GenImage per `docs/SETUP.md`
4. `make data` → `download-data` + `build-manifest` + `prepare-image-subset` + leakage audit
5. `make smoke` → `pytest tests/smoke/ -v`
6. `make train` → Stage 0 → 1 → 2 sequentially
7. `make eval` → full §7 evaluation suite

---

## 13. One-week schedule

| Day | Stage | Deliverable | Hours |
|---|---|---|---|
| **Day 0** | Scaffolding | Package layout, base classes, registries, `core/{seed,checkpoints,config,paths,logging,class_map}.py`, shared `preprocessing/{patch_dct,tokenizers}.py`, pyproject.toml + ruff + pre-commit + CI. Smoke tests per registry. **Verify V3 test F1 number** against on-disk artifacts. | 6 |
| **Day 1** | Data | All dataset builders + manifest schema + stratified 70/15/15 split + `leakage_audit.py` + `download_data.py` + `prepare_image_subset.py` + build `forensic_5class_v4.csv` | 6 |
| **Day 2** | Encoders + caches | Implement `FNDCLIPSemanticEncoder`, `UnivFDForensicEncoder`, `Qwen2TextForensicEncoder`. Precompute caches: `v_semantic_fnd`, `v_imgfor_dct`, `v_textfor_qwen`. Stage 0 fine-tune FND-CLIP. | 8 |
| **Day 3** | Stage 1 | UnivFD binary pretrain on Stage-1 set. Save `forensic_model.pth`. Eval Stage 1 binary metrics. | 5 |
| **Day 4** | Stage 2 | V3 fusion training: 3 seeds (42, 1337, 2024), two in parallel on dual 4090, bf16. Save best.pt + provenance per seed. | 5 |
| **Day 5** | Eval | Full §7 evaluation per seed: in-distribution + MMFakeBench transfer. Report mean ± std. Per-source breakdown. Pick best-seed ckpt. | 5 |
| **Day 6** | Server + docs | Rewrite `app/server.py` registry-driven + `app/server_config.yaml`. Side-by-side parity 8081 vs 8080. Write 10 markdown docs + README rewrite + DECISIONS.md from this interview. | 6 |
| **Day 7** | Buffer | Doctor handoff prep: SPEC_COMPLIANCE_MAP.md verification, writeup, archive orphans, final cleanup. | variable |

---

## 14. Critical files

### To create
- `src/v4/core/{registry,checkpoints,config,seed,paths,logging,class_map}.py`
- `src/v4/models/encoders/{base,fnd_clip,univfd,qwen_text}.py`
- `src/v4/models/fusion/{base,v3_pairwise}.py`
- `src/v4/models/classifier/mlp_head.py`
- `src/v4/data/manifest.py`, `src/v4/data/datasets/{base,cached,runtime}.py`
- `src/v4/data/builders/{newsclippings,dgm4,mmfakebench,genimage,visualnews,merge,leakage_audit}.py`
- `src/v4/data/preprocessing/{patch_dct,tokenizers}.py`
- `src/v4/training/{trainer,losses,schedulers}.py`
- `src/v4/evaluation/{evaluator,transfer,reporting}.py`
- `src/v4/cli.py`
- `app/server.py` (rewritten), `app/server_config.yaml`
- `configs/v4_pipeline_qwen.yaml`, `configs/v4_pipeline_stage0.yaml`, `configs/v4_pipeline_stage1.yaml`
- `docs/{ARCHITECTURE,SETUP,CONTRIBUTING,ADD_ENCODER,ADD_DATASET,ADD_FUSION,EVAL_PROTOCOL,SPEC_COMPLIANCE_MAP,DECISIONS}.md`
- `README.md` (rewrite), `CHANGELOG.md`
- `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `Makefile`
- `tests/spec_compliance/`, `tests/unit/`, `tests/smoke/`, `tests/integration/`, `tests/fixtures/`
- `scripts/{download_data,download_pretrained,prepare_image_subset,compare_servers}.py`

### Existing files referenced (read-only)
- `src/models/fnd_clip.py` — V1 architecture reference
- `src/models/v3_pipeline.py` — V3 fusion reference
- `src/models/univfd_encoder.py` — UnivFD reference
- `v3/scripts/precompute_patch_dct.py` — patch-DCT reference
- `v3/scripts/precompute_qwen_textfor_v3_1.py` — Qwen extraction reference
- `app/server.py` — current server reference
- `C:\Users\FSOS\Downloads\Implementation Guidelines V3_1.pdf` — doctor's V3.1 spec (canonical)
- `C:\Users\FSOS\Downloads\Implementation Guidelines (V2) (1).pdf` — doctor's V2 spec
- `C:\Users\FSOS\Downloads\Dataset + V1 (instructions).pdf` — doctor's V1 spec

---

## 15. Reusability assertions (doctor's brief)

After V4 is built, a future student can:
1. **Swap text encoder** (Qwen → RoBERTa → DeBERTa): change `encoders.text.type` in YAML + `feature_keys.v_textfor.subdir` + precompute new cache + retrain Stage 2. New file: `src/v4/models/encoders/<encoder>.py` (≤80 LOC).
2. **Swap image encoder**: change `encoders.image.type` + precompute + retrain.
3. **Add a new dataset**: one builder script (≤100 LOC) emitting canonical-schema CSV.
4. **Reproduce a past run**: `python -m v4 reproduce --run <path>` via provenance metadata.
5. **Change num_classes** (e.g. 5→3): edit `class_map.py` AND `mlp_head.py` (the latter because we locked V3.1 §6 EXACT — see D3/D6 in §17).

---

## 16. Verification matrix

| Layer | What | Frequency |
|---|---|---|
| Spec-compliance unit tests | ~10 tests | Every push (CI) |
| Shape/dtype tests | Encoders + fusion | Every push (CI) |
| Smoke tests | 1-epoch on 128 samples per stage | Every push (CI) |
| Integration test | Full pipeline on fixtures | Every push (CI) |
| Server side-by-side | 8080 (V3) vs 8081 (V4) JSON sanity | Day 6 (manual) |
| Doctor review gate | `SPEC_COMPLIANCE_MAP.md` audit | Day 7 (manual) |

---

## 17. Locked decisions register (DECISIONS.md source)

| Branch | Decision | Choice | Reason |
|---|---|---|---|
| A1 Team | Code organization | Solo implementer, 4-layer structure preserved | Future handoff to original team possible |
| A3 Env | Implementation PC | Windows + Python 3.12 + CUDA 12.4 + dual 4090 | Same as dev PC; carries Windows workarounds |
| A4/A5 Git | Review + branches | Doctor sole reviewer, v4 branch from main, per-stage feature branches | Squash-merge each stage after approval |
| B2 Qwen | Variant | Qwen2-7B-Instruct, last layer (-1), 3584 dim | V3-proven; spec's 30/4096 matches Qwen1.5-7B |
| C1 Size | Per-class count | 3,000/class = 15,000 total | Matches V3 scale; DGM4 augments class 4 |
| C3 MMFB | Partition policy | All MMFB rows in dataset; per-source breakdown for OOD | Matches V3 effective behavior |
| C4 Split | Stratification | Per-class, per-source stratified 70/15/15 | Eliminates source-fingerprint leak |
| C5/C6 Data ops | New-PC procedure | **Superseded by C9 (TeamViewer-hybrid)**. See C9 below for current method. | Originally planned full-HF; switched to TeamViewer-hybrid when TeamViewer access confirmed |
| C8 GenImage | Subset for Stage 1 | All 8 official GenImage generators (SD v1.4, SD v1.5, MidJourney, ADM, GLIDE, VQDM, BigGAN, Wukong), ~1,250 each = 10k | Max generator diversity for UnivFD generalization |
| C9 Handoff | Bulk data transfer between PCs | **TeamViewer-hybrid**: TeamViewer File Transfer for DGM4 + GenImage + VisualNews + Qwen HF cache + blur_jpg (the bulk + manual ones); HF auto-download for MMFakeBench + NewsCLIPpings annotations + BERT/CLIP warming | User has TeamViewer access to new PC. Saves DGM4 HF auth dance, GenImage manual click-through, VisualNews Google Drive quota issues. Current PC must stay awake during transfer (~9 h overnight). |
| D3/D6 Classifier | Flexibility | Strict V3.1 §6, hardcoded, no flexibility hooks | User chose conservative spec-match |
| E Training | Pipeline | Single BaseTrainer, single-GPU per stage, bf16, 3 seeds, resume support | Spec fidelity + variance + dual-4090 usage |
| F Eval | Scope | Strict V3.1 §7: P, R, F1-macro, CM only. No CI, no ablations, no shortcut audit | User chose conservative spec-match. R1/R2 logged. |
| G Deploy | Migration | Day-6 server migration + side-by-side parity. No Docker. | Validates V4 end-to-end |
| H Repro | Provenance | Full provenance metadata. Opt-in cudnn determinism. | ~30 LOC; saves pain later |
| I Build | Hygiene | pyproject + ruff + pre-commit + CI + logging + type hints | Doctor's brief is about handoff |
| J Tests | Scope | Spec-compliance + shape + smoke + integration (~3.5 h) | Matches I full hygiene |
| K Docs | Depth | 10 markdown files + README rewrite + inline docstrings | Future-student onboarding; doctor audit |
| L Cleanup | Ckpt handling | Keep all ckpts now; archive orphans to git tag after V4 ships | Safety net during V4 build |

---

## 18. Risk register

| ID | Risk | Mitigation | Watch for |
|---|---|---|---|
| R1 | Undetected shortcut (V3 class-4 caption leak) | Source stratification reduces leak surface but no detector | F1 on any class >0.90 unexpectedly |
| R2 | Per-source bias | Stratified split eliminates train/test source overlap | Per-source breakdown variance >0.15 |
| R3 | Doctor changes spec mid-build | Frozen at the three PDFs read on Day 0 | Re-audit if new spec arrives |
| R4 | TeamViewer File Transfer fails / disconnects on the long DGM4/VisualNews transfers | Pre-flight Step 0.1 tests TeamViewer; Step 0.2 disables sleep on both PCs. Fallback: re-attempt with resume; if persistent, switch to HF download for DGM4 (Appendix A.5.2 alt) | Resume attempts > 3 → switch to HF for that dataset |
| R7 | Current PC slept mid-transfer | Power plan reverted to default | Pre-flight Step 0.2 disables sleep. Add `powercfg /requestsoverride DISPLAY SYSTEM` for belt-and-braces. Monitor TeamViewer session during overnight runs. | Connection state checked at hour 1 and hour 4 |
| R5 | Class 3 too small even with augmentation | No DGM4 substitute exists; capped at MMFakeBench AI-text | If F1 noisy → consider lowering target to 2,200/class |
| R6 | Windows-specific issues recur on new PC | CONTRIBUTING.md captures V3 workarounds as MANDATORY | Smoke test on Day 0 must pass |

---

## 19. Open items requiring confirmation before Day 0

- [ ] **Verify V3 test F1-macro number** (read `v3/outputs/v3_pipeline_qwen/` artifacts on Day 0). The 0.5377 vs 0.7215 ambiguity must be resolved before V4's PASS threshold is defined.
- [ ] **Confirm Qwen2-7B-Instruct cached on new PC** (~15 GB HuggingFace cache).
- [ ] **Confirm dual 4090 NCCL works** (only needed if DDP ever required; recommended single-GPU plan avoids).

---

## 20. What this plan explicitly does NOT do

- No new architecture invention (use V3.1's exactly)
- No Optuna / Slurm / TorchServe / MLflow / W&B
- No Docker (defer to future contributors)
- No reuse of V1/V2/V3 checkpoints — fresh training per purist choice
- No GenImage in the 5-class training data (Stage 1 only)
- No shortcut-detection harness (R1 logged)
- No bootstrap CI
- No structural ablation studies
- No mypy strict mode
- No Sphinx/mkdocs site
- No task-flexibility hooks in classifier
- **No implementation in this session** — plan only; implementation happens on the other PC

---

# Appendix A — Dataset acquisition (TeamViewer-hybrid recipe)

Extract this into the repo as `docs/DATA_DOWNLOAD.md`. Run it on the new PC BEFORE any V4 code lands.

**Strategy:** TeamViewer File Transfer from the current PC (which already has all the bulk data) for the big/manual datasets (DGM4, VisualNews, GenImage, Qwen HF cache, blur_jpg). HuggingFace auto-download for the small/clean ones (MMFakeBench, BERT, CLIP). Saves manual web-UI steps and avoids Google Drive quota issues; requires the current PC to be awake during transfer (~9 hours overnight) plus ~10 min of HF downloads.

## A.0 — Prerequisites

| Requirement | How to verify |
|---|---|
| Python ≥ 3.10 on new PC PATH | `python --version` |
| ≥ 150 GB free disk on new PC data drive | `Get-PSDrive C` (PowerShell) or `df -h .` (Linux) |
| TeamViewer connected to current PC with **File Transfer** mode | TeamViewer → Files & Extras → File Transfer |
| Current PC stays AWAKE during the bulk transfer | Settings → Power → Screen and Sleep → Never on current PC |
| HuggingFace account (still needed for MMFakeBench + BERT/CLIP) | https://huggingface.co/join |
| Internet ≥ 10 Mbps on new PC (small HF downloads only) | `speedtest-cli` |

## A.1 — Install tools (one-time)

```bash
pip install --upgrade "huggingface_hub[cli]" tqdm
```

## A.2 — HuggingFace authentication

```bash
huggingface-cli login
# Paste a token with READ access. Create at https://huggingface.co/settings/tokens.
# The token persists in ~/.cache/huggingface/token. Do NOT commit it.
```

## A.3 — TeamViewer File Transfer setup

1. On the new PC, open TeamViewer.
2. Enter the current PC's TeamViewer ID and choose **"File Transfer"** (NOT Remote Control). Or: connect normally, then Files & Extras → Open File Transfer.
3. In the file-transfer window: **left pane = new PC (destination)**, **right pane = current PC (source)**.
4. Source root for all bulk transfers below: `C:\Desktop\Multimodal-fake-news-detection\data\raw\` on the current PC.
5. Destination root: `C:\path\to\MultiGuard\data\raw\` on the new PC (the cloned V4 repo).

**Throughput tips:**
- TeamViewer relay tops out at ~5 MB/s. Direct P2P (LAN or same NAT) reaches 20+ MB/s — TeamViewer auto-negotiates this.
- Start one big folder transfer and leave it overnight. Disable sleep on BOTH PCs.
- TeamViewer File Transfer supports resume — if interrupted, restart and accept "Resume" when prompted.

**Note:** because we're transferring DGM4 directly from the current PC, **DGM4 HuggingFace terms acceptance is NOT required** in this hybrid recipe. Skip it.

## A.4 — Create the workspace

```bash
mkdir -p data/raw
cd data/raw
```

After this section runs, `data/raw/` will contain:

```
data/raw/
├── DGM4/                  ~80 GB
├── MMFakeBench/           ~3 GB
├── NewsCLIPpings/         ~200 MB (annotations)
├── visualnews/            ~3-50 GB (images, subset or full)
├── GenImage/              ~10 GB (10k subset across 8 generators)
└── pretrained/
    └── blur_jpg_prob0.pth ~100 MB

# Plus, outside data/raw/ but referenced:
~/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/  ~15 GB
```

## A.5 — Per-dataset downloads (can run in parallel in separate terminals)

### A.5.1 — NewsCLIPpings annotations (~200 MB) — TeamViewer or manual

The current PC may already have these from V3. **TeamViewer-transfer if found** at e.g. `C:\Desktop\Multimodal-fake-news-detection\data\raw\NewsCLIPpings\`.

Fallback (if not on current PC):
1. Visit https://github.com/g-luo/news_clippings
2. Follow the **"Annotations"** section's Google Drive link
3. Download `news_clippings.tar.gz`
4. Extract to `data\raw\NewsCLIPpings\`:
   ```bash
   cd data/raw/NewsCLIPpings
   tar -xzf news_clippings.tar.gz
   ```

Expected: `NewsCLIPpings/news_clippings/data/{merged_balanced,...}.json`

### A.5.2 — DGM4 (~80 GB) — TeamViewer, ~4–9 hours

- **Source on current PC:** `C:\Desktop\Multimodal-fake-news-detection\data\raw\DGM4\`
- **Destination on new PC:** `C:\path\to\MultiGuard\data\raw\DGM4\`

Action:
1. In TeamViewer File Transfer, navigate source to the `DGM4` directory.
2. Drag the entire `DGM4` directory from source pane → destination pane.
3. Leave overnight. ~4-9 hours depending on TeamViewer throughput.
4. If interrupted, restart TeamViewer File Transfer — it will offer to resume.

Verify after transfer (PowerShell on new PC):
```powershell
$size = (Get-ChildItem data\raw\DGM4 -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("DGM4: {0:N1} GB" -f $size)   # expect ~80
(Get-ChildItem data\raw\DGM4\manipulation\HFGI | Measure-Object).Count   # > 50000
```

### A.5.3 — MMFakeBench (HF, public) — ~3 GB, ~5 min

```bash
huggingface-cli download liuxuannan/MMFakeBench \
    --repo-type dataset \
    --local-dir MMFakeBench \
    --local-dir-use-symlinks False
```

Verify:
```bash
du -sh MMFakeBench/             # ~3 GB
ls MMFakeBench/                 # MMFakeBench_val/  MMFakeBench_test/
ls MMFakeBench/MMFakeBench_val/fake/ | head     # subset directories
```

### A.5.4 — GenImage 10k subset — TeamViewer + HF/Drive fallback

- **Source on current PC:** `C:\Desktop\Multimodal-fake-news-detection\data\raw\GenImage\`
- **Destination on new PC:** `C:\path\to\MultiGuard\data\raw\GenImage\`

Action:
1. TeamViewer-transfer the `GenImage` directory.
2. Verify which generators came across:
   ```powershell
   Get-ChildItem data\raw\GenImage | Select-Object Name
   # Goal: 8 subdirectories — sd14, sd15, midjourney, adm, glide, vqdm, biggan, wukong
   ```
3. V3 used MidJourney heavily; the current PC may only have some of the 8 generators on disk. For **any missing generators**, fall back to one of:
   - Official GenImage benchmark Google Drive at https://github.com/GenImage-Dataset/GenImage
   - HuggingFace community mirror (verify URL before running):
     ```bash
     huggingface-cli download <community/genimage-<generator>-mirror> \
         --repo-type dataset --local-dir data/raw/GenImage/<generator>
     ```
4. V4 builder (Day 1) will subsample to 1,250 per generator for the 10k Stage 1 set.

Expected:
```bash
du -sh data/raw/GenImage/       # ~10 GB total once all 8 generators are present
```

### A.5.5 — VisualNews images (~50 GB) — TeamViewer, ~3–8 hours

- **Source on current PC:** `C:\Desktop\Multimodal-fake-news-detection\data\raw\visualnews\`
- **Destination on new PC:** `C:\path\to\MultiGuard\data\raw\visualnews\`

Action:
1. TeamViewer-transfer the entire `visualnews` directory.
2. **Performance warning:** VisualNews has hundreds of thousands of small JPEGs. Per-file overhead on TeamViewer is high.
3. **Recommended tip — zip first:** on the current PC, compress to a single `.zip`, then transfer one file:
   ```powershell
   # On current PC (PowerShell), before starting TeamViewer:
   Compress-Archive -Path C:\Desktop\Multimodal-fake-news-detection\data\raw\visualnews `
                    -DestinationPath C:\temp\visualnews.zip
   ```
4. TeamViewer-transfer `C:\temp\visualnews.zip` (one big file, much faster than 100k small files).
5. On new PC:
   ```powershell
   Expand-Archive -Path C:\temp\visualnews.zip -DestinationPath data\raw\
   ```

Expected after extract:
```powershell
Get-ChildItem data\raw\visualnews\origin | Select-Object Name
# Should include: guardian  usa_today  bbc  washington_post  ...
```

### A.5.6 — blur_jpg_prob0.pth (~100 MB) — TeamViewer or download

The current PC may have this from V3 — search for `blur_jpg_prob0.pth` in `outputs\`, `pretrained\`, or near `univfd_genimage\`. If found, **TeamViewer-transfer** to `data\raw\pretrained\blur_jpg_prob0.pth`.

Fallback if not on current PC, download from CNNDetection releases:
```powershell
mkdir data\raw\pretrained -Force
cd data\raw\pretrained
Invoke-WebRequest -Uri "https://github.com/PeterWang512/CNNDetection/releases/download/v1.0/blur_jpg_prob0.pth" -OutFile blur_jpg_prob0.pth
```

If the release-asset URL doesn't resolve, see https://github.com/PeterWang512/CNNDetection README for the current location.

Verify:
```powershell
Get-Item data\raw\pretrained\blur_jpg_prob0.pth | Select-Object Length   # ~100 MB
```

### A.5.7 — Qwen2-7B-Instruct (~15 GB) — TeamViewer (Option A) or HF (Option B)

**Option A — TeamViewer copy from current PC's HF cache (faster):**

- **Source on current PC:** `C:\Users\FSOS\.cache\huggingface\hub\models--Qwen--Qwen2-7B-Instruct\` (whole directory tree)
- **Destination on new PC:** `C:\Users\<your-username>\.cache\huggingface\hub\models--Qwen--Qwen2-7B-Instruct\`

⚠️ **Windows symlink warning:** HuggingFace cache uses symlinks from `snapshots/<hash>/*.safetensors` to `blobs/*`. TeamViewer File Transfer **may not preserve Windows symlinks**. After copying, smoke-test:
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct'); print('OK')"
```
If you get "file not found", the symlinks broke — use Option B.

**Option B — HF download on new PC (cleaner, slower, ~25 min):**
```bash
huggingface-cli download Qwen/Qwen2-7B-Instruct
```

**Recommended order:** try Option A first; fall back to Option B if symlinks fail.

Verify:
```powershell
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen2-7B-Instruct" -Recurse | Measure-Object -Property Length -Sum | ForEach-Object { "{0:N1} GB" -f ($_.Sum/1GB) }
# ~15 GB
```

### A.5.8 — BERT + CLIP (auto-downloaded on first use, no action needed)

- `bert-base-uncased` (~440 MB)
- `openai/clip-vit-base-patch32` (~600 MB)

These download to `~/.cache/huggingface/hub/` the first time the V4 FND-CLIP encoder is instantiated. No pre-download required, but you can warm them now:

```bash
python -c "from transformers import AutoTokenizer, CLIPProcessor; \
           AutoTokenizer.from_pretrained('bert-base-uncased'); \
           CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
```

## A.6 — Master verification

After all downloads complete, run this one-liner to sanity-check the layout:

```bash
cd /path/to/project/root

for d in DGM4 MMFakeBench NewsCLIPpings visualnews GenImage pretrained; do
  if [ -d "data/raw/$d" ]; then
    size=$(du -sh "data/raw/$d" 2>/dev/null | cut -f1)
    echo "  OK    data/raw/$d   $size"
  else
    echo "  MISS  data/raw/$d"
  fi
done

[ -f data/raw/pretrained/blur_jpg_prob0.pth ] && echo "  OK    blur_jpg_prob0.pth" || echo "  MISS  blur_jpg_prob0.pth"
[ -d ~/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct ] && echo "  OK    Qwen2-7B-Instruct in HF cache" || echo "  MISS  Qwen2-7B-Instruct"
```

Expected output: 7 lines all starting with `OK`.

## A.7 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| TeamViewer file transfer freezes mid-way | Network instability or sleep mode on either PC | Disable sleep on BOTH PCs. Reconnect TeamViewer File Transfer and accept "Resume". |
| TeamViewer < 1 MB/s | Going through relay (not P2P) | Verify both PCs on same network if possible; try different time of day. Or accept it — overnight transfers still finish. |
| Current PC slept during transfer | Default power plan triggered | Settings → Power → Screen and Sleep → "Never" while connected to AC. Optionally `powercfg /requestsoverride DISPLAY SYSTEM` |
| Qwen2-7B "files not found" after TeamViewer copy | HF cache symlinks broke in transit | Switch to A.5.7 Option B (HF download) |
| GenImage doesn't have all 8 generators on current PC | V3 only used some generators | Fall back to official GenImage Google Drive or community HF mirror for missing ones (A.5.4) |
| VisualNews transfer takes forever (per-file overhead) | Many tiny JPEGs over TeamViewer | Use the zip-first approach in A.5.5 |
| Out of disk during DGM4 transfer | DGM4 is ~80 GB | Free disk first. If desperate, drop `face_swap`/`face_edit` subdirs (keep `face_attribute`/HFGI for class 2) |
| HF download "401 Unauthorized" on MMFakeBench | HF token expired/missing | `huggingface-cli login` again with a fresh READ token |
| `blur_jpg_prob0.pth` 404 | CNNDetection release URL changed | Check the current URL at https://github.com/PeterWang512/CNNDetection README |

## A.8 — When you're done

You've successfully completed A.0–A.7. The new PC now has:
- All 5 raw datasets in `data/raw/`
- Pretrained UnivFD weights at `data/raw/pretrained/blur_jpg_prob0.pth`
- Qwen2-7B-Instruct in the HF cache

You can now clone the V4 repo (which will be empty initially with just the plan + this download recipe) and start Day 0 of V4 implementation per §13 of the plan.

The V4 builder scripts (Day 1, §3 "Dataset acquisition procedure") will then read this `data/raw/` layout and produce the Stage-1 binary CSV and the 5-class `forensic_5class_v4.csv` manifest.
