# MultiGuard V4 — Architecture

This document describes the layered package, the data flow at training and serving time, and where each V3.1 spec section lives in code.

For the full V3.1 to V4 mapping with file:line citations, see [`SPEC_COMPLIANCE_MAP.md`](SPEC_COMPLIANCE_MAP.md).

---

## 1. Layered package

```
app/
  server.py                          FastAPI inference server (registry-driven)
  server_config.yaml                 Encoder + fusion specs + ckpt paths
  static/                            Frontend assets (optional)

src/v4/
  core/                              Cross-cutting infrastructure
    registry.py                      ENCODER/FUSION/DATASET dicts + @register decorator + import_all()
    checkpoints.py                   save_checkpoint, load_checkpoint, build_provenance, strip_prefix
    config.py                        V4Config dataclass + load_config + config_hash
    seed.py                          seed_all() - Python/NumPy/torch/CUDA
    paths.py                         PROJECT_ROOT, DATA_ROOT, CACHE_ROOT, OUTPUTS_ROOT resolution
    logging.py                       get_logger(__name__) + configure(level)
    class_map.py                     LABELS, LABELS_AR, IMAGE_FAKE_CLASSES, binary_image_label

  models/                            Trainable building blocks
    encoders/
      base.py                        EncoderBase ABC (required_inputs, output_dim, modality)
      fnd_clip.py                    FND-CLIP V1 (ResNet50 + BERT + CLIP + modality attn) + sem_proj(512->768)
      univfd.py                      ResNet50 + conv1(1ch, Kaiming init) + blur_jpg_v0 strict=False, out=768
      qwen_text.py                   Qwen2-7B-Instruct, last layer, masked-mean pool + Linear(3584->768)+GELU
    fusion/
      base.py                        FusionBase ABC (expected_inputs, num_classes)
      v3_pairwise.py                 LN per stream -> 3-pair MHA-SUM -> Conv1d stack -> AdaptiveAvgPool -> [B,1024]
    classifier/
      mlp_head.py                    Linear(1024,512)+BN+GELU+Dropout(0.5) -> Linear(512,256)+GELU -> Linear(256,5)

  data/
    manifest.py                      Canonical CSV: (sample_id, text, image_path, label, source, split)
    datasets/
      base.py                        MultimodalManifestDataset (subclasses override load_sample)
      cached.py                      CachedFeatureDataset (reads cache/v4/<subdir>/<sample_id>.pt)
      runtime.py                     RuntimeImageDataset (computes patch-DCT + FND-CLIP inputs live)
    builders/
      build_newsclippings.py         -> classes 0, 1
      build_dgm4.py                  -> class 2 + class 4 augmentation (capped 800)
      build_mmfakebench.py           -> classes 2/3/4 via SUBSET_LABEL_MAP
      build_genimage_stage1.py       -> Stage 1 binary pretrain ONLY (never enters 5-class)
      build_stage1_binary.py         -> ~30k balanced binary CSV
      merge.py                       undersample to 3000/class + stratified 70/15/15 split by (label, source)
      leakage_audit.py               duplicate sample_id + path/label collision checks + path verification
    preprocessing/
      patch_dct.py                   cv2.dct on 8x8 Y-channel patches -> [1, 224, 224]  (Windows-safe)
      tokenizers.py                  BERT + CLIP + Qwen tokenizer/processor helpers + masked_mean_pool

  training/
    losses.py                        CompositeLoss (CE on main + 0.1xBCE on aux with detach_input)
    schedulers.py                    build_scheduler (steplr x0.1 @ epoch 30, cosine, none)
    trainer.py                       BaseTrainer (bf16 autocast, latest.pt resume, best.pt on val-F1)

  evaluation/
    reporting.py                     compute_metrics, write_confusion_matrix_png, classification_report
    evaluator.py                     BaseEvaluator (writes metrics.json + per_source_metrics.json)
    transfer.py                      run_transfer_probe (MMFakeBench OOD evaluation)

  cli.py                             python -m v4 + entry points: v4-train, v4-eval, v4-server, ...
  __main__.py                        Enables `python -m v4 <subcommand>`

scripts/
  precompute.py                      Build per-modality .pt feature shards
  download_pretrained.py             blur_jpg_prob0.pth from CNNDetection
  download_data.py                   MMFakeBench from HuggingFace
  prepare_image_subset.py            Copy only manifest-referenced VisualNews images

configs/
  v4_pipeline_qwen.yaml              Stage 2 fusion training (3 encoders + V3PairwiseFusion)
  v4_pipeline_stage0.yaml            FND-CLIP fine-tune
  v4_pipeline_stage1.yaml            UnivFD binary pretrain
```

---

## 2. Data flow

### 2.1 Training (Stage 2, cached features)

```
forensic_5class_v4.csv
   |
   v
CachedFeatureDataset (loads cache/v4/<subdir>/<sample_id>.pt per feature_key)
   |
   v
DataLoader (collate_cached, batch=64)
   |
   v
V3PairwiseFusion(features) -> {main_logits, aux_logits, fused}
   |
   v
CompositeLoss: CE(main, label) + 0.1 x BCE(aux, binary_image_label(label))
   |
   v
AdamW (lr=1e-4, wd=1e-4)  ->  StepLR x0.1 @ epoch 30  ->  ES patience 10 (val F1-macro)
   |
   v
outputs/v4/stage2_fusion/seed_<S>/best.pt + latest.pt + provenance
```

### 2.2 Serving (FastAPI, runtime preprocessing)

```
POST /api/analyze  {text, image}
   |
   |    pil_img = Image.open(image).convert("RGB")
   |
   +---> compute_patch_dct_from_pil(pil_img)         ->  [1, 1, 224, 224]
   +---> prepare_fnd_inputs(text, pil_img)            ->  BERT ids + CLIP pixels
   +---> Qwen tokenizer (in encoder.runtime_forward)  ->  [1, 3584]
   |
   v
encoders.{semantic, image, text}  (all torch.no_grad, .eval(), frozen)
   |
   v
V3PairwiseFusion(features) -> main_logits[1, 5]
   |
   v
softmax -> JSON {verdict, verdict_ar, confidence, probabilities, modules, explanation, explanation_ar}
```

### 2.3 Stage 1 (UnivFD binary pretrain) and Stage 0 (FND-CLIP fine-tune)

Same `BaseTrainer` parametrized via `CompositeLoss`:

| Stage | Loss spec |
|-------|-----------|
| Stage 0 | `[{type: bce, target: main_logits, weight: 1.0}]` |
| Stage 1 | `[{type: bce, target: main_logits, weight: 1.0}]` |
| Stage 2 | `[{type: ce, target: main_logits, weight: 1.0}, {type: bce, target: aux_logits, weight: 0.1, detach_input: v_imgfor}]` |

---

## 3. Registry pattern

`v4.core.registry` defines three module-level dicts. Importing `import_all()` walks `src/v4/models/` and `src/v4/data/` to trigger the `@register(...)` decorators.

```python
from v4.core.registry import register, ENCODER_REGISTRY, build_encoder

@register(ENCODER_REGISTRY, "univfd")
class UnivFDEncoder(EncoderBase):
    ...

# Later:
enc = build_encoder({"type": "univfd", "ckpt": "..."})
```

The server and trainer both go through `build_encoder` / `build_fusion`, so adding a new encoder requires:
1. Drop a new file under `src/v4/models/encoders/`.
2. Subclass `EncoderBase`.
3. Decorate with `@register(ENCODER_REGISTRY, "<key>")`.
4. Reference `<key>` in YAML.

No discovery magic - every new module name must be imported by `core.registry.import_all` either directly or via the package `__init__.py`.

---

## 4. V3.1 spec section to file mapping (summary)

| V3.1 section | What it specifies | V4 file |
|--------------|-------------------|---------|
| Sec 1 (modalities) | v_semantic, v_imgfor, v_textfor; 768-dim each | `core/class_map.py`, `models/encoders/*` |
| Sec 3.1 (FND-CLIP) | ResNet50 + BERT + CLIP + modality attn | `models/encoders/fnd_clip.py` |
| Sec 3.2 (UnivFD) | ResNet50, 1-ch conv1, Kaiming, blur_jpg init | `models/encoders/univfd.py` |
| Sec 4 (Qwen text) | layer pool + Linear+GELU adapter | `models/encoders/qwen_text.py` |
| Sec 5 (fusion) | LN -> 3-pair MHA-SUM -> Conv1d -> AvgPool | `models/fusion/v3_pairwise.py` |
| Sec 5.5 (training) | AdamW lr=1e-4 wd=1e-4, StepLR, ES patience 10 | `configs/*.yaml` + `training/trainer.py` |
| Sec 6 (classifier) | 1024->512+BN+GELU+Drop(0.5) -> 256+GELU -> 5 | `models/classifier/mlp_head.py` |
| Sec 7 (eval) | Precision, Recall, F1-macro, CM, MMFakeBench probe | `evaluation/evaluator.py`, `evaluation/transfer.py` |

The exhaustive line-level mapping lives in [`SPEC_COMPLIANCE_MAP.md`](SPEC_COMPLIANCE_MAP.md).

---

## 5. Deviations from spec (forced by reality)

| ID | Issue | Resolution | File |
|----|-------|-----------|------|
| F1 | FND-CLIP output is 512, spec assumes 768 | `sem_proj`: `Linear(512, 768) + GELU` adapter | `models/encoders/fnd_clip.py` |
| F2 | Qwen2-7B has 28 layers / hidden 3584, spec says "layer 30 / 4096" | Use last layer (-1), `Linear(3584, 768) + GELU` | `models/encoders/qwen_text.py` |
| F3 | Stage 1 pretrain corpus unspecified | 10k GenImage + 3k DGM4 + 2k MMFB-tampered (fake) vs 10k VisualNews + 3k DGM4-origin + 2k MMFB-real | `data/builders/build_stage1_binary.py` |

Each deviation is logged with a `[F<n>]` tag in the source file's docstring or comments.

---

## 6. Reproducibility surface

Every checkpoint saved by `BaseTrainer` carries:
- `config_hash` (sha256 of the YAML)
- `data_hash` (sha256 of the manifest CSV)
- `git_sha` (current HEAD)
- `seed`, `torch_version`, `cuda_version`
- `spec_version = "V3.1"`
- `stage` in `{stage0, stage1, stage2}`

See `core/checkpoints.py::build_provenance`.
