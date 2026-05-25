# Phase V3 / V3.1 — 5-Class Three-Signal Fusion (Qwen Text Forensics)

**Status:** Active reference implementation. **Superseded by V4** for new modular work, but the V3.1 architecture is the spec V4 implements. V3 outputs (especially `outputs/v3/_checkpoints/v3_pipeline_qwen/best.pt`) are the canonical baseline numbers.

**Source brief:** `docs/doctor-briefs/Implementation Guidelines V3_1.pdf`

---

## What V3 / V3.1 was

5-class multimodal classifier with three forensic streams:

- **Semantic** — FND-CLIP (frozen, from V1) -> `v_semantic` [B, 768]
- **Image forensic** — UnivFD (ResNet50 + 1ch conv1 + Kaiming init + blur_jpg_v0 strict=False) on patch-DCT -> `v_imgfor` [B, 768]
- **Text forensic** — Qwen2-7B-Instruct, layer 30 hidden state, masked-mean pool -> Linear(4096, 768) + GELU -> `v_textfor` [B, 768]

**Fusion (V3.1 section 5):** LayerNorm per signal -> 3 bidirectional pairwise MHA (8 heads) -> **element-wise SUM** of both directions per pair -> Conv1d stack -> AdaptiveAvgPool -> [B, 1024]

**Classifier (V3.1 section 6):** Linear(1024,512) + BN + GELU + Dropout(0.5) -> Linear(512,256) + GELU -> Linear(256,5)

**Loss (V3.1 section 5.5):** CE(main_logits) + 0.1 * BCE(aux_logits) with `.detach()` on aux input.

**5 classes:** 0 Real / 1 OOC / 2 Manipulated / 3 AI-Text / 4 Fully-Fabricated.

---

## Final V3 / V3.1 metrics (canonical)

From `outputs/v3/_checkpoints/v3_pipeline_qwen/best.pt` (rev 11, Qwen2-7B-Instruct variant):

| Surface | F1-macro | Accuracy | AUC-macro |
|---|---|---|---|
| In-distribution val | 0.7215 | 0.7208 | 0.9305 |
| In-distribution test | ~0.5377 (reported in IMAGE_ONLY_METRICS.pdf) | — | ~0.84 |
| MMFakeBench transfer | 0.3832 | — | 0.8168 |

The val-vs-test gap was investigated extensively (rev 7-12). Class 4 "Double-Fake" had a BLIP-2 caption fingerprint shortcut diagnosed in rev 12 (c4fix); rewriting captions via LLM partially closed the gap.

---

## Revisions

| Rev | What changed |
|---|---|
| 7 | Initial 5-class extension from V2, RoBERTa text encoder, aux head with detach() |
| 8 | Unfrozen forensic encoder retrain (bf16) |
| 9 | V3.1-spec integration: trainer + evaluator + config using student modules |
| 10 | lr=5e-5 retry (vs default 1e-4) |
| 11 | Qwen2-7B-Instruct swap (replaced RoBERTa as v_textfor source) — **canonical** |
| 12 | Mid-layer Qwen ablation (layer 14 vs 30) + Class 4 BLIP-2 caption diagnosis + LLM caption rewrite (c4fix) |

---

## Files

```
phases/v3/
├── src/
│   └── models/
│       ├── v3_pipeline.py            # V3.1 fusion + classifier (PairwiseCrossAttention + V3FusionModule + MLPClassifier)
│       ├── text_fluoroscopy.py       # Qwen2-7B-Instruct hidden state extractor
│       ├── univfd_encoder.py         # UnivFD: ResNet50 + 1ch conv1 (Kaiming) + patch-DCT
│       └── fnd_clip.py               # Mirrored from V1 (frozen v_semantic encoder)
├── scripts/
│   ├── precompute_patch_dct.py       # V3 patch-DCT preprocessor (cv2.dct-based)
│   └── eval_image_only.py            # Image-only ablation (aux head metrics on test split)
└── .venv-qwen/                       # Python 3.12 venv with bitsandbytes (Qwen quantization), gitignored

cache/v3/_features/                   # Precomputed features (gitignored, 6.9 GB)
├── v_semantic/                       # FND-CLIP [B, 512]
├── v_imgfor/                         # UnivFD [B, 768]
├── v_textfor/, v_textfor_qwen*/      # Qwen [B, 3584] variants (canonical + c4fix + l14 ablation)
└── v_semantic_c4fix/, v_textfor_c4fix/   # c4fix variants

outputs/v3/
├── _checkpoints/                     # gitignored (3.2 GB)
│   ├── stage2_5class/                # V3.1 canonical
│   ├── v3_pipeline_qwen/             # rev 11 canonical (best.pt is the baseline)
│   ├── v3_pipeline_qwen_c4fix/       # rev 12 c4fix
│   ├── v3_pipeline_qwen_l14/         # rev 12 ablation
│   └── ...
├── genimage_eval/                    # Class-4 BLIP-2 caption shortcut diagnostics
└── image_only_eval/                  # Aux head metrics on 2475-sample test split
```

---

## How to re-run

V3 used `.logs/c4fix_*.sh` and `.logs/qwen_*.sh` (now `.logs/*.sh` after sed-fix to point at `MultiGuard/`) to chain precompute -> train -> eval. After unification:

```bash
cd C:\Desktop\MultiGuard
# Activate V3-specific qwen venv:
phases/v3/.venv-qwen/Scripts/Activate.ps1  # has bitsandbytes for Qwen quantization
# Or root .venv if you don't need quantization:
.venv\Scripts\Activate.ps1
bash .logs/c4fix_train_run.sh   # or .logs/qwen_*.sh
```

(Import paths inside V3 scripts may need patching after migration. Archived code — fix as you go.)

---

## Why V3 became the spec for V4

V3.1 IS the locked specification. The doctor approved every architectural detail (3 pairwise MHA SUM, the 1024->512+BN->256->5 classifier, the 0.1*BCE-aux with detach). V4 was a clean re-implementation of V3.1 with proper registry/modularity for future swappability.

For new work, use V4. V3 stays here as the canonical baseline + IMAGE_ONLY_METRICS reference.

---

## Doctor's IMAGE_ONLY_METRICS report

See `docs/doctor-briefs/IMAGE_ONLY_METRICS.pdf` — evaluates V3 Qwen rev 11's aux head on 2475-sample test split (binary AI/tampered vs real, positive label = classes 2+4). Numbers in V3.1 metrics table above.
