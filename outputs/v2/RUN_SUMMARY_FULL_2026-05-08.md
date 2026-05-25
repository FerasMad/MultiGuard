# MultiGuard V2 — FULL-SCALE Run Summary (2026-05-08)


## Dataset scale
- **Manifest: 128,040 samples** (42,680 per class — smallest class = NewsCLIPpings genuine/OOC)
- Splits per class: 29,876 train / 6,402 val / 6,402 test
- Built directly from raw DGM4 + NewsCLIPpings + VisualNews (no V1 5-scenario intermediate)


## the spec compliance (V2)
- Datasets balanced to smallest class (42,680 each), 70/15/15 per-class split ✓
- Frozen FND-CLIP, leak-free (fresh BERT/ResNet/CLIP backbones) ✓
- DCT [1,224,224], log scale, per-image min/max norm, cached ✓
- ResNet-18 1-channel conv1, 768-dim ✓
- Aux Linear(768,3) ✓
- Bi-direction cross-attention 8 heads, 512 proj, 1024 fused ✓
- MLP 1024→512→256→3 ✓
- Adam 1e-4, wd 1e-4, batch 64, max 50 epochs, StepLR×0.1@30, early-stop p=10, grad-clip 1.0, total = main_CE + 0.1·aux_CE ✓


## Hardware
- NVIDIA GeForce RTX 4070 (12 GiB), CUDA 12.4, torch 2.6.0+cu124


## Headline comparison vs prior 6,774-sample run


| Metric | Phase 1 small (6,774) | Phase 1 **full (128,040)** | Phase 2 small | Phase 2 **full** |
|---|---|---|---|---|
| Test F1-macro | 0.4747 | **0.5367** | 0.5111 | **0.5395** |
| Test AUC-ROC | 0.6533 | **0.7582** | 0.7250 | **0.7659** |
| MMFakeBench F1 | 0.3320 | **0.3406** | 0.3266 | **0.3633** |
| MMFakeBench AUC | 0.4975 | **0.5023** | 0.5419 | **0.5784** |


## Phase 1 — forensic baseline (DCT only)


### Test split


- accuracy: 0.5381
- f1_macro: 0.5367
- per-class P / R / F1:
 - **Real**: P=0.4740 R=0.5291 F1=0.5000
 - **Manipulated**: P=0.7462 R=0.7331 F1=0.7396
 - **Ooc**: P=0.3907 R=0.3522 F1=0.3705
- AUC-ROC (macro one-vs-rest): 0.7582
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [3387, 789, 2226]
 - [419, 4693, 1290]
 - [3340, 807, 2255]


### MMFakeBench transfer


- accuracy: 0.3800
- f1_macro: 0.3406
- per-class P / R / F1:
 - **Real**: P=0.4467 R=0.5033 F1=0.4734
 - **Manipulated**: P=0.1364 R=0.2400 F1=0.1739
 - **Ooc**: P=0.4892 R=0.3033 F1=0.3745
- AUC-ROC (macro one-vs-rest): 0.5023
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [151, 67, 82]
 - [63, 24, 13]
 - [124, 85, 91]




## Phase 2 — full pipeline (frozen FND-CLIP + forensic + cross-attention + MLP)


### Test split


- accuracy: 0.5448
- f1_macro: 0.5395
- per-class P / R / F1:
 - **Real**: P=0.3768 R=0.2908 F1=0.3283
 - **Manipulated**: P=0.7886 R=0.7498 F1=0.7687
 - **Ooc**: P=0.4650 R=0.5939 F1=0.5216
- AUC-ROC (macro one-vs-rest): 0.7659
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [1862, 622, 3918]
 - [1145, 4800, 457]
 - [1935, 665, 3802]


### MMFakeBench transfer


- accuracy: 0.4171
- f1_macro: 0.3633
- per-class P / R / F1:
 - **Real**: P=0.4303 R=0.3500 F1=0.3860
 - **Manipulated**: P=0.2065 R=0.1900 F1=0.1979
 - **Ooc**: P=0.4615 R=0.5600 F1=0.5060
- AUC-ROC (macro one-vs-rest): 0.5784
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [105, 34, 161]
 - [46, 19, 35]
 - [93, 39, 168]




## Artifacts
- Phase 1 ckpt: `outputs/forensic_baseline_full/best.pt`
- Phase 2 ckpt: `outputs/full_pipeline_full/best.pt`
- ROC curves: `outputs/{forensic_baseline_full,full_pipeline_full}/{test,mmfb}_roc.png`
- Train history: `outputs/{forensic_baseline_full,full_pipeline_full}/training_history.csv`
