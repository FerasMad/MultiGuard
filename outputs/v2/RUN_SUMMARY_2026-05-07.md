# MultiGuard V2 — Run Summary (2026-05-07)


## the spec compliance
- Datasets: NewsCLIPpings genuine = Real, DGM4 face_swap/face_attribute = Manipulated, NewsCLIPpings falsified = OOC ✓
- Frozen FND-CLIP semantic encoder ✓
- ResNet-18 forensic encoder, 1-channel DCT-Y, 768-dim ✓
- Auxiliary classifier 3-class ✓
- Cross-attention fusion 8 heads, 512 proj, 1024 fused ✓
- MLP 1024 → 512 → 256 → 3 ✓
- Joint loss: CE(main) + 0.1·CE(aux) ✓


## Hardware
- NVIDIA GeForce RTX 4070 (12 GiB), CUDA 12.4, torch 2.6.0+cu124


## Phase 1 — forensic baseline (DCT only)


### Test split


- accuracy: 0.4694
- f1_macro: 0.4747
- per-class P / R / F1:
 - **Real**: P=0.3901 R=0.4408 F1=0.4139
 - **Manipulated**: P=0.6507 R=0.5237 F1=0.5803
 - **Ooc**: P=0.4167 R=0.4438 F1=0.4298
- AUC-ROC (macro one-vs-rest): 0.6533
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [149, 48, 141]
 - [92, 177, 69]
 - [141, 47, 150]


### MMFakeBench transfer


- accuracy: 0.3957
- f1_macro: 0.3320
- per-class P / R / F1:
 - **Real**: P=0.4589 R=0.4467 F1=0.4527
 - **Manipulated**: P=0.0833 R=0.1000 F1=0.0909
 - **Ooc**: P=0.4618 R=0.4433 F1=0.4524
- AUC-ROC (macro one-vs-rest): 0.4975
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [134, 48, 118]
 - [53, 10, 37]
 - [105, 62, 133]




## Phase 2 — full pipeline (frozen FND-CLIP + forensic + cross-attention + MLP)


### Test split


- accuracy: 0.5187
- f1_macro: 0.5111
- per-class P / R / F1:
 - **Real**: P=0.4687 R=0.5089 F1=0.4879
 - **Manipulated**: P=0.6198 R=0.7041 F1=0.6593
 - **Ooc**: P=0.4411 R=0.3432 F1=0.3860
- AUC-ROC (macro one-vs-rest): 0.7250
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [172, 71, 95]
 - [48, 238, 52]
 - [147, 75, 116]


### MMFakeBench transfer


- accuracy: 0.3529
- f1_macro: 0.3266
- per-class P / R / F1:
 - **Real**: P=0.4377 R=0.4800 F1=0.4579
 - **Manipulated**: P=0.1629 R=0.3600 F1=0.2243
 - **Ooc**: P=0.4467 R=0.2233 F1=0.2978
- AUC-ROC (macro one-vs-rest): 0.5419
- confusion matrix (rows=true [Real, Manipulated, OOC], cols=pred):
 - [144, 93, 63]
 - [44, 36, 20]
 - [141, 92, 67]




## Artifacts
- Phase 1 ckpt: `outputs/forensic_baseline/best.pt`
- Phase 2 ckpt: `outputs/full_pipeline/best.pt`
- Phase 1 ROC: `outputs/forensic_baseline/{test,mmfb}_roc.png`
- Phase 2 ROC: `outputs/full_pipeline/{test,mmfb}_roc.png`
- Train history: `outputs/{forensic_baseline,full_pipeline}/training_history.csv`
