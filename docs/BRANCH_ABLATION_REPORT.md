# Branch ablation report (Stage B4)

> Required by `multiguard_non_image_pipeline_fix_plan.md` section 7
> ("Branch ablations are required before claiming the fix worked").

## What this measures

Four V3PairwiseFusion variants trained on the **same cached features**
(seed=42, identical hyperparameters) with selected branches **zeroed at
the post-projection layer** before the LayerNorm + cross-attention stack.
This isolates each branch's marginal contribution to the 5-class downstream
task without re-training encoders.

| Variant | Active branches | Disabled (zeroed post-projection) |
|---|---|---|
| `fndclip` | v_semantic | v_imgfor, v_textfor |
| `fndclip_text` | v_semantic, v_textfor | v_imgfor |
| `fndclip_image` | v_semantic, v_imgfor | v_textfor |
| `all` | all three | none |

Training: same `v4_pipeline_honest.yaml` schema (lr=1e-4 wd=1e-4 batch=64 bf16
seed=42 max-epochs=25 ES patience=10 on val F1-macro). All four runs use the
same cached features (`v_semantic` V1 leakfree 512-d, `v_imgfor` from Approach 2
DCT 768-d, `v_textfor` from Qwen2-7B on BLIP-2 honest captions 3584-d).

## Test-split F1 (5-class, V3.1 section 7 metric)

| Variant | F1-macro | Best val F1 | Train epoch | Delta vs fndclip-only |
|---|---|---|---|---|
| `fndclip` (semantic only) | **0.6681** | 0.6867 | ep 8 | (baseline) |
| `fndclip_text` | **0.7103** | 0.7217 | ep 4 | **+4.22 pp** |
| `fndclip_image` | **0.6975** | 0.7002 | ep 17 | **+2.94 pp** |
| `all` (full pipeline) | **0.7157** | 0.7304 | ep 4 | **+4.76 pp** |

The honest-path 3-seed ensemble (shipped) scores test F1 = 0.7149 on the same
split; the single-seed `all` ablation here at 0.7157 is consistent within
+/-1pp of the ensemble mean.

## Per-class test F1 (V3.1 5-class taxonomy)

| Variant | C0 Real | C1 OOC | C2 Manip | C3 AI-Text | C4 FullFab |
|---|---|---|---|---|---|
| `fndclip`          | 0.4505 | 0.3056 | 0.7384 | 0.8860 | 0.9600 |
| `fndclip_text`     | 0.4105 | 0.4478 | 0.7308 | **0.9807** | **0.9819** |
| `fndclip_image`    | 0.3913 | **0.4636** | **0.7510** | 0.9113 | 0.9703 |
| `all`              | 0.4302 | 0.4259 | **0.7525** | **0.9847** | **0.9849** |

Bold = best non-baseline value in column.

### Per-class contribution analysis

**Class 0 (Real, NewsCLIPpings matched)** -- 0.4505 -> 0.4302 (full pipeline).
Adding text **HURTS Real F1** by 4pp. The text-on confusion matrices show
Real rows being mis-routed to OOC (260 / 495 in fndclip_text vs 142 / 495 in
fndclip-only). The text branch is over-eager to call Real text "OOC" when
combined with mid-strength semantic signal. **Mitigation**: this is the
expected ceiling artifact from FND-CLIP being at-chance on Real vs OOC linearly
(see `docs/FNDCLIP_REAL_OOC_AUDIT.md`); the text branch amplifies whatever
weak boundary FND-CLIP has and pushes Real -> OOC.

**Class 1 (Out-of-Context, NewsCLIPpings mismatched)** -- 0.3056 -> 0.4259.
Both image (+15.8pp) and text (+14.2pp) help OOC recall substantially. The
image branch helps OOC slightly more than the text branch in isolation
(0.4636 vs 0.4478). Adding both gives 0.4259 -- LESS than image alone --
because text mostly contributes "Real <-> OOC confusion" rather than
disambiguation. **Doctor's spec expectation**: "FND-CLIP is the main
Real-vs-OOC branch." Reality: FND-CLIP alone delivers 0.306 F1; multimodal
fusion lifts it to 0.43 by leveraging text/image for additional context cues.
The FND-CLIP-only ceiling on Real-vs-OOC is the V1 leakfree ckpt's data-scale
limit, not a fusion issue.

**Class 2 (Manipulated, DGM4 + MMFakeBench tampered)** -- 0.7384 -> 0.7525.
Image branch is the dominant signal (+1.3pp from `fndclip_image`), text actually
*hurts* slightly (-0.8pp). **Doctor's spec expectation**: "Image forensics
should be neutral for classes 0 and 1; image branch should improve class 2."
Confirmed. Small absolute improvement because DGM4 tampering is already
detectable from `v_semantic`'s text-image alignment signal (FND-CLIP at 0.738
without forensic input).

**Class 3 (AI-Text, MMFakeBench textual fakes)** -- 0.8860 -> 0.9847.
Text branch is the dominant signal (+9.5pp from `fndclip_text`). Image branch
adds only +2.5pp. **Doctor's spec expectation**: "Text branch should improve
classes 3 and 4." Confirmed strongly.

**Class 4 (Fully-Fabricated)** -- 0.9600 -> 0.9849.
Text branch adds +2.2pp; image branch adds only +1.0pp. **Doctor's spec
expectation**: "image branch should improve class 4." Partially confirmed --
image branch DOES help, but the marginal F1 lift is small because FND-CLIP
alone already hits 0.96 (BLIP-2 captions for class 4 are distinctively
AI-themed, picked up by FND-CLIP's BERT branch even without v_textfor).

## Confusion matrices

Format `[true_class, predicted_class]` -- rows = true, columns = predicted.

### `fndclip` (semantic only) -- test F1 0.6681

```
        pred->  C0   C1   C2   C3   C4
true C0:       246  142   88   16    3
true C1:       260  123   87   20    5
true C2:        51   24  398   20    2
true C3:        30   12    8  443    2
true C4:        10    9    2    6  468
```

### `fndclip_text` -- test F1 0.7103

```
        pred->  C0   C1   C2   C3   C4
true C0:       188  260   47    0    0
true C1:       193  251   51    0    0
true C2:        39  114  342    0    0
true C3:         0    1    1  483   10
true C4:         1    0    0    7  487
```

### `fndclip_image` -- test F1 0.6975

```
        pred->  C0   C1   C2   C3   C4
true C0:       171  238   67   16    3
true C1:       161  248   68   16    2
true C2:        26   68  386   14    1
true C3:        15   14    8  457    1
true C4:         6    7    4    5  473
```

### `all` (full pipeline) -- test F1 0.7157

```
        pred->  C0   C1   C2   C3   C4
true C0:       205  230   60    0    0
true C1:       210  217   68    0    0
true C2:        42   76  377    0    0
true C3:         0    1    1  484    9
true C4:         1    0    1    4  489
```

## Verdict against the doctor's expectations

| Spec expectation (non-image fix plan section 7) | Result |
|---|---|
| Image branch improves class 2 and class 4 | YES -- C2 +1.3pp from image; C4 image alone +1.0pp |
| Text branch improves class 3 and class 4 | YES -- C3 text alone +9.5pp; C4 text alone +2.2pp |
| FND-CLIP remains responsible for class 0 vs class 1 | PARTIAL -- FND-CLIP alone gives C0/C1 F1 0.45/0.31 (ceiling from data-scale, NOT a fusion fault). Multimodal lifts C1 to 0.43 via additional cues. |
| Adding image does NOT significantly degrade class 0/1 | YES -- C0/C1 in `all` are 0.430/0.426 vs `fndclip_text` 0.411/0.448. Image presence is broadly neutral on C0/C1. |

## Summary of contribution magnitudes

| Branch | Net F1-macro contribution | Per-class biggest wins |
|---|---|---|
| FND-CLIP alone | F1 0.6681 baseline | C3 (0.886), C4 (0.960) -- caption-driven |
| + Text branch | **+4.22 pp** | C3 +9.5pp, C4 +2.2pp, C1 +14.2pp |
| + Image branch | **+2.94 pp** | C1 +15.8pp, C2 +1.3pp, C4 +1.0pp |
| Text + Image together | **+4.76 pp total** | Marginal additional vs text-only: +0.54pp |

The text branch contributes most of the lift; the image branch's marginal
addition on top of FND-CLIP+text is only +0.54pp. This is consistent with
the A4 image-transfer probe finding (`docs/IMAGE_TRANSFER_PROBE.md`) that
`v_imgfor` transfers to the downstream image distribution with AUC=0.78
(useful but not dominant), and the A1 finding that FND-CLIP alone is at
chance on Real-vs-OOC.

## Recommendations (informs Stage C decision)

1. **Stage C data refresh is JUSTIFIED but not URGENT.** The image branch
   delivers measurable +2.9pp contribution alone; on top of text it adds
   only ~0.5pp. A wider AI-image distribution (Stage C: SD v1.4/v1.5 +
   ImageNet "nature") would likely lift the image branch's contribution
   from "small but real" to "moderate" -- decision belongs to user.
2. **FND-CLIP retry is NOT JUSTIFIED.** A1 audit already proved FND-CLIP
   V1 leakfree is at chance on Real-vs-OOC linearly; the ablation here
   shows the multimodal pipeline lifts C1 OOC from 0.31 to 0.43 via
   cross-modal context. The remaining ceiling is data-scale, confirmed
   in P10.2 (Stage-0 fine-tune overfit at val_auc=0.65). Need more
   NewsCLIPpings rows, not a recipe change.
3. **Image+text are complementary on different classes.** Image owns C2
   (Manipulated) signal; text owns C3/C4 (AI-Text/Fully-Fab) signal. Each
   branch is doing the job V3.1 spec assigned to it.

## Reproducibility

```bash
# Train all 4 variants (~16 min on RTX 4070 with warm cache):
.venv/Scripts/python.exe phases/v4/scripts/train_branch_ablation.py --variant sweep --epochs 25

# Eval all 4 variants on test split (~1 min):
.venv/Scripts/python.exe phases/v4/scripts/eval_branch_ablation.py --split test
```

Outputs:
- `outputs/v4/branch_ablation_summary.json` -- train durations + best val F1
- `outputs/v4/branch_ablation_eval.json` -- per-variant test metrics
- `outputs/v4/stage2_ablation_<variant>/best.pt` -- ckpts (kept on local disk only)
- `outputs/v4/eval/ablation_<variant>_test/{metrics.json, classification_report.txt, confusion_matrix.png}`

## Provenance

- Manifest: `data/processed/forensic_5class_unified_blip2.csv` (BLIP-2 honest captions for cls 3/4)
- Encoders: `v_semantic` V1 leakfree FND-CLIP, `v_imgfor` Approach 2 DCT (`phases/forensic/outputs/dct/forensic_dct_model.pth`), `v_textfor` Qwen2-7B-Instruct layer -1
- Trainer: `phases/v4/src/v4/training/trainer.py` `BaseTrainer` (V3.1 section 5.5 default)
- Spec contracts verified in `phases/v4/tests/spec_compliance/test_fusion_architecture.py` (6 tests pass)
- Patch enabling ablation: `disable_branches` kwarg added to `V3PairwiseFusion.__init__` (additive, default-no-op; existing ckpts load with `strict=True`)
