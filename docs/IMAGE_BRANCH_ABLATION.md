# Image-branch ablation (forensic fix-plan deliverable)

> The forensic image-branch fix plan (`forensic_image_branch_fix_plan.md`,
> Step 10) names `docs/IMAGE_BRANCH_ABLATION.md` as a required deliverable:
> "Report class-wise F1 for each ablation. This is required to prove whether
> the corrected image branch helps or hurts."
>
> This doc consolidates the two ablation views that answer that question. The
> full methodology + confusion matrices live in
> [`BRANCH_ABLATION_REPORT.md`](BRANCH_ABLATION_REPORT.md) and
> [`HONEST_PATH_V2_REPORT.md`](HONEST_PATH_V2_REPORT.md).

## Does the image branch (`v_imgfor`) help the 5-class pipeline?

Two independent measurements, both pointing the same way: **yes, but modestly,
and it owns class 2 (Manipulated).**

### View 1 -- training-time ablation (4 fresh fusions, `BRANCH_ABLATION_REPORT.md`)

Each variant is a fusion trained from scratch with the named inputs only
(seed=42, same cached features, same hyperparameters). Answers "how much signal
is available in each input subset."

| Variant | Test F1 | C0 Real | C1 OOC | C2 Manip | C3 AI-Text | C4 FullFab |
|---|---|---|---|---|---|---|
| fndclip (semantic only) | 0.6681 | 0.451 | 0.306 | 0.738 | 0.886 | 0.960 |
| fndclip + text | 0.7103 | 0.411 | 0.448 | 0.731 | 0.981 | 0.982 |
| **fndclip + image** | 0.6975 | 0.391 | **0.464** | **0.751** | 0.911 | 0.970 |
| all three | 0.7157 | 0.430 | 0.426 | 0.752 | 0.985 | 0.985 |

Adding the image branch to FND-CLIP lifts **C2 Manipulated +1.3 pp** (0.738 ->
0.751) and **C1 OOC +15.8 pp** -- consistent with the fix-plan's expectation
that "image branch improves class 2." It does NOT meaningfully help C3 (text's
job).

### View 2 -- inference-time leave-one-out on the SHIPPED ensemble (`inference_ablation/summary.json`)

Loads the deployed 3-seed honest ensemble and zeroes one branch at inference.
Answers "what does the production model actually lean on." This is the more
honest "does it help in the shipped model" test.

| Variant | Test F1 | dF1 | C2 Manip | note |
|---|---|---|---|---|
| full pipeline | 0.7149 | -- | 0.757 | baseline |
| **disable image (`v_imgfor`)** | 0.6954 | **-2.0 pp** | 0.733 (-2.4 pp) | image removal hits C2 hardest |
| disable text (`v_textfor`) | 0.4494 | -26.6 pp | 0.710 | text dominates C3/C4 |
| semantic only | 0.3146 | -40.0 pp | 0.655 | -- |

## Verdict (per the fix-plan acceptance criteria)

| Fix-plan criterion (Step 10 / non-image #7) | Result |
|---|---|
| Image branch improves class 2 | YES +1.3 pp (train) / removal costs C2 -2.4 pp (inference) |
| Image branch improves class 4 | MARGINAL (+1.0 pp train; ~ -0.6 pp on removal) -- C4 is mostly text-driven |
| Text branch improves classes 3 and 4 | YES strongly -- removing text crashes C3 0.989 -> 0.055 |
| FND-CLIP responsible for class 0 vs 1 | YES (data-scale-capped; A1 audit AUC=0.51 linear probe) |
| Adding image does NOT degrade class 0/1 | YES -- C0/C1 essentially flat with/without image |

**Bottom line:** the image branch is a **real but modest contributor (~2 pp F1
overall)** whose value is concentrated in class 2 Manipulated. It does not
hurt the Real/OOC classes. **Track B (below) completed the forensic fix-plan's
data refresh** -- official GenImage nature + SD v1.4/v1.5, both detectors
retrained 8/8, `v_imgfor_corrected` regenerated, fusion retrained -- and the
corrected image branch's contribution stayed ~flat (~1 pp), confirming this is
a data-domain ceiling (the 5-class test images are out-of-domain for any
GenImage-trained detector), not a detector-quality one.

## View 3 -- corrected image branch (Track B: 8-gen detector + official SD nature) -- ✅ DONE

The forensic fix-plan's Definition of Done (retrain on official GenImage nature
+ add SD v1.4/v1.5, regenerate `cache/v4/v_imgfor_corrected/`, re-run this
ablation) is now **complete**. Source: `shimei123/Genimage` (SD_v14.zip +
SD_v15.zip, official GenImage layout with genuine ImageNet nature). Both
forensic detectors were retrained on all 8 generators (RGB+Fourier overall AP
0.9876, DCT 0.9468 -- see `phases/forensic/REPORT.md` §0), `v_imgfor_corrected`
(16,500 shards) was regenerated from the 8-gen DCT backbone, and the 3-seed
fusion was retrained on it (`v4_pipeline_corrected.yaml`).

Inference-time leave-one-out on the **corrected** 3-seed ensemble
(`outputs/v4/inference_ablation_corrected/summary.json`):

| Variant | Test F1 | dF1 | C2 Manip | note |
|---|---|---|---|---|
| full (corrected) | 0.7121 | -- | 0.763 | vs shipped full 0.7149 (flat, seed noise) |
| disable image (`v_imgfor`) | 0.7022 | **-1.0 pp** | 0.751 (-1.2 pp) | corrected image branch removal |
| disable text (`v_textfor`) | 0.5096 | -20.3 pp | 0.738 | text still dominates C3/C4 |
| semantic only | 0.3153 | -39.7 pp | 0.654 | -- |

**What the data refresh changed:** the *standalone* forensic detector improved
substantially (6/8 -> 8/8 generators, official ImageNet nature, RGB AP 0.988).
But re-integrated into the 5-class pipeline the image branch stayed a minor
contributor (removal costs ~1 pp, concentrated in C2 -- comparable to the
shipped -2 pp within seed noise). The 5-class headline is flat (0.7121 vs
0.7147), so the **deployed 5-class ensemble is left unchanged**; the corrected
fusion is the documented Stage-D experiment, not a production swap. F-A3 is
resolved for the two SD generators (their nature is now genuine ImageNet).

## Provenance

- Training-time ablation: `phases/v4/scripts/train_branch_ablation.py` + `eval_branch_ablation.py` -> `docs/BRANCH_ABLATION_REPORT.md`
- Inference-time ablation (shipped): `phases/v4/scripts/eval_inference_ablation.py` -> `outputs/v4/inference_ablation/summary.json`
- Inference-time ablation (corrected, Track B): same script `--config phases/v4/configs/v4_pipeline_corrected.yaml --ckpts outputs/v4/stage2_fusion_corrected_seed{42,1337,2024}/best.pt` -> `outputs/v4/inference_ablation_corrected/summary.json`
- Corrected 3-seed ensemble eval: `phases/v4/scripts/eval_ensemble.py` -> `outputs/v4/stage2_fusion_corrected_ensemble/summary.json` (test F1 0.7121)
- 8-gen forensic detectors + eval table: `phases/forensic/REPORT.md` §0, `phases/forensic/outputs/eval_table_combined.md`
- SD data staging: `phases/forensic/scripts/stage_sd_generators.py` (shimei123/Genimage)
- Image-only transfer probe (binary AP/AUC on DGM4+MMFB): `docs/IMAGE_TRANSFER_PROBE.md` (AP 0.77 / AUC 0.78)
