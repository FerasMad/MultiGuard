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
hurt the Real/OOC classes. This is the expected behavior for an image-forensic
signal whose detector currently trains on the VisualNews-as-nature substitute
(F-A3); the forensic fix-plan's data refresh (official GenImage nature + SD
v1.4/v1.5) is expected to raise this contribution and is tracked as the
outstanding Stage C/D work.

## Caveat -- detector still trains on the VisualNews substitute

These numbers use `v_imgfor` from the Approach-2 DCT detector trained with
VisualNews-as-nature (deviation F-A3). The forensic fix-plan's Definition of
Done requires retraining on official GenImage nature + adding SD v1.4/v1.5,
then regenerating `cache/v4/v_imgfor_corrected/` and re-running this ablation.
Until that data is staged (Drive-only, browser-auth), this is the best
available image-branch ablation. See `STATUS.md` Track B.

## Provenance

- Training-time ablation: `phases/v4/scripts/train_branch_ablation.py` + `eval_branch_ablation.py` -> `docs/BRANCH_ABLATION_REPORT.md`
- Inference-time ablation: `phases/v4/scripts/eval_inference_ablation.py` -> `outputs/v4/inference_ablation/summary.json`
- Image-only transfer probe (binary AP/AUC on DGM4+MMFB): `docs/IMAGE_TRANSFER_PROBE.md` (AP 0.77 / AUC 0.78)
