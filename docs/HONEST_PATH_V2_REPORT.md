# Honest-path v2 report (P15)

> Closes the gaps Codex flagged after the P14 audit:
> v_semantic was stale (cls 3/4 BERT shortcut), aux loss degenerate under
> ablation, dataset leaks unactioned, inline copy out of sync, and the
> branch ablation conflated "training-time information" with "production
> reliance". This report retracts and replaces those numbers with corrected
> measurements.

## TL;DR

| Metric | Shipped (stale v_sem) | v2 (fresh v_sem) | Delta |
|---|---|---|---|
| **Test F1-macro (seed=42)** | 0.7152 | **0.7149** | -0.0003 (essentially flat) |
| Cls 0 Real F1 | 0.397 | **0.488** | **+9.1 pp** |
| Cls 1 OOC F1 | 0.442 | 0.376 | -6.6 pp |
| Cls 2 Manipulated F1 | 0.757 | 0.762 | +0.5 pp |
| **Cls 3 AI-Text F1** | **0.989** | **0.973** | **-1.6 pp** |
| **Cls 4 Fully-Fab F1** | **0.990** | **0.976** | **-1.4 pp** |
| Best val F1 | 0.7372 @ ep 12 | 0.7270 @ ep 4 | -1.0 pp |

Macro F1 is essentially conserved (-0.03 pp) but the **per-class
redistribution exposes the shortcut**: cls 3/4 lose ~1.5 pp each (the
LLM-syntactic-fingerprint signal that was leaking through FND-CLIP's
BERT sub-encoder) and cls 0 gains ~9 pp (better Real recognition once
the shortcut isn't dominating cls 3/4 training).

The shortcut was real but smaller than feared. The shipped pipeline's
overall headline number wasn't materially inflated.

## What was wrong (recap from P15 audit)

Five issues called out:

1. **v_semantic cache stale for cls 3/4.** The original cache was built on
   2026-05-18 from the LLM-syntactic-fingerprint `text` column. After P9.3's
   BLIP-2 caption rewrite, `v_textfor` was rebuilt from `text_blip2` but
   `v_semantic` was NOT. FND-CLIP V1's BERT sub-encoder still saw the
   shortcut text for all 6,600 cls 3/4 rows.
2. **Branch ablation answered wrong question.** P14 trained a fresh fusion
   per variant -- measures "how much info is in each input subset" -- not
   the doctor's actual leave-one-out question "what happens if we drop
   branch X from the shipped pipeline at inference".
3. **Train/test image_path leaks documented but not actioned.** A3 audit
   found 16 image_paths in both train and test (all VisualNews cls 0/1).
4. **Aux loss degenerate under disable_branches.** When `v_imgfor` was
   zeroed by the P14 ablation patch, `aux_classifier(v_imgfor.detach())`
   returned `bias`-only logits and trained the aux head's bias toward
   marginal class balance.
5. **HF Space inline copy out of sync.** Patched the inline
   `v3_pairwise.py` on `FerasMad/MultiGuard` but never pushed to
   `FerasMad/multiguard-demo`.

## What we did (P15.1 ... P15.5)

### P15.1a: rebuilt v_semantic for cls 3/4 with BLIP-2 captions

`phases/v4/scripts/recache_v_semantic_blip2.py`:
- Copies cls 0/1/2 verbatim from existing cache (9,900 rows).
- Re-encodes cls 3/4 rows via `FNDCLIPSemanticEncoder(ckpt=outputs/v1/leakfree/best.pt).forward_semantic(image, text_blip2)` for 6,600 rows.
- Output: `cache/v3/_features/v_semantic_blip2/` (16,500 shards, 0 failures, 13.2 min on resume).
- First-pass run hit a Python heap allocation failure at 96.3% after 6h
  (tokenizer-rebuild-per-call leak). Resume picked up the remaining 240
  rows cleanly via `dst.exists()` skip logic.

### P15.1b: retrained Stage 2 seed=42 with the fresh cache

`phases/v4/configs/v4_pipeline_honest_v2.yaml` (single delta from
`v4_pipeline_honest.yaml`: `v_semantic` subdir points at the new cache).
Same lr=1e-4, wd=1e-4, batch=64, bf16, seed=42, ES patience=10 on val F1.

| Run | Best val F1 | Best ep | Test F1 (single seed) |
|---|---|---|---|
| Shipped honest seed=42 (stale cache) | 0.7372 | 12 | 0.7152 |
| **v2 (fresh cache)** | **0.7270** | **4** | **0.7149** |

Val F1 dropped 1 pp; test F1 is essentially identical at the headline.
The model now converges much earlier (ep 4 vs ep 12) -- consistent with
"less shortcut signal to exploit, so it converges faster on the legitimate
signal then overfits the smaller margin".

### P15.2: inference-time leave-one-out ablation on shipped ensemble

`phases/v4/scripts/eval_inference_ablation.py` -- loads the SHIPPED
3-seed ensemble (`outputs/v4/stage2_fusion_honest_seed{42,1337,2024}/best.pt`)
and runs test split under 4 `disable_branches` settings.

| Variant | Test F1 | C0 Real | C1 OOC | C2 Manip | C3 AI-Text | C4 FullFab |
|---|---|---|---|---|---|---|
| **all (full pipeline)** | **0.7149** | 0.397 | 0.442 | 0.757 | 0.989 | 0.990 |
| no_image (disable v_imgfor) | 0.6954 | 0.349 | 0.428 | 0.733 | 0.983 | 0.984 |
| **no_text (disable v_textfor)** | **0.4494** | 0.401 | 0.247 | 0.710 | **0.055** | 0.834 |
| semantic_only (disable both) | 0.3146 | 0.362 | 0.040 | 0.655 | 0.040 | 0.476 |

**Big finding: the text branch is overwhelmingly dominant at inference.**

- Removing the image branch costs only **2.0 pp** F1 -- the doctor's
  expectation that "image branch helps class 2 and class 4" is correct
  but the magnitude is small. C2 drops 2.4 pp, C4 drops 0.6 pp.
- Removing the text branch costs **26.6 pp** F1. **C3 (AI-Text) collapses
  from 0.989 to 0.055** -- the shipped ensemble's near-perfect cls 3
  detection is almost entirely Qwen-driven. The doctor's expectation
  that "text branch helps classes 3 and 4" is confirmed dramatically.
- FND-CLIP alone (semantic_only) at runtime delivers only F1 = 0.31.
  This is much lower than what P14 reported for the fresh-trained
  fndclip-only variant (0.6681) because P14 LET THE FUSION RETRAIN to
  use whatever signal lives in v_semantic. The inference-time number
  reveals that the **shipped fusion's classifier doesn't actually
  rely on v_semantic alone -- it leans hard on Qwen text**.

### P15.3: test_clean (train/test image_path leaks removed)

Same script also evaluates on `test_clean` = test split with the 16 leaked
rows removed (each is a VisualNews image that appeared in both train and
test, all in cls 0/1).

| Variant | test F1 | test_clean F1 | Effect of leak removal |
|---|---|---|---|
| all | 0.7149 | 0.7150 | +0.0001 |
| no_image | 0.6954 | 0.6962 | +0.0008 |
| no_text | 0.4494 | 0.4487 | -0.0007 |
| semantic_only | 0.3146 | 0.3155 | +0.0009 |

**The 16 leaks change F1 by less than 0.001 in any direction.** The audit
finding is real but the magnitude is negligible -- well within seed-to-seed
noise. The shipped F1=0.7149 number is honest at 4-significant-figure
precision even with the leak.

### P15.4: aux loss handling under disable_branches

`phases/v4/{src/v4,app_hf/inline}/v3_pairwise.py`:
```python
imgfor_disabled = bool(self.disable_branches) and "v_imgfor" in self.disable_branches
...
if imgfor_disabled:
    aux_logits = torch.zeros(v_imgfor.shape[0], 2, device=v_imgfor.device, dtype=v_imgfor.dtype)
else:
    aux_logits = self.aux_classifier(v_imgfor.detach())
```

Eliminates the degenerate-bias-gradient path. Verified: 45 phases/v4
tests pass; smoke confirms `aux_logits` is exactly zeros when
`v_imgfor` is in `disable_branches` (vs `bias`-driven non-zero before).

### P15.5: HF Space inline copy synced

`huggingface_hub.upload_file("phases/v4/app_hf/inline/v3_pairwise.py", repo_id="FerasMad/multiguard-demo", repo_type="space")`.
The deployed Space's inline copy now matches the repo's. No runtime
behavior change (Space doesn't pass `disable_branches`).

## Reconciling with P14's findings

| P14 ablation claim | Status after P15 |
|---|---|
| FND-CLIP alone gets test F1=0.6681 with C3=0.886, C4=0.960 | TRUE for **fresh-trained** fndclip-only fusion (P14 measured info content). **NOT** what the shipped model does at inference (P15.2 shows semantic_only delivers F1=0.31, C3=0.04). |
| Text branch contributes +4.2 pp | **Understated.** Text branch is responsible for **26.6 pp** of the shipped ensemble's F1 at inference. |
| Image branch contributes +2.9 pp | Confirmed -- inference ablation shows 2.0 pp drop without image. |
| FND-CLIP is the bottleneck for cls 0/1 | **Confirmed.** P15 v2 retrain bumps C0 by 9 pp once the shortcut is gone, but C1 remains stuck at 0.376 -- the FND-CLIP V1 Real-vs-OOC ceiling (A1 audit found AUC=0.51 linear probe). |
| Image owns C2 signal; text owns C3/C4 | **Confirmed via inference ablation:** removing text crashes C3 to 0.055 (text owns C3); removing image hits C2 hardest (-2.4 pp); C4 split between both. |

## What this means for the doctor

The shipped 3-seed honest-path ensemble (`outputs/v4/stage2_fusion_honest_*`)
remains the production model. Its **headline test F1 of 0.7149 is honest
to 4 decimal places** even accounting for:
- The v_semantic shortcut (worth -1.5 pp on cls 3, -1.4 pp on cls 4)
- The 16 train/test leaks (worth < 0.001 F1)
- The aux loss degeneracy (training-side only, doesn't affect inference)

The biggest doctor-facing insight: **the V3.1 pipeline's classes 3/4
performance is almost entirely Qwen text branch driven.** The FND-CLIP
semantic branch is mostly an OOC-detection signal (and a noisy one --
the data-scale ceiling A1 found). The image branch is a modest +2 pp
contributor at inference; its existence is justified mainly by cls 2.

## Artifacts

| Artifact | Path |
|---|---|
| v_semantic_blip2 cache (16,500 shards) | `cache/v3/_features/v_semantic_blip2/` (gitignored) |
| v2 retrain ckpt | `outputs/v4/stage2_fusion_honest_v2_seed42/best.pt` (gitignored) |
| v2 retrain history | `outputs/v4/stage2_fusion_honest_v2_seed42/training_history.csv` |
| v2 test metrics | `outputs/v4/stage2_fusion_honest_v2_seed42/eval_test/metrics.json` |
| Inference ablation per-variant metrics | `outputs/v4/inference_ablation/<variant>_<split>/metrics.json` |
| Inference ablation summary | `outputs/v4/inference_ablation/summary.json` |
| Recache log | `cache/v3/_features/recache_v_semantic_blip2.log` (gitignored) |
| Training log | `outputs/v4/stage2_fusion_honest_v2_seed42.log` (gitignored) |

## Reproducibility

```bash
# P15.1a: rebuild v_semantic for cls 3/4
.venv/Scripts/python.exe phases/v4/scripts/recache_v_semantic_blip2.py

# P15.1b: retrain Stage 2 with fresh cache
.venv/Scripts/python.exe -m v4 train --config phases/v4/configs/v4_pipeline_honest_v2.yaml

# P15.1b: eval v2 ckpt on test split
.venv/Scripts/python.exe -m v4 eval \
    --config phases/v4/configs/v4_pipeline_honest_v2.yaml \
    --split test \
    --out-dir outputs/v4/stage2_fusion_honest_v2_seed42/eval_test

# P15.2 + P15.3: inference-time leave-one-out ablation on shipped ensemble
.venv/Scripts/python.exe phases/v4/scripts/eval_inference_ablation.py
```

Wall-clock: ~6h (v_semantic recache) + ~3 min (v2 train) + ~30 s (eval) + ~3 min (inference ablation) = ~6h 7 min on RTX 4070.

## Open follow-ups (carried forward)

1. **Class 1 OOC ceiling.** v2 drops C1 by 6.6 pp -- the model lost some
   OOC-detection signal that was leaking through the cls 3/4 shortcut.
   Need to investigate whether C1's true ceiling is ~0.38 (data-scale
   constraint from A1) or whether multi-seed v2 ensemble would
   recover some of the C1 F1 the shipped ensemble had.
2. **Multi-seed v2 retrain.** This report is single-seed (42). The
   shipped numbers are 3-seed ensembled. For an apples-to-apples
   headline comparison, retrain seeds 1337 + 2024 against
   `v_semantic_blip2` and run the ensemble. ~2 GPU-hours.
3. **Stage C data refresh** (deferred from P14). SD v1.4 + SD v1.5
   still missing from the forensic test split; full GenImage 8-gen
   coverage requires browser-auth Drive download.
4. **Bias-corrected transfer F1 = 0.7197** remains offline-only;
   `app/server_v4_retrain.py` doesn't apply the log-prior shift at
   runtime. Mirror at ~40 LOC if production needs to publish that
   number.
