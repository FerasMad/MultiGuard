# MultiGuard — Honest-path retrain report

> **Status:** placeholder generated at the start of the 12h run.
> Will be overwritten with real numbers at Phase 7 (end of the run).

## Methodology change

Earlier ckpt (`outputs/v4/stage2_fusion_dctforensic/best.pt`, val F1 0.7334) trained against V3-era Qwen text cache built from **original** captions for classes 3/4. Those captions carry a syntactic LLM-fingerprint shortcut — Qwen detected it instantly, hence 99.7/99.8% F1 on 3/4.

The honest-path run regenerates v_textfor for class 3/4 from **BLIP-2-OPT-2.7B image-grounded captions** ("a man playing poker..." instead of "Colin Kaepernick is a poker player."). Text branch can no longer use syntactic shortcuts. Expected drop in 3/4 F1 to a more defensible 0.70-0.85.

## Per-class F1 (test split, ensemble) — to be filled in

| Class | Display | Old | New | Δ |
|---|---|---|---|---|
| 0 | Real news | 0.41 | ___ | ___ |
| 1 | Out-of-context | 0.47 | ___ | ___ |
| 2 | Real text + Fake image | 0.76 | ___ | ___ |
| 3 | Fake text + Real image | 0.997 | ___ | ___ |
| 4 | Fully fake | 0.998 | ___ | ___ |

## Headline — to be filled in

| Config | Test F1-macro | MMFakeBench transfer |
|---|---|---|
| Old (shortcut) | 0.7267 | 0.4805 |
| Honest seed=42 | ___ | ___ |
| 3-seed mean ± std | ___ ± ___ | ___ ± ___ |
| Ensemble | ___ | ___ |
| Ensemble + TTA | ___ | ___ |
| Ensemble + TTA + calibration | ___ | ___ |

## Pipeline changes

| Branch | Before | After |
|---|---|---|
| Text Credibility | Qwen on original captions | Qwen on BLIP-2 captions for class 3/4 |
| Image Authenticity | DCT-Forensic random head | DCT-Forensic seed=42 head (parity fix) |
| Cross-Modal Consistency | V1 leakfree FND-CLIP | V1 leakfree (Stage-0 fresh attempt converged to local min) |
| Attention Fusion | V3PairwiseFusion | V3PairwiseFusion (retrained on new caches) |
| MLP classifier | V3.1 §6 | V3.1 §6 unchanged |

## Inference-time improvements

| Technique | Why | Expected lift |
|---|---|---|
| 3-seed ensemble | reduces single-seed variance | +1-2 pp F1 |
| Horizontal-flip TTA | image symmetry, free at inference | +0.5-1 pp on 0/2/4 |
| Temperature calibration | scalar T on val NLL | unchanged predictions, honest confidence |

## Server-vs-eval parity (P9.1 fix)

Cosine sim runtime vs cached v_imgfor on `genimage_mj_003416`:
- **Before:** 0.047 (random init mismatch)
- **After:** 1.000000 (deterministic seed=42 + saved head state)

## Reproduce

```bash
python phases/v4/scripts/precompute_v_imgfor_dctforensic.py --seed 42
python phases/v4/scripts/blip2_caption_rewrite.py
python phases/v4/scripts/recache_v_textfor_blip2.py
for seed in 42 1337 2024; do
    python -m v4 train --config phases/v4/configs/v4_pipeline_honest.yaml \
        --seed $seed --out-dir outputs/v4/stage2_fusion_honest_seed$seed
done
python phases/v4/scripts/calibrate_temperature.py \
    --config phases/v4/configs/v4_pipeline_honest.yaml \
    --checkpoint outputs/v4/stage2_fusion_honest_seed42/best.pt \
    --out outputs/v4/stage2_fusion_honest_ensemble/temperature.json
python phases/v4/scripts/eval_ensemble.py \
    --config phases/v4/configs/v4_pipeline_honest.yaml \
    --ckpts outputs/v4/stage2_fusion_honest_seed*/best.pt \
    --out-dir outputs/v4/stage2_fusion_honest_ensemble \
    --splits test mmfakebench-transfer
```
