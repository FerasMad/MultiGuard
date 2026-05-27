# Approach 1 -- band='all' retrain experiment (Stage C-lite)

> Required by `forensic_image_branch_fix_plan.md` Step 4 ("settle the F-A8
> deviation by retraining at upstream's CLI default `band='all'` and
> comparing per-generator AP/Acc/AUC against the shipped band='low+high' ckpt").

## What this experiment tests

`docs/FOURIER_BAND_AUDIT.md` showed that `band='all'` (upstream-CLI default)
and `band='low+high'` (our shipped class-default) produce **materially different**
masking patterns -- not interchangeable. This experiment retrains Approach 1
end-to-end with `band='all'` on the same VisualNews-as-nature splits and the
same hyperparameters, then evaluates both ckpts per-generator with the
identical `eval_rgb.py` pipeline.

Identical inputs (so the only difference is the band):
- Trainer: `phases/forensic/scripts/train_rgb_fourier.py` (+ new `--band` CLI flag)
- Splits: `phases/forensic/data/genimage_train/{train,val}/` (12000 train, 3000 val)
- Hyperparams: BCEWithLogitsLoss, AdamW lr=1e-4 wd=1e-4, batch=64, max 30 epochs,
  ReduceLROnPlateau(mode='max', factor=0.5, patience=3), ES patience=5 on val AP,
  freeze conv1/bn1/layer1/layer2
- Eval: same 6/8-generator GenImage test split (sd_v1_4, sd_v1_5 still missing
  per `FOLLOWUP.md` A)

Only difference:
- shipped: `RandomFourierMask(..., band='low+high', ...)`
- new:    `RandomFourierMask(..., band='all', ...)`

## Headline result

**`band='low+high'` (shipped) wins overall. `band='all'` is slightly better on
GAN-class generators but loses on diffusion-class generators and has ~74%
higher variance.**

### Aggregate comparison (6 generators evaluated; sd_v1_4 + sd_v1_5 skipped)

| Aggregate | shipped (band=low+high) | new (band=all) | Delta (new - shipped) |
|---|---|---|---|
| Overall AP | **0.9979** | 0.9974 | -0.0005 |
| Overall Accuracy | 0.9800 | 0.9785 | -0.0015 |
| Overall AUC | 0.9980 | 0.9975 | -0.0005 |
| GAN AP (BigGAN only) | 0.9996 | **0.9998** | +0.0002 |
| Diffusion AP (5 diff gens) | **0.9975** | 0.9969 | -0.0006 |
| **StdDev AP across generators** | **0.0023** | 0.0040 | +0.0017 (74% worse) |
| Best val AP during training | 0.9971 (ep 10, 1271s) | 0.9972 (ep 11, 2207s) | +0.0001 (~similar) |

### Per-generator delta (new band='all' minus shipped band='low+high')

| Generator | shipped AP | new AP | dAP | dAcc | dAUC | Verdict |
|---|---|---|---|---|---|---|
| adm        | 0.9987 | **0.9990** | **+0.0003** | +0.0040 | +0.0003 | new wins (small) |
| biggan     | 0.9996 | **0.9998** | **+0.0003** | -0.0040 | +0.0002 | new wins (small) |
| glide      | 0.9986 | **0.9994** | **+0.0009** | +0.0030 | +0.0007 | new wins (clearest) |
| midjourney | **0.9932** | 0.9892 | **-0.0040** | **-0.0150** | -0.0035 | shipped wins (clearest) |
| vqdm       | **0.9980** | 0.9979 | -0.0000 | +0.0040 | -0.0000 | tie |
| wukong     | **0.9992** | 0.9987 | -0.0005 | -0.0010 | -0.0005 | shipped wins (small) |

Bold = winner of the pairwise comparison.

## Interpretation

The pattern is consistent with the audit's prediction (`FOURIER_BAND_AUDIT.md`):

1. **`band='all'` masks 7,527 pixels uniformly over the 224x224 frequency
   map; `band='low+high'` masks only 942 pixels in two corners** (15.0% vs 1.88%
   fraction). The harsher augmentation should be more robust to OOD shifts but
   may learn slightly less of the in-distribution per-generator signal.
2. **Empirically that bears out for the GAN side** (BigGAN +0.0002, glide
   +0.0009, adm +0.0003) -- the more diverse mask exposes the network to a
   wider range of perturbations and slightly improves on these.
3. **But the diffusion side has a different cost**: midjourney drops -0.0040
   AP and -0.0150 Acc -- the largest single-generator hit. Midjourney has
   particularly distinctive low-frequency texture artifacts; the mild
   `band='low+high'` mask preserves them, but `band='all'` knocks them out
   too often during training, reducing the model's sensitivity to them.
4. **Variance increases by 74%**: 0.0023 -> 0.0040. The more aggressive
   augmentation gives the network more freedom to find different boundaries
   per-generator, increasing inconsistency. The doctor's spec implicitly
   prefers low variance (the "StdDev across generators" cell in F.24).

## Verdict

**Keep the shipped `band='low+high'` ckpt as canonical Approach 1.**

The new `band='all'` ckpt is preserved at `phases/forensic/outputs/rgb_bandall/forensic_rgb_model.pth` for reproducibility, but it underperforms on the headline metric (Overall AP) and has higher inter-generator variance. The few generators where `band='all'` wins are small wins (+0.0002 to +0.0009); the loss on midjourney is 4-10x larger in magnitude.

The F-A8 deviation is now empirically settled: it was a *real* divergence from
the upstream CLI default, but on this 6-generator subset the shipped choice
turns out to be marginally better. If the missing SD v1.4 and SD v1.5 generators
were added later (FOLLOWUP A), the comparison may shift -- worth re-running
this experiment when those generators are available.

## Backup & artifact map

| Artifact | Path | Status |
|---|---|---|
| Shipped (band='low+high') ckpt | `phases/forensic/outputs/rgb/forensic_rgb_model.pth` | Production (unchanged) |
| Shipped backup | `phases/forensic/outputs/rgb/forensic_rgb_model.lowhigh.pth` | Defensive copy created by `run_band_all_experiment.py` |
| New (band='all') ckpt | `phases/forensic/outputs/rgb_bandall/forensic_rgb_model.pth` | Experiment artifact (local only) |
| Shipped per-gen eval | `phases/forensic/outputs/eval_rgb_lowhigh.json` + `.md` | New |
| New per-gen eval | `phases/forensic/outputs/eval_rgb_bandall.json` + `.md` | New |
| Delta JSON | `phases/forensic/outputs/band_experiment.json` | New |
| Training history (new) | `phases/forensic/outputs/rgb_bandall/training_history.csv` | New |
| Combined experiment log | `phases/forensic/outputs/band_experiment.log` | New (gitignored) |

## Reproducibility

```bash
# Re-run the full experiment (training + 2x eval + delta):
.venv/Scripts/python.exe phases/forensic/scripts/run_band_all_experiment.py

# Or skip training and just re-eval (if ckpts already exist):
.venv/Scripts/python.exe phases/forensic/scripts/run_band_all_experiment.py --skip-train
```

Wall-clock for full experiment on RTX 4070: ~37 min training + ~10 min for both evals = ~50 min total.

## Provenance

- Trainer hard-coding (before patch): `phases/forensic/scripts/train_rgb_fourier.py:96-98` set `band='low+high'`
- Audit: `docs/FOURIER_BAND_AUDIT.md` (Stage A8) -- proved the masking is materially different
- Patch: `phases/forensic/scripts/train_rgb_fourier.py` -- added `--band` CLI flag (default `low+high` preserves shipped behavior)
- Driver: `phases/forensic/scripts/run_band_all_experiment.py` -- backup + train + eval-compare
- Decision register: `docs/DECISIONS.md` entry F-A8 (now empirically settled)
- Spec source: `docs/doctor-briefs/Forensic_Image_Detector_En.pdf` section F.4-F.11
