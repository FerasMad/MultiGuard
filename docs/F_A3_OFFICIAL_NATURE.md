# F-A3 resolved — official ImageNet nature for all 8 generators

> Deviation **F-A3** (forensic fix-plan): the doctor's spec F.3 says the "real"
> class is GenImage's official ImageNet *nature*. Earlier builds substituted
> **VisualNews** news photos for the real class (ImageNet was not staged), and
> Track B fixed it only for SD v1.4/v1.5. This document records the **full fix**:
> all 8 generators retrained with genuine ImageNet ILSVRC2012 nature, and the
> measured before/after.

## What changed

Source: `shimei123/Genimage` hosts every generator as a discrete zip in the
official GenImage layout (`0_real/` = ImageNet ILSVRC2012, `1_fake/` = generated).
All 8 generators were re-staged from these zips — **both** the AI (fake) and the
nature (real) class are now the official GenImage release, replacing the prior
mix (bitmind/local AI + VisualNews nature). Pipeline:
`stage_sd_generators.py` (all 8) → `build_splits.py` → `precompute_dct.py` →
`compute_dct_stats.py` → retrain both detectors → eval on `genimage_test_official`.

- New DCT z-score (official train): mean `0.245056`, std `2.527811` (vs mixed-8gen
  0.2075 / VisualNews-6gen 0.1187 — the official nature shifts the distribution).
- Checkpoints: `phases/forensic/outputs/{dct_official,rgb_official}/forensic_*_model.pth`.

## Before / after (overall, 8/8 generators)

| Approach | Mixed-8gen (VisualNews nature for 6 gens) | **Official-8gen (ImageNet nature, all 8)** | Δ overall AP |
|----------|------------------------------------------|--------------------------------------------|--------------|
| A1 RGB+Fourier | AP 0.9876 / Acc 0.947 / StdDev 0.015 | **AP 0.9841 / Acc 0.940 / StdDev 0.011** | **−0.4 pp** |
| A2 DCT | AP 0.9468 / Acc 0.886 / StdDev 0.065 | **AP 0.9085 / Acc 0.834 / StdDev 0.061** | **−3.8 pp** |

The honest (official-nature) numbers are **lower**, especially for the DCT detector.
That is expected and is the whole point of fixing F-A3.

## The decisive evidence: per-generator before/after (DCT)

The 6 generators whose nature *switched* VisualNews→official dropped; the 2 SD
generators whose nature was *already* official held or rose.

| Generator | Nature source change | DCT AP mixed → official | Direction |
|-----------|---------------------|-------------------------|-----------|
| midjourney | VisualNews → ImageNet | 0.9158 → 0.8277 | ↓ −8.8 pp |
| vqdm | VisualNews → ImageNet | 0.9859 → 0.8302 | ↓ −15.6 pp |
| wukong | VisualNews → ImageNet | 0.9828 → 0.9023 | ↓ −8.1 pp |
| glide | VisualNews → ImageNet | 0.9938 → 0.9750 | ↓ −1.9 pp |
| adm | VisualNews → ImageNet | 0.9952 → 0.9706 | ↓ −2.5 pp |
| biggan | VisualNews → ImageNet | 0.9998 → 0.9761 | ↓ −2.4 pp |
| **sdv1_4** | **already ImageNet** | **0.8381 → 0.8904** | **↑ +5.2 pp** |
| **sdv1_5** | **already ImageNet** | **0.8631 → 0.8956** | **↑ +3.3 pp** |

**Interpretation.** The detector trained with VisualNews-as-nature was partly
learning a *news-photo-vs-AI domain cue* rather than pure generation artifacts:
every generator that lost the VisualNews shortcut dropped (some sharply, e.g.
VQDM −15.6 pp). The two SD generators, which never had the shortcut, actually
*improved* once the other six contributed honest (harder) ImageNet-nature
training signal. This is exactly the failure mode F-A3 warned about, now
quantified. The RGB+Fourier detector is far more robust to the change (−0.4 pp
overall) — its Fourier-masked frequency features depend less on the real-class
domain.

## Verdict

- **F-A3 is fully resolved.** All 8 generators use genuine ImageNet ILSVRC2012
  nature; no VisualNews substitute remains anywhere in the forensic pipeline.
- The **official numbers are now canonical** (`eval_table_combined.md`). The prior
  mixed-source numbers are preserved as `outputs/*.mixed8gen.*` and the original
  6-gen as `outputs/*.6gen.*`.
- **A1 RGB+Fourier (overall AP 0.984)** is the recommended detector — strong and
  robust. **A2 DCT (0.908)** is the honest frequency-domain baseline; its larger
  drop is itself the finding.

## Provenance

- Staging: `phases/forensic/scripts/stage_sd_generators.py --out data/raw/GenImage_official`
- Detectors: `phases/forensic/outputs/{dct_official,rgb_official}/`
- Eval: `phases/forensic/outputs/eval_{dct,rgb}_official.json`, `eval_table_combined.md`
- Stats: `phases/forensic/data/dct_stats_official.json`
