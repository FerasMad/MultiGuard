# Fourier band: "all" vs "low+high" behavioral audit (Stage A8)

> Required by `forensic_image_branch_fix_plan.md` Step 4 ("settle whether `band='all'` retrain is needed by first proving they are NOT behaviorally identical").

## Question

Doctor's spec F.8 says "Fourier masking 50% probability, mask_ratio=0.15" but does not specify a *band* parameter. Upstream `FakeImageDetection/train.py` defaults to `band='all'` (train.py:307); upstream's shipped `mask_15/rn50ft_fouriermask.pth` was trained at `band='all'`. Our `train_rgb_fourier.py:97` hard-codes `band='low+high'`. Is this benign (different name, identical behavior) or material (different masking, different model)?

## Source of truth

`phases/forensic/external/FakeImageDetection/mask.py:21-128` -- the `FrequencyMaskGenerator` class. The relevant logic is `_create_balanced_mask(height, width)`, which iterates over `self.band.split('+')` and for each band-name accumulates `(y, x)` indices to zero out in the frequency map.

## Band-name -> rectangular region map (for a 224x224 input)

`mask.py:71-82`:

```python
if band == 'low':
    y_start, y_end = 0, height // 4              # 0 .. 56
    x_start, x_end = 0, width // 4               # 0 .. 56
elif band == 'mid':
    y_start, y_end = height // 4, 3 * height // 4    # 56 .. 168
    x_start, x_end = width // 4, 3 * width // 4      # 56 .. 168
elif band == 'high':
    y_start, y_end = 3 * height // 4, height     # 168 .. 224
    x_start, x_end = 3 * width // 4, width       # 168 .. 224
elif band == 'all':
    y_start, y_end = 0, height                   # 0 .. 224
    x_start, x_end = 0, width                    # 0 .. 224
```

The frequency map is the 2D FFT shape `(H, W)` per channel; the indexing is the raw FFT layout (corner-indexed, NOT centered), so:

- `'low'`  = top-left  56x56 corner (low-frequency content)
- `'high'` = bottom-right 56x56 corner (high-frequency content)
- `'mid'`  = inner 112x112 square
- `'all'`  = entire 224x224 frequency map

## Indices-zeroed count comparison (for 224x224)

For each band, `mask.py:86-87` computes the per-region area and then `np.ceil(region_area * ratio)` indices to zero. Then `mask.py:107-109` randomly picks that many from that region (without replacement).

| Band | Region area | Indices zeroed (at ratio=0.15) | Fraction of full map |
|---|---|---|---|
| `'all'` | 50,176 (= 224 x 224) | ceil(50,176 * 0.15) = **7,527** | **15.0%** |
| `'low'` | 3,136 (= 56 x 56) | ceil(3,136 * 0.15) = 471 | 0.94% |
| `'high'` | 3,136 | 471 | 0.94% |
| `'mid'` | 12,544 (= 112 x 112) | ceil(12,544 * 0.15) = 1,882 | 3.75% |
| `'low+high'` | 3,136 + 3,136 = 6,272 (two disjoint corners) | 471 + 471 = **942** | **1.88%** |
| `'low+mid'` | 15,680 | 471 + 1,882 = 2,353 | 4.69% |
| `'mid+high'` | 15,680 | 2,353 | 4.69% |
| `'low+mid+high'` | 18,816 (= 6272 + 12544) | 471 + 1,882 + 471 = 2,824 | 5.63% |

Notes:
- `'low+high'` is NOT equivalent to `'low+mid+high'` -- it deliberately skips the mid 112x112 square.
- `'all'` is NOT equal to `'low+mid+high'` either -- `'all'` covers the entire 224x224 (50,176 pixels) and selects from that whole pool, while `'low+mid+high'` covers 18,816 pixels (the three disjoint regions sum). The corners outside those three rectangles (NE and SW 56x112 strips) are never touched by `'low+mid+high'`.

## Behavioral difference summary

| Property | `band='all'` | `band='low+high'` |
|---|---|---|
| Total indices zeroed (at ratio=0.15) | 7,527 | 942 |
| Fraction of 224x224 map masked | 15.0% | 1.88% |
| Hits mid-band frequencies? | YES (mid is part of 'all') | NO (mid skipped entirely) |
| Random selection scope | uniform over 50,176 pixels | uniform within each 56x56 corner |
| What the model learns to ignore | Generic per-pixel frequency dropouts | Strong corner-frequency dropouts; mid-band features remain intact |

**They are NOT equivalent.** `'all'` provides ~8x more masking pixels per pass AND covers the mid band; `'low+high'` provides much less masking AND deliberately preserves the mid-band content. From an augmentation-theory standpoint these produce models with very different inductive biases:

- A `'low+high'`-trained model learns to detect synthetic-image artifacts in the **mid frequency band** (because that band is consistently present through training).
- An `'all'`-trained model learns to detect artifacts that survive **uniform random frequency dropout** across the whole spectrum.

## Why we shipped `'low+high'`

`train_rgb_fourier.py:96-98`:

```python
self.gen = FrequencyMaskGenerator(
    ratio=ratio, band="low+high", transform_type="fourier", channel="all"
)
```

A comment on line 92-95 says: "Doctor's spec says Fourier masking. Defaults match the repo's `FrequencyMaskGenerator(ratio, band='low+high', transform_type='fourier', channel='all')`". But this is wrong: the **class-default** in `mask.py:22` is `band='low+high'`, but upstream's **train.py CLI default** is `band='all'` (line 307), and the shipped `mask_15/rn50ft_fouriermask.pth` was trained at `band='all'`. We picked the class default; we should have picked the CLI default to match the inheriting ckpt.

This is the F-A8 deviation documented in `docs/DECISIONS.md`. Effect: our shipped `forensic_rgb_model.pth` (Approach 1, val AP 0.9971 on the 6/8-generator GenImage subset) was trained with a strictly weaker augmentation than upstream's recipe demanded.

## Predicted effect of switching to `band='all'`

Hypothesis (to verify in Stage C-lite):

1. `'all'` augmentation -> model less reliant on mid-band-only signal -> potentially **better OOD robustness** on the MMFakeBench transfer probe (where image distribution shifts hard).
2. `'all'` augmentation provides 8x more masked pixels per pass -> potentially **slower convergence** but **higher peak val AP** at convergence.
3. In-distribution per-generator AP may go slightly down (0.9971 -> ~0.97-0.99) because the augmentation is harsher, but external/MMFB transfer may go up.

These are predictions; Stage C-lite (single retrain at `band='all'`, same hyperparams as the shipped run) will produce the empirical delta.

## Verdict

`band='all'` and `band='low+high'` produce **materially different** masking patterns and **materially different** trained models. The two are not interchangeable. The retrain experiment in Stage C-lite is justified; the audit confirms the experiment matters.

## Provenance

- Mask source: `phases/forensic/external/FakeImageDetection/mask.py:21-128`
- Our trainer hard-coding: `phases/forensic/scripts/train_rgb_fourier.py:96-98`
- Upstream CLI default: `phases/forensic/external/FakeImageDetection/train.py:307`
- Shipped Approach-1 ckpt: `phases/forensic/outputs/rgb/forensic_rgb_model.pth` (val AP 0.9971, band='low+high')
- Decision register: `docs/DECISIONS.md` entry F-A8
