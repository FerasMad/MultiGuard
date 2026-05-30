# Forensic Detector — Follow-up Plan

Tracks left open after the May 2026 overnight ship of Approach 2. None are blocking
for the current doctor handoff (REPORT.md + REPORT.docx), but they extend the
detector toward the full 8-generator + 2-approach + V4-integrated end state.

---

## A. Close the 2 missing generators (SD v1.4 / SD v1.5) — ✅ RESOLVED (May 2026)

**Status: DONE.** The earlier conclusion ("only remaining path is the GenImage
Google Drive") was wrong — a working HuggingFace source was found:

- **`shimei123/Genimage`** hosts each generator as a discrete zip. `SD_v14.zip`
  (3.55 GB) and `SD_v15.zip` (4.74 GB) carry the **official GenImage layout**:
  `0_real/ILSVRC2012_*.JPEG` (genuine ImageNet nature) + `1_fake/` (SD-generated).
- Pulled self-serve via `huggingface_hub` (no Drive, no browser auth), staged by
  the new `scripts/stage_sd_generators.py` (1750 ai + 1750 nature each).
- Because the zips ship official ImageNet nature, this **also fixes F-A3** for
  the two SD generators (§C below) — their real class is genuine ImageNet, not
  the VisualNews substitute.

The full chain was re-run end-to-end (not just test re-eval): `stage_sd_generators.py`
→ `build_splits.py` (auto-discovers 8 gens) → `precompute_dct.py` (all splits) →
`compute_dct_stats.py` → **retrain both detectors on 8 generators** →
`eval_dct.py` + `eval_rgb.py`. So this is an apples-to-apples 8-gen retrain, not
an OOD probe with the 6-gen ckpt.

**Result (8/8, canonical `outputs/eval_table_combined.md`):** A1 RGB+Fourier
overall AP **0.9876**, A2 DCT overall AP **0.9468**. SD gens are the hardest
(A1 ~0.97, A2 ~0.85 AP). See REPORT.md §0.

Other mirrors checked (for the record): `genimage-224` (sd15 only, no sd14),
`TheKernel01/Tiny-GenImage` (declares SD14 label but 0 rows), `bitmind/*`
(no SD at all). `shimei123/Genimage` was the one with both as usable zips.

---

## B. Approach 1 (RGB + Fourier mask)

**Status:** Not built. Doctor's brief calls for BOTH approaches. Approach 1
needs the chandlerbing65nm/FakeImageDetection repo (Linux-only shell scripts).

**Effort:** ~6 h (1 h setup + 4 h train + 1 h eval).

**Steps (locked decisions in F-A1 + planned in V4_PLAN.md):**

1. Clone external repo:
   ```bash
   git clone https://github.com/chandlerbing65nm/FakeImageDetection \
       phases/forensic/external/FakeImageDetection
   ```

2. Download checkpoint `mask_15/rn50ft_fouriermask.pth` from the Google Drive
   link in their README. Save to `phases/forensic/external/FakeImageDetection/checkpoints/mask_15/`.

3. Write + run `phases/forensic/scripts/adapt_fakeimagedetection.py`:
   - Patches hardcoded Linux paths (`/mnt/SCRATCH/chadolor/Datasets/...`) to
     `C:/Desktop/MultiGuard/phases/forensic/data/genimage_*` (forward slashes
     work in PyTorch DataLoaders on Windows).
   - Injects `model.change_output(1)` + freeze `conv1`/`bn1`/`layer1`/`layer2`,
     filter AdamW params to `layer3`/`layer4`/`fc` only.
   - Idempotent via marker comment `# FORENSIC_PATCHED v1`.

4. Run `bash phases/forensic/scripts/train_rgb_fourier.sh` via Git Bash on
   Windows. Wraps their `train.sh "0"` with our data paths.

5. Eval: `python phases/forensic/scripts/eval_rgb.py --ckpt phases/forensic/outputs/rgb/best_rgb.pth`.

6. Final combined table: `python phases/forensic/scripts/build_final_table.py`
   joins Approach 1 + Approach 2 results into a unified eval_table.md per F.25
   (rows = 8 gens x cols = AP / Acc / AUC x {RGB, DCT}).

---

## C. Replace VisualNews-as-nature with ImageNet "nature"

**Status:** F-A3 documented deviation. The model may have learned a partial
"news-photo vs AI-image" cue rather than pure forensic frequency artifacts.

**Effort:** ~1 h (data + retrain + eval).

**Validation procedure:**

1. Download ImageNet "nature" subset bundled in the original GenImage Drive
   release (per generator: ~1500 real images at 256x256 paired with the AI ones).
   Save to `data/raw/GenImage_v2/<gen>/nature_imagenet/`.

2. Modify `prepare_genimage_v2.py::sample_visualnews_nature` to read from
   `nature_imagenet/` instead of VisualNews when available.

3. Re-run from `build_splits.py` onward. Same train hyperparameters.

4. Compare per-generator AP between the two runs:
   - If ImageNet-nature AP > VisualNews-nature AP: VisualNews was harder (good,
     means the detector wasn't just memorizing news-photo distribution).
   - If ImageNet-nature AP < VisualNews-nature AP by a wide margin: the VisualNews
     run benefited from a domain-cue shortcut. Doctor handoff should be updated
     with the ImageNet numbers.

---

## D. V4 cache regen with Approach 2 as image-forensic encoder

**Status:** V4 currently uses `blur_jpg_v0.pth` for Stage 1 (UnivFD) init. The
trained `forensic_dct_model.pth` is a stronger frequency-domain detector and
should give better Stage 1 features.

**Effort:** ~1-2 days (V4 cache regen + Stage 2 train + 3-seed eval). Best
done on the dual-4090 multiGuard PC (Stage 2 is the bottleneck at ~5 h/seed).

**Steps:**

1. Wrap `forensic_dct_model.pth` as a V4 `EncoderBase` subclass:
   - New file: `phases/v4/src/v4/models/encoders/dct_forensic.py`
   - Inputs: PIL image -> dual-DCT preprocessing (reuse `forensic.preprocessing.dual_dct`)
   - Pooled layer4 features (2048-dim) -> Linear(2048, 768) adapter -> v_imgfor
   - Register as `@register(ENCODER_REGISTRY, "dct_forensic_v1")` per V4 pattern.

2. Update `phases/v4/configs/v4_pipeline_qwen.yaml`:
   - `encoders.image.type: dct_forensic_v1`
   - `encoders.image.ckpt: phases/forensic/outputs/dct/forensic_dct_model.pth`

3. Precompute new V4 image cache:
   ```bash
   python -m v4 precompute --feature v_imgfor --config phases/v4/configs/v4_pipeline_qwen.yaml
   ```
   (~30 min on RTX 4070 for 15K images.)

4. Reuse existing `cache/v3/v_textfor_qwen/` (3584-dim, V3-built) - V4 fusion's
   `text_proj: Linear(3584, 768)` matches.

5. Need to also build V4 5-class manifest from raw data - see V4_PLAN.md sec 3
   "Dataset acquisition procedure".

6. Train V4 Stage 0 FND-CLIP (~3 h on FSOS PC or ~1.5 h on dual-4090).

7. Train V4 Stage 2 with 3 seeds {42, 1337, 2024} - sequential on single-GPU
   FSOS PC (~15 h) or two-in-parallel on dual-4090 (~7 h).

8. Run V4 sec 7 eval: P/R/F1-macro/CM on test split + MMFakeBench transfer probe.
   Compare against the V3 baselines (test F1 ~ 0.5377, AUC ~ 0.84) - V4 with the
   new image-forensic encoder should improve on those numbers.

---

## E. Optional: ship checkpoints to HuggingFace Hub

The trained `forensic_dct_model.pth` (282 MB) is too big for vanilla git but
small enough for an HF model repo.

**Effort:** ~10 min.

```bash
huggingface-cli upload FerasMad/forensic-dct-v1 \
    phases/forensic/outputs/dct/forensic_dct_model.pth \
    forensic_dct_model.pth
huggingface-cli upload FerasMad/forensic-dct-v1 \
    phases/forensic/data/dct_stats.json \
    dct_stats.json
```

Then add a `download_checkpoint.py` script in scripts/ that fetches from HF for
anyone reproducing.

---

## F. CI hardening (optional, low-priority)

CI is green now (P5.13 fixed). To extend:
- Add `pip install torch --index-url https://download.pytorch.org/whl/cpu` to
  CI so smoke tests actually validate v4 module imports (currently silent on
  ImportError).
- Add a CI step that runs the full forensic test suite - but that needs scipy
  + scikit-learn + Pillow at minimum (already in core deps).
- Set up Dependabot for ruff and pytest version bumps.

---

## G. Tracking

The TaskList items still open after this overnight ship:
- #80 P5.10: V4 cache regen - see section D above.
- #81 P5.11: V4 Stage 0 + Stage 2 training (3 seeds) - see section D above.
- #70 P6: Final verification + commit + push final state - partial; this
  forensic ship counts toward it. The unification work (P0-P4) was the bulk
  of P6's verification surface.
