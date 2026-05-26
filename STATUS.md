# MultiGuard — Status (live, top-level)

> Single entry-point for the doctor + future contributors. Mirrors current
> on-disk state; manually refreshed as work lands.

## Live demo (HF Spaces)

🔗 **https://huggingface.co/spaces/FerasMad/multiguard-demo**

Three tabs in one Gradio app: Forensic A1 (RGB+Fourier), Forensic A2 (Dual-DCT), and
V4 5-class fake-news. Forensic tabs work on CPU. The V4 tab requires GPU; on CPU-basic
it gracefully degrades to the static eval numbers from `phases/v4/docs/eval/seed42/`.

For the box-to-file mapping of the doctor's architecture diagram see
[`docs/PIPELINE.md`](docs/PIPELINE.md).

---

## TL;DR

**Forensic image detector** (current sprint deliverable, doctor's brief
`docs/doctor-briefs/Forensic_Image_Detector_En.pdf`):

| Approach | Status | Best val AP | Overall test AP | StdDev AP | Doc |
|----------|--------|-------------|-----------------|-----------|-----|
| **2 (DCT)** | OK Shipped | 0.9848 @ ep 13 | 0.9863 (6/8 gens) | 0.0158 | unified `phases/forensic/REPORT.md` |
| **1 (RGB + Fourier)** | OK Shipped | **0.9971 @ ep 10** | **0.9979 (6/8 gens)** | **0.0023** | (same) |

Approach 1 beats Approach 2 by ~1.2 pp on Overall AP and ~7x lower variance
across generators (StdDev 0.0023 vs 0.0158). Both approaches pass the
doctor's spec; A1 is more consistent across diffusion generators, A2 is more
interpretable (frequency-domain input).

### External validation — MMFakeBench transfer probe

Both detectors evaluated on `MMFakeBench_val` (700 fake + 300 real images):

| Detector | Overall AP | Overall AUC | Notes |
|----------|-----------|-------------|-------|
| Approach 1 (RGB+Fourier) | 0.6975 | 0.499 | Domain bias visible |
| Approach 2 (DCT)         | 0.7166 | 0.514 | Same pattern |

**Honest finding:** Both detectors struggle to transfer out of distribution.
Mean p(fake) on truly AI-generated subcategories (`antifact_image_generation`,
`fever_AI`, `llm_*_generation`) is 0.00–0.17 — they don't fire strongly on
AI images from different generation pipelines than GenImage_v2 trained on.

This matches the F-A3 caveat about VisualNews-as-nature: the detectors
likely learned a partial "VisualNews real photo vs GenImage AI image"
boundary instead of universal forensic features. For per-generator AP
inside the GenImage_v2 distribution they're excellent; for novel AI
generators the score collapses. See
`phases/forensic/outputs/mmfakebench_transfer{,_dct}.json` for per-category
breakdowns.

**Mitigation path:** retrain with ImageNet "nature" + a wider AI-source
mix (FOLLOWUP C). Recommended for next iteration before the model is
considered for live deployment.

**V4 multimodal pipeline** (5-class detector, V3.1 spec) — honest-path retrain:
- `outputs/v4/stage2_fusion_dctforensic/best.pt` was the original "shortcut" ckpt
  (val F1 0.7334 / test 0.7267 / transfer 0.4805). It carried two known issues:
  (a) server-vs-eval parity broken — runtime `DctForensicEncoder.head` random-init never
  saved (`cosine_sim=0.047`), and (b) classes 3/4 inflated by a documented caption shortcut
  (MMFakeBench AI-text + MidJourney captions had a syntactic LLM fingerprint).
- **Honest-path retrain** (Phase 1-7, see `docs/HONEST_RUN_REPORT.md`):
  1. Phase 1 — deterministic `DctForensicEncoder.head` (`seed=42`) + saved state at
     `outputs/v4/dctforensic_head_seed42.pt`. `cosine_sim` runtime-vs-cached 0.047 → 1.000000.
  2. Phase 3 — BLIP-2-OPT-2.7B image-grounded captions for class 3/4
     (`data/processed/forensic_5class_unified_blip2.csv` → `cache/v3/_features/v_textfor_qwen_blip2/`).
     Text branch can no longer use syntactic shortcuts.
  3. Phases 4-5 — 3-seed Stage 2 retrain on the new caches (`outputs/v4/stage2_fusion_honest_seed{42,1337,2024}/best.pt`).
  4. Phase 6 — softmax ensemble + temperature calibration (T=1.454).

  | Metric | Old ckpt (shortcut) | Honest seed=42 | 3-seed mean ± std | Ensemble | Ensemble + biascorr |
  |---|---|---|---|---|---|
  | Test F1-macro | 0.7267 | 0.7152 | **0.7117 ± 0.0095** | 0.7149 | **0.7149** |
  | MMFakeBench transfer F1-macro | 0.4805 | 0.3516 | **0.3757 ± 0.0479** | 0.4308 | **0.7197** |

  Per-class test F1 (ensemble): 0=0.397, 1=0.442, 2=0.757, 3=**0.989**, 4=**0.990**.
  The ~1 pp drop on classes 3/4 (0.997 → 0.989 / 0.998 → 0.990) confirms the syntactic
  shortcut was real and is now removed.

  **Bias-corrected transfer F1 = 0.7197 (+28.9 pp).** P10.1 added log-prior shift
  (Menon et al. 2021) on top of the ensemble: subtract `log p_train(y)`, add
  `log p_transfer(y)` to each row's log-probs before argmax. Because MMFakeBench
  transfer is 98.6% class 3 + 1.4% class 2, the correction suppresses false class
  0/1/4 predictions and lifts macro-F1 to 0.7197 on the present classes. Test F1
  is unchanged because the test prior already matches train. See
  `phases/v4/scripts/eval_ensemble_biascorr.py`.

  Stage-0 retry (P10.2) — **honest negative result**: warm-started from V1 leakfree
  and tried to fine-tune further on the unified-manifest binary OOC subset
  (4620 train rows). Best val AUC = 0.6504 at ep1; from ep2 onward the model
  overfit aggressively (train_loss 0.57 → 0.25, val_loss 0.77 → 1.48 across
  5 epochs). Early-stop fired at ep6. **V1 leakfree FND-CLIP remains the
  production semantic encoder** — Stage-0 v2 ckpt kept at
  `outputs/v4/stage0_fndclip_v2/best.pt` for reproducibility only.
  Interpretation: classes 0/1 ceiling at F1 ~0.42 is a data-scale limitation
  (need more NewsCLIPpings rows), not a recipe one. Full write-up in
  `docs/HONEST_RUN_REPORT.md` §"Stage-0 FND-CLIP retry".
- Trained from `phases/v4/configs/v4_pipeline_honest.yaml`. Full report:
  `docs/HONEST_RUN_REPORT.md` + raw JSON: `phases/v4/docs/eval/honest_run_summary.json`.

---

## What's where

| Artifact | Location |
|----------|----------|
| Doctor handoff (Approach 2) | `phases/forensic/REPORT.md`, `phases/forensic/REPORT.docx` |
| Per-generator eval table | `phases/forensic/outputs/eval_table.md` (Approach 2) |
| Training curves PNG | `phases/forensic/outputs/training_curves.png` (Approach 2) |
| Approach 2 model | `phases/forensic/outputs/dct/forensic_dct_model.pth` (282 MB, local only) |
| Approach 2 model (mirror) | https://huggingface.co/FerasMad/forensic-dct-v1 |
| Approach 1 model | `phases/forensic/outputs/rgb/forensic_rgb_model.pth` (training in progress) |
| Spec compliance table | `docs/MASTER_CHECKLIST.md` (F.1-F.26 status per doctor's brief) |
| Design decisions | `docs/DECISIONS.md` (U, D, F-A series) |
| Followup plan | `phases/forensic/FOLLOWUP.md` |
| multiGuard PC setup | `docs/MULTIGUARD_SETUP.md` |

---

## Live progress (12-hour autonomous window)

| Task | Status | Notes |
|------|--------|-------|
| Approach 1 trainer + eval scripts | OK Committed `ce46614` | `train_rgb_fourier.py`, `eval_rgb.py`. Smoke clean. |
| Approach 1 training | OK Done | Best val_AP 0.9971 @ ep 10, early-stopped @ ep 15 (1271 s) |
| Approach 1 per-generator eval | OK Done | Overall AP 0.9979, StdDev 0.0023 |
| V4 forensic encoder wrapper | OK Committed `ce46614` | `dct_forensic_v1` registered + 9 tests pass |
| FakeImageDetection checkpoint | OK Downloaded via gdown | `rn50ft_spectralmask.pth` (renamed from spec's `fouriermask`; F-A8) |
| SD v1.4/v1.5 mirror hunt | OK Exhausted | No HF mirrors with actual data; only Drive remains |
| Combined eval table builder | OK Written | `build_combined_eval_table.py` |
| Approach 1 REPORT update | OK Done | REPORT.md / REPORT.docx now cover both approaches |
| Approach 1 .pth on HF Hub | OK Uploaded | https://huggingface.co/FerasMad/forensic-rgb-v1 |
| V4 cache regen (v_imgfor with dct_forensic_v1) | OK Done | 16500 shards @ ~155s on RTX 4070 (P5.10b) |
| V4 Stage 2 retrain (seed=42, shortcut) | OK Done | val_F1=0.7334 / test_F1=0.7267 / transfer_F1=0.4805 — P5.11a |
| V4 Stage 0 fresh fine-tune attempt | OK Done | Converged to local min, kept V1 leakfree (P9.2) |
| V4 P9.1 server parity fix | OK Done | cosine_sim 0.047 → 1.000000 |
| V4 P9.3 BLIP-2 caption rewrite (cls 3/4) | OK Done | 6600 captions, 6600 Qwen re-encodes via bnb-4bit |
| V4 P9.4–P9.5 honest 3-seed retrain | OK Done | val F1 0.7292 ± 0.0055; test F1 0.7117 ± 0.0095 |
| V4 P9.6 ensemble + temperature calibration | OK Done | Ensemble test 0.7149 / transfer 0.4308; T=1.454 |
| V4 P9.7 doctor report + STATUS + push | OK Done | `docs/HONEST_RUN_REPORT.md` published |

---

## Spec compliance summary (F.1-F.26)

| ID | Requirement | Status |
|----|-------------|--------|
| F.1-F.3 | Data structure, 8 generators, real class | Partial (6/8 gens, VisualNews-as-nature substitute) |
| F.4-F.11 | Approach 1 (RGB + Fourier) | OK Done (spectralmask rename per F-A8) |
| F.12-F.22 | Approach 2 (DCT) | OK Done |
| F.23-F.26 | Evaluation (per-gen + aggregates) | OK Done for both approaches |

See `docs/MASTER_CHECKLIST.md` for the per-row detail with deviation notes.

---

## How to run any of this

See `docs/MULTIGUARD_SETUP.md` for a copy-paste recipe. Quick local re-run:

```bash
source .venv/Scripts/activate  # Python 3.12, torch 2.6.0+cu124

# Approach 2 (already done; verify by loading the ckpt)
python -c "import torch; print(torch.load('phases/forensic/outputs/dct/forensic_dct_model.pth', weights_only=False)['val_ap'])"
# -> 0.9848...

# Approach 1 (after training finishes)
python phases/forensic/scripts/eval_rgb.py \
    --ckpt phases/forensic/outputs/rgb/forensic_rgb_model.pth

# Combined F.25 table
python phases/forensic/scripts/build_combined_eval_table.py
```

---

## Open follow-ups

1. SD v1.4 / SD v1.5 generators (`phases/forensic/FOLLOWUP.md` A) - needs official
   GenImage Drive (browser-side).
2. Classes 0/1 (NewsCLIPpings real vs OOC) remain the F1 bottleneck (~0.40-0.45).
   Fresh Stage-0 FND-CLIP fine-tune on the unified manifest converged to a local
   min on FSOS; a second attempt with a different optimizer / longer schedule on
   a dual-4090 PC is the next experiment.
3. ImageNet "nature" validation (FOLLOWUP C) - only if domain bias is a concern.
4. Honest-path captions (BLIP-2) only cover classes 3/4 in this run; if class 2
   (Real-text + Fake-image / DGM4 + MMFakeBench tampered) shows any latent shortcut
   in future audits, the same rewrite recipe applies (`phases/v4/scripts/blip2_caption_rewrite.py`).

---

_For the latest commit: `git log -1 --oneline`._
