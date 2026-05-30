# MultiGuard — Full Model Report

> Consolidated report on the deployed MultiGuard model: architecture, what
> analyzes the text, full 5-class + forensic numbers, and a **live test of the
> HuggingFace Space** (all three tabs exercised in a real browser). Generated
> May 2026.

🔗 **Live demo:** https://huggingface.co/spaces/FerasMad/multiguard-demo
📦 **Repo:** https://github.com/FerasMad/MultiGuard

---

## 1. What the model is

MultiGuard is a **5-class multimodal fake-news detector** (V3.1 spec) plus two
**standalone binary AI-image forensic detectors**. The 5-class pipeline fuses
three independent branches:

```
 TEXT  ─► Qwen2-7B-Instruct ──► v_textfor (3584-d) ─┐
 IMAGE ─► Dual-DCT ResNet50 ──► v_imgfor  (768-d)  ─┼─► V3PairwiseFusion ─► MLP ─► 5 classes
 BOTH  ─► FND-CLIP (ResNet50+BERT+CLIP) ─► v_semantic (512-d) ─┘   (3-pair MHA-SUM →
                                                                   Conv1d → 1024-d)
```

Output classes: **0 Real · 1 Out-of-Context · 2 Manipulated · 3 AI-Text ·
4 Fully-Fabricated**. Deployed as a **3-seed ensemble** (seeds 42/1337/2024,
softmax-averaged).

## 2. What analyzes the text  ← (your question)

Two components read the text:

| Component | Role | What it catches |
|-----------|------|-----------------|
| **Qwen2-7B-Instruct** (the *text-forensic* branch) | text → tokenize (left-pad, ≤512) → **last hidden layer** → **masked-mean pool** over real tokens → 3584-d → `Linear(3584→768)+GELU` | **Is the text AI-written?** Drives classes 3 (AI-Text) and 4 (Fully-Fabricated). |
| **BERT-base** (inside the *FND-CLIP semantic* branch) | text → BERT → fused with the image via CLIP + modality attention | **Does the text match the image?** Drives class 1 (Out-of-Context). |

So: **Qwen2-7B = "is this AI-written?"**, **BERT/FND-CLIP = "does it match the
image?"**. (Spec deviation F2: the V3.1 PDF named Qwen layer 30 / hidden 4096,
which matches Qwen1.5; we use Qwen2-7B-Instruct → last layer / hidden 3584,
documented + unit-tested in `docs/TEXT_BRANCH_AUDIT.md` + `test_qwen_pooling.py`.)

## 3. 5-class performance (deployed 3-seed honest ensemble)

**Test split (2,475 held-out samples) — F1-macro = 0.7149**

| Class | Precision | Recall | F1 | n |
|-------|-----------|--------|-----|---|
| 0 Real | 0.428 | 0.370 | 0.397 | 495 |
| 1 Out-of-Context | 0.447 | 0.438 | 0.442 | 495 |
| 2 Manipulated | 0.707 | 0.814 | 0.757 | 495 |
| 3 AI-Text | 0.984 | 0.994 | **0.989** | 495 |
| 4 Fully-Fabricated | 0.994 | 0.986 | **0.990** | 495 |

**Confusion matrix** (rows = true, cols = predicted):

|        | Real | OOC | Manip | AI-Text | Fab |
|--------|------|-----|-------|---------|-----|
| **Real**  | 183 | 227 | 85 | 0 | 0 |
| **OOC**   | 196 | 217 | 82 | 0 | 0 |
| **Manip** | 49 | 42 | 403 | 1 | 0 |
| **AI-Text** | 0 | 0 | 0 | 492 | 3 |
| **Fab**   | 0 | 0 | 0 | 7 | 488 |

**Reading it:** the **text-driven classes (3, 4) are near-perfect** and never leak
into 0/1/2 — the Qwen branch cleanly isolates AI-written text. The **error mass is
entirely in Real ↔ OOC** (183/227 and 196/217): the model cannot reliably tell a
genuine image-caption pair from a mismatched one, because both come from
NewsCLIPpings and discriminating them needs strong cross-modal alignment that the
**frozen FND-CLIP can't provide** (audited in `docs/FNDCLIP_REAL_OOC_AUDIT.md` —
linear-probe AUC ≈ 0.51, i.e. at-chance). This is a **data/encoder ceiling, not a
fusion bug**, and it's the project's main honest limitation.

**External transfer (MMFakeBench, zero-shot):** raw F1 **0.4308**; with a
log-prior bias correction for the transfer set's class imbalance, F1 **0.7197**.

## 4. Forensic image detectors (8/8 generators, official ImageNet nature)

Two binary real-vs-AI detectors, retrained on **all 8 GenImage generators** with
**genuine ImageNet nature** (F-A3 fully resolved — see
`docs/F_A3_OFFICIAL_NATURE.md`). Per-generator AP (`outputs/eval_table_combined.md`):

| Generator | A1 RGB+Fourier AP | A2 DCT AP |
|-----------|-------------------|-----------|
| midjourney | 0.965 | 0.828 |
| sdv1_4 | 0.988 | 0.890 |
| sdv1_5 | 0.985 | 0.896 |
| wukong | 0.980 | 0.902 |
| vqdm | 0.976 | 0.830 |
| adm | 0.997 | 0.971 |
| glide | 0.987 | 0.975 |
| biggan | 0.995 | 0.976 |
| **Overall AP** | **0.9841** | **0.9085** |
| **Overall Acc** | 0.940 | 0.834 |
| **Std-dev AP** | 0.011 | 0.061 |

**A1 (RGB+Fourier) is the recommended detector** — strong and consistent
(AP ≥ 0.965 on every generator). **A2 (DCT) is the honest frequency-domain
baseline** — weaker, especially on MidJourney/VQDM.

## 5. Live HuggingFace Space test (real browser, all 3 tabs)

Driven via Playwright against the running Space. Test images: official GenImage
held-out samples.

| Tab | Input | Live Space output | Correct? |
|-----|-------|-------------------|----------|
| **Forensic A1 (RGB)** | MidJourney AI image | **AI-Generated, P(AI)=94.3%** | ✅ |
| **Forensic A2 (DCT)** | same MidJourney AI image | Real, P(AI)=0.6% | ❌ (DCT's known MidJourney weakness) |
| **Forensic A1/A2** | real ImageNet photo | Real, P(AI)=0.0% / 0.0% | ✅ |
| **5-Class** | AI image + plausible caption | **Real 71.4%** (OOC 28.3 / Manip 0.3 / AI-Text 0 / Fab 0) | conservative call* |

\* The 5-class call used an AI MidJourney image + a *custom plausible caption*.
With neither an obviously-AI text (Qwen) nor a strong image-forensic signal (DCT
is weak on MidJourney), the model stays conservative → "Real". This illustrates
that class-4 detection on arbitrary inputs leans on the **text** looking AI-written.

**Deployment parity verified:** running the exact Space inference code locally on
the same images reproduced the live outputs **to the decimal** (A1 94.3%, A2 0.6%),
confirming the deployed Space is byte-faithful to the trained weights. A
100-image/class accuracy sample: A1 RGB 0.96–1.00, A2 DCT 0.75–0.96 (weaker on AI).

**All three tabs are live and functional.** The 5-class path is unchanged from the
shipped honest ensemble (parity preserved); the forensic tabs serve the new 8/8
official detectors.

## 6. Honest limitations

1. **Real vs OOC is at-chance** (F1 ≈ 0.40/0.44) — frozen FND-CLIP can't do the
   cross-modal alignment; a data-scale/encoder ceiling, not a bug.
2. **Classes 3/4 are near-perfect** partly because AI-text has strong syntactic
   signal; on arbitrary, human-plausible AI text the margin shrinks.
3. **DCT detector (A2) is weak on MidJourney/VQDM** (AP ~0.83) — prefer A1.
4. **Zero-shot transfer** needs the log-prior bias correction to be meaningful.
5. **5-class single-input behavior** is less reliable than the aggregate F1
   (0.71) suggests — it is a research model, not a production fact-checker.

## 7. Provenance

- Deployed 5-class ckpts: `FerasMad/multiguard-v4-honest` (seeds 42/1337/2024) +
  `FerasMad/multiguard-v1-fndclip` + `FerasMad/forensic-dct-v1` (encoder).
- Forensic tab ckpts: `FerasMad/forensic-rgb-v1`, `FerasMad/forensic-dct-8gen`.
- Metrics: `outputs/v4/stage2_fusion_honest_ensemble/`, `phases/forensic/outputs/eval_table_combined.md`.
- Full audits: `docs/*_AUDIT.md`, `docs/IMAGE_BRANCH_ABLATION.md`, `docs/F_A3_OFFICIAL_NATURE.md`.
