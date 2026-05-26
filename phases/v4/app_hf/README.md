---
title: MultiGuard — Forensic + 5-class Fake-News Detector
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: V4 5-class fake-news + two forensic image detectors
---

# MultiGuard — live demo

Three tabs in one Space:

| Tab | What it does | Model |
|-----|--------------|-------|
| **Forensic A1** | RGB + Fourier-mask binary fake-image detector | `FerasMad/forensic-rgb-v1` (ResNet50, ~260 MB) |
| **Forensic A2** | Dual-DCT (8x8 + 16x16) binary fake-image detector | `FerasMad/forensic-dct-v1` (ResNet50 + 1-ch conv1, ~270 MB) |
| **V4 5-class** | Multimodal Real / OOC / Manipulated / AI-Text / Fully-Fabricated | FND-CLIP + DCT forensic + Qwen2-7B-Instruct + V3 pairwise fusion |

## What's inside

- Two forensic image detectors implemented per the doctor's brief
  (`Forensic_Image_Detector_En.pdf`). Both pass the per-generator AP/Acc/AUC
  evaluation. Approach 1 is more consistent, Approach 2 is more interpretable.
- V4 5-class detector is the V3.1-spec multimodal fusion pipeline retrained
  with the new DCT forensic encoder; val F1-macro = 0.7334, test F1-macro =
  0.7267, MMFakeBench transfer F1-macro = 0.4805.

The V4 5-class tab pulls Qwen2-7B-Instruct on first call (~15 GB) and runs
on Spaces ZeroGPU (A100). First load ~30 s; subsequent calls ~5 s. The two
forensic tabs run on CPU and respond in well under a second.

For full repo + training scripts + checkpoints + reports see
[`FerasMad/MultiGuard`](https://github.com/FerasMad/MultiGuard).

## Honest limitations

- V4 5-class is single-seed (42); the multi-seed and fresh Stage-0 sweep
  is deferred to a real GPU.
- The Class 3 / Class 4 captions in the V3-era text cache contain a partial
  shortcut, so those two classes are near-perfect at test time; treat with
  appropriate skepticism.
- Both forensic detectors were trained with VisualNews-as-nature (no
  ImageNet "nature"); the MMFakeBench OOD AP collapses outside the
  training distribution, see the report for details.

Built by Feras Madkhali. Model + data + code released under MIT.
