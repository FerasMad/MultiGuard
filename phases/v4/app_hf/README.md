---
title: MultiGuard
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: 5-class fake-news + binary AI-image (honest-path ensemble)
---

# MultiGuard

Multimodal fake-news detector. 5 classes: Real / Out-of-Context / Manipulated / AI-Text / Fully-Fabricated.

Uses Gradio SDK so ZeroGPU works, but mounts a FastAPI app inside Gradio to serve the same bilingual EN/AR website UI from `static/` at the root of the Space. The Gradio interface itself is hidden at `/gradio`.

## Numbers (V4 honest-path 3-seed ensemble)

| Config | Test F1 | MMFakeBench transfer F1 |
|---|---|---|
| 3-seed mean +/- std | 0.7117 +/- 0.0095 | 0.3757 +/- 0.0479 |
| Ensemble (softmax avg) | **0.7149** | 0.4308 |
| Ensemble + bias correction | 0.7149 | **0.7197** |

## Stack

- FND-CLIP V1 semantic (`FerasMad/multiguard-v1-fndclip`)
- DCT-Forensic image (`FerasMad/forensic-dct-v1`) + P9.1 parity head
- Qwen2-7B-Instruct text
- V3PairwiseFusion + MLP (3 seeds from `FerasMad/multiguard-v4-honest`)

## API

`POST /api/analyze` with multipart `text` + `image` -> verdict + probabilities (EN/AR).

Repo: https://github.com/FerasMad/MultiGuard
