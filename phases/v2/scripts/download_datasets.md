# Dataset Download Guide

Download each dataset into `data/raw/`. All three require filling forms or accepting licenses.

## 1. DGM4
- Paper: arXiv 2304.02556
- Source for Real (origin) and Manipulated (face_swap, face_attribute) classes
- HuggingFace: huggingface.co/datasets/rshaojimmy/DGM4
- Expected path: `data/raw/DGM4/`

## 2. NewsCLIPpings
- Paper: arXiv 2104.05893
- Source for OOC pairs and Real genuine pairings
- Repo: https://github.com/g-luo/news_clippings
- Expected path: `data/raw/NewsCLIPpings/`

## 3. MMFakeBench (transfer evaluation only)
- Paper: arXiv 2406.08772
- Used for out-of-distribution transfer evaluation
- HuggingFace: huggingface.co/datasets/liuxuannan/MMFakeBench
- Expected path: `data/raw/MMFakeBench/`

## After downloading

```bash
# Phase 1: build balanced OOC dataset and train
python scripts/build_balanced_dataset.py
python -m src.train --config config/v1_ooc.yaml

# Phase 2: build 3-class dataset, precompute features, train
python scripts/build_forensic_3class_clean.py
python scripts/precompute_dct.py
python scripts/precompute_fnd_features.py
python src/train_full_pipeline.py
```
