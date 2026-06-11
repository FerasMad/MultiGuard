# MultiGuard

Multimodal fake-news detector (5 classes) + binary AI-image forensic detector.
KSU graduation project, Feras Madkhali.

## Live demo

https://huggingface.co/spaces/FerasMad/multiguard-demo
( currently unactive )

## Numbers

**V4 5-class pipeline** (FND-CLIP + DCT-Forensic + Qwen2-7B + V3PairwiseFusion, honest-path retrain):

| Config | Test F1 | MMFakeBench transfer F1 |
|---|---|---|
| 3-seed mean +/- std | 0.7117 +/- 0.0095 | 0.3757 +/- 0.0479 |
| Ensemble | 0.7149 | 0.4308 |
| Ensemble + bias-correction | **0.7149** | **0.7197** |

**Forensic image detector** (binary, GenImage 6/8 generators):

| Approach | Test AP | StdDev |
|---|---|---|
| RGB + Fourier mask | 0.9979 | 0.0023 |
| Dual-DCT | 0.9863 | 0.0158 |

## Repo layout

```
phases/v1/        FND-CLIP binary OOC baseline (frozen)
phases/v2/        3-class forensic fusion (frozen)
phases/v3/        5-class Qwen pipeline (frozen)
phases/v4/        Active 5-class rebuild (canonical)
phases/forensic/  Binary AI-image detector
app/              FastAPI server
docs/             Doctor briefs + checklists + reports
tools/            Utilities
```

## Run locally

```bash
git clone https://github.com/FerasMad/MultiGuard.git
cd MultiGuard
python -m venv .venv && .venv/Scripts/Activate.ps1
pip install -e ".[gpu,server,dev]"
make data precompute train eval server
```

## Reports

- `STATUS.md` - live status, open follow-ups
- `docs/HONEST_RUN_REPORT.md` - V4 honest-path retrain results
- `phases/forensic/REPORT.md` - forensic detector handoff

## Tests

```
pytest phases/v4/tests/ phases/forensic/tests/ tests/ -q
# 95 passed
```

## License

MIT.
