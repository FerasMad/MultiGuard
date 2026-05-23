# MultiGuard

Multimodal fake-news detection — 5-class classifier (Real / Out-of-Context / Manipulated / AI-Text / Fully Fabricated) over text + image inputs, built per the V3.1 Implementation Guidelines.

## Quick start

```bash
git clone https://github.com/FerasMad/MultiGuard.git
cd MultiGuard

# 1. Stage raw datasets at data/raw/  (DGM4, MMFakeBench, NewsCLIPpings, VisualNews, GenImage, blur_jpg_prob0.pth, Qwen2-7B HF cache)
#    Acquisition is environment-specific — see `docs/SETUP.md` Step 6 for the per-source list.

# 2. Install + build + train + serve
make setup        # package + dev/gpu/server extras
make data         # build manifests + leakage audit
make precompute   # feature caches (patch-DCT, FND-CLIP, Qwen)
make train        # Stage 0 -> Stage 1 -> Stage 2 (3 seeds)
make eval         # V3.1 section 7 in-distribution + MMFakeBench transfer
make server       # FastAPI on port 8081
```

## Documentation

| Doc | What's inside |
|-----|---------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layered package, data flow, registry pattern, forced deviations |
| [`docs/SETUP.md`](docs/SETUP.md) | Windows + Python 3.12 + CUDA 12.4 install, dependency layout, troubleshooting |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Git workflow, lint/format, MANDATORY Windows workarounds |
| [`docs/ADD_ENCODER.md`](docs/ADD_ENCODER.md) | Swap or add an encoder in ~80 LOC |
| [`docs/ADD_DATASET.md`](docs/ADD_DATASET.md) | Build a new data source against the canonical manifest schema |
| [`docs/ADD_FUSION.md`](docs/ADD_FUSION.md) | Add a fusion module under `FusionBase` |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | V3.1 section 7 evaluation + risks accepted |
| [`docs/SPEC_COMPLIANCE_MAP.md`](docs/SPEC_COMPLIANCE_MAP.md) | V3.1 section -> file:symbol audit table |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 15 locked design decisions with rationale |
| [`CHANGELOG.md`](CHANGELOG.md) | V1 -> V2 -> V3 -> V4 timeline |

## Spec sources (canonical)

- Implementation Guidelines V3.1 — drives the 5-class architecture
- Implementation Guidelines V2 — predecessor 3-class baseline
- Dataset + V1 Instructions — V1 baseline + dataset construction rules

`docs/SPEC_COMPLIANCE_MAP.md` is the doctor-audit table mapping every V3.1 section to the file that implements it.
