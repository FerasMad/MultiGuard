# MultiGuardPipeline

Pluggable, dependency-injection pipeline. Five files, one class each — models are
**injected from outside**, every class exposes an auto-inferred `out_dim`, the
four branch files never import each other, and `main_pipeline.py` is the only
orchestrator.

```
MultiGuardPipeline/
├── __init__.py            # re-exports the 5 classes + build_multiguard_pipeline
├── semantic.py            # SemanticEncoder
├── forensic_text.py       # ForensicTextEncoder
├── image_forensic.py      # ImageForensicEncoder
├── fusion.py              # FusionModule
└── main_pipeline.py       # MainPipeline  (orchestrator) + build_multiguard_pipeline()
```

## Architecture (matches the design diagram)

```
caller / server
      │
      ▼
  MainPipeline ──► SemanticEncoder      ──► FND-CLIP
              ──► ForensicTextEncoder   ──► Qwen2-7B
              ──► ImageForensicEncoder  ──► DCT ResNet50
              ──► FusionModule          ──► V3PairwiseFusion ──► 5-class output
```

## Use it

**Generic (any models, injected by you):**
```python
from MultiGuardPipeline import MainPipeline
pipe = MainPipeline.from_models(
    semantic_model=..., image_model=..., text_model=...,
    fusion_builder=lambda ds, di, dt: MyFusion(ds, di, dt),
)
out = pipe(batch, text=raw_text)   # {'main_logits':[B,5], 'aux_logits':[B,2], 'fused':[B,1024]}
```

**Real MultiGuard models (wired for you):**
```python
from MultiGuardPipeline import build_multiguard_pipeline
pipe = build_multiguard_pipeline(device="cuda")   # FND-CLIP + Qwen2 + DCT + V3PairwiseFusion
out = pipe(batch, text=raw_text)
```

`build_multiguard_pipeline` builds the real backbones via the v4 registry and
injects them; requires the full training environment (torch, transformers,
weights). The five wrapper classes themselves depend only on `torch`.

## Quick demo (no real weights)
```bash
python main_pipeline.py
```
