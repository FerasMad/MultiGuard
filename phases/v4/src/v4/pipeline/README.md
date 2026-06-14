# Pluggable pipeline (`v4.pipeline`)

A dependency-injection rebuild of the pipeline: **models are passed in from the
outside**, never built inside. Five units, one class each:

| File | Class | Role |
|------|-------|------|
| `semantic.py` | `SemanticEncoder` | wraps the semantic branch model |
| `forensic_text.py` | `ForensicTextEncoder` | wraps the text-forensic model |
| `image_forensic.py` | `ImageForensicEncoder` | wraps the image-forensic model |
| `fusion.py` | `FusionModule` | wraps the fusion model |
| `main_pipeline.py` | `MainPipeline` | **the only orchestrator** — wires the four |

## Design rules

1. **Dependency injection** — every class receives its model via `__init__`; it
   never instantiates a model.
2. **Auto `out_dim`** — every class exposes `out_dim`, inferred from the injected
   model (a probe forward is the source of truth; if no example is given it falls
   back to a dimension attribute the model already exposes). No hardcoded dims.
3. **Strict decoupling** — the four branch files never import each other. Only
   `main_pipeline.py` imports them.
4. **Swap a model = edit one file** — pass a different model to the matching
   wrapper; the rest is unchanged.

## Quick demo

```bash
python main_pipeline.py        # runs an end-to-end demo with dummy models
```

## Wiring the real MultiGuard models

The fusion's input widths flow from the encoders' `out_dim`, so nothing is
hardcoded:

```python
import torch
from v4.core.registry import build_encoder, build_fusion, import_all
from v4.pipeline import MainPipeline

import_all()

# build the real models OUTSIDE (registry, YAML, checkpoints, ...)
semantic_model = build_encoder({"type": "fnd_clip",  "feat_dim": 512, "output_dim": 768})
image_model    = build_encoder({"type": "univfd",    "out_dim": 768})
text_model     = build_encoder({"type": "qwen2_7b",  "hidden_size": 3584, "output_dim": 768})

# these encoders already expose `output_dim`, so no example inputs are needed
pipeline = MainPipeline.from_models(
    semantic_model=semantic_model,
    image_model=image_model,
    text_model=text_model,
    fusion_builder=lambda d_sem, d_img, d_txt: build_fusion({
        "type": "v3_pairwise", "feat_dim": d_sem, "num_classes": 5,
    }),
)

print(pipeline.encoder_dims)   # {'v_semantic': 768, 'v_imgfor': 768, 'v_textfor': 768}
print(pipeline.out_dim)        # 5

# inference: one merged batch drives all three branches; raw text -> runtime path
batch = {**fnd_inputs, "v_imgfor_dct": dct}
out = pipeline(batch, text=raw_text)   # {'main_logits': [B,5], 'aux_logits': [B,2], ...}
```

To change, say, the image branch to a different detector, build a different
`image_model` and pass it — `image_forensic.py` is the only thing that conceptually
changes, and even that only via the injected model.
