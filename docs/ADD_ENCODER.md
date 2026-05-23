# Adding a new encoder

This walkthrough shows how to swap or extend an encoder. The doctor's brief is that a future student should be able to swap any single encoder in ~80 LOC. The example below replaces Qwen2-7B with a hypothetical `roberta_large` text encoder.

---

## 1. The contract

Every encoder subclasses `EncoderBase` from `src/v4/models/encoders/base.py`:

```python
class EncoderBase(nn.Module):
    name: ClassVar[str]                          # registry key, e.g. "qwen2_7b"
    required_inputs: ClassVar[tuple[str, ...]]   # keys it expects in the input dict
    output_dim: int                              # output feature dim
    modality: ClassVar[Literal["text", "image", "multi"]]

    def forward(self, batch: dict[str, Tensor]) -> Tensor: ...
    def load_legacy_checkpoint(self, path: str) -> None: ...
```

---

## 2. Create the file

`src/v4/models/encoders/roberta_text.py`:

```python
"""RoBERTa-large text-forensic encoder.

Replaces Qwen2-7B as the v_textfor source. Outputs 1024-dim masked-mean
of the last-layer hidden state, then projects to 768 via a learnable adapter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from v4.core.registry import register, ENCODER_REGISTRY
from v4.models.encoders.base import EncoderBase
from v4.data.preprocessing.tokenizers import masked_mean_pool


@register(ENCODER_REGISTRY, "roberta_large")
class RoBERTaTextEncoder(EncoderBase):
    required_inputs = ("v_textfor_roberta",)
    modality = "text"

    def __init__(
        self,
        model_name: str = "roberta-large",
        hidden_size: int = 1024,
        out_dim: int = 768,
        layer_index: int = -1,
        load_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.output_dim = out_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = (
            AutoModel.from_pretrained(model_name, output_hidden_states=True)
            if load_backbone else None
        )
        self.proj = nn.Sequential(nn.Linear(hidden_size, out_dim), nn.GELU())

    # ---- training-time: cached features ------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Expects a precomputed 1024-d pooled vector per sample.
        x = batch[self.required_inputs[0]]
        return self.proj(x)

    # ---- server-time: raw text ---------------------------------------
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        if self.backbone is None:
            raise RuntimeError("Set load_backbone=True for runtime inference")
        device = next(self.backbone.parameters()).device
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        out = self.backbone(**enc)
        h = out.hidden_states[self.layer_index]            # [1, T, 1024]
        pooled = masked_mean_pool(h, enc["attention_mask"]) # [1, 1024]
        return self.proj(pooled)                            # [1, 768]
```

---

## 3. Register the import

Ensure `src/v4/models/encoders/__init__.py` imports the new module so `import_all()` discovers the decorator side-effect:

```python
from v4.models.encoders import (
    fnd_clip,
    qwen_text,
    roberta_text,   # <-- add
    univfd,
)

__all__ = ["fnd_clip", "qwen_text", "roberta_text", "univfd"]
```

---

## 4. Add a precompute path

`scripts/precompute.py::_precompute_qwen` is a template. Copy it as `_precompute_roberta`:

```python
def _precompute_roberta(df, out_dir, config_path, device):
    import yaml
    if not config_path:
        raise ValueError("v_textfor_roberta requires --config")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    text_spec = dict(cfg["encoders"]["text"])
    text_spec["load_backbone"] = True
    enc = build_encoder(text_spec).to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    pbar = tqdm(df.itertuples(index=False), total=len(df), desc="RoBERTa")
    for row in pbar:
        shard = out_dir / f"{row.sample_id}.pt"
        if shard.exists():
            continue
        try:
            with torch.no_grad():
                pooled = enc.encode_text(str(row.text or "")).squeeze(0).cpu().float()
            torch.save(pooled, shard)
        except Exception as e:
            log.warning("RoBERTa fail %s: %s", row.sample_id, e)
```

Wire it into `run_precompute` for `modality == "v_textfor_roberta"`.

---

## 5. Update the YAML config

`configs/v4_pipeline_roberta.yaml`:

```yaml
data:
  feature_keys:
    - { name: v_semantic, subdir: v_semantic_fnd,     dim: 512 }
    - { name: v_imgfor,   subdir: v_imgfor_dct,       dim: null }
    - { name: v_textfor,  subdir: v_textfor_roberta,  dim: 1024 }    # <-- new cache subdir + dim
encoders:
  semantic: { type: fnd_clip, ckpt: outputs/v4/stage0_fndclip/best.pt, feat_dim: 512 }
  image:    { type: univfd,   ckpt: outputs/v4/stage1_univfd/forensic_model.pth, out_dim: 768 }
  text:     { type: roberta_large, model_name: roberta-large, hidden_size: 1024, layer_index: -1 }
fusion:
  type: v3_pairwise
  feat_dim: 768
  projections:
    v_semantic: { in_dim: 512,  out_dim: 768 }
    v_textfor:  { in_dim: 1024, out_dim: 768 }    # <-- adjust to new text dim
```

---

## 6. Precompute the new cache and retrain

```bash
python -m v4.cli precompute --modality v_textfor_roberta --config configs/v4_pipeline_roberta.yaml
python -m v4.cli train --config configs/v4_pipeline_roberta.yaml --seed 42 --out-dir outputs/v4/roberta/seed_42
```

---

## 7. Smoke test

Add a one-shot import test in `tests/smoke/test_imports.py`:

```python
def test_imports_roberta():
    from v4.models.encoders.roberta_text import RoBERTaTextEncoder
    assert RoBERTaTextEncoder.name == "roberta_large"
```

Run `make test` and `make lint` before opening a PR.

---

## 8. Server wiring

Update `app/server_config.yaml`:

```yaml
encoders:
  text: { type: roberta_large, model_name: roberta-large, hidden_size: 1024, layer_index: -1, load_backbone: true }
```

The server already calls `enc.encode_text(text)` for the text path. As long as `encode_text` is implemented, no `app/server.py` changes are needed.
