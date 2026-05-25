# Adding a new fusion module

V4 ships a single fusion implementation, `V3PairwiseFusion`, that matches V3.1 section 5 exactly. If a future contributor wants to compare against a different fusion strategy (concatenation baseline, gated cross-attention, etc.), follow this recipe.

---

## 1. The contract

Every fusion subclasses `FusionBase` in `src/v4/models/fusion/base.py`:

```python
class FusionBase(nn.Module):
    name: ClassVar[str]                          # registry key
    expected_inputs: ClassVar[tuple[str, ...]]   # feature names it consumes
    num_classes: int

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Args:
            features: dict mapping feature name -> [B, D] tensor

        Returns:
            {
                "main_logits": [B, num_classes],
                "aux_logits":  [B, 2],
                "fused":       [B, fused_dim],
            }
        """
        ...
```

The trainer + evaluator only depend on this dict shape - keys `main_logits` and `aux_logits` are mandatory; `fused` is informational for ablation.

---

## 2. Create the file

`src/v4/models/fusion/concat_baseline.py`:

```python
"""Concatenation baseline fusion.

Concat v_semantic + v_imgfor + v_textfor along the feature dim,
then a 3-layer MLP -> main_logits. Useful as a sanity baseline.
"""
from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

from v4.core.registry import register, FUSION_REGISTRY
from v4.models.fusion.base import FusionBase


@register(FUSION_REGISTRY, "concat_baseline")
class ConcatBaselineFusion(FusionBase):
    expected_inputs: ClassVar[tuple[str, ...]] = (
        "v_semantic", "v_imgfor", "v_textfor",
    )

    def __init__(
        self,
        feat_dim: int = 768,
        fused_dim: int = 1024,
        num_classes: int = 5,
        dropout: float = 0.3,
        projections: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.projs = nn.ModuleDict()
        if projections:
            for k, spec in projections.items():
                self.projs[k] = nn.Linear(spec["in_dim"], spec["out_dim"])

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * len(self.expected_inputs), fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
        )

        self.head = nn.Linear(fused_dim, num_classes)
        self.aux_head = nn.Linear(feat_dim, 2)   # operates on v_imgfor only

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Apply optional dim projections.
        proj_features = {}
        for k in self.expected_inputs:
            x = features[k]
            if k in self.projs:
                x = self.projs[k](x)
            proj_features[k] = x

        cat = torch.cat([proj_features[k] for k in self.expected_inputs], dim=-1)
        fused = self.mlp(cat)
        main_logits = self.head(fused)
        aux_logits = self.aux_head(proj_features["v_imgfor"].detach())

        return {
            "main_logits": main_logits,
            "aux_logits":  aux_logits,
            "fused":       fused,
        }
```

---

## 3. Register the import

`src/v4/models/fusion/__init__.py`:

```python
from v4.models.fusion import (
    concat_baseline,   # <-- add
    v3_pairwise,
)

__all__ = ["concat_baseline", "v3_pairwise"]
```

---

## 4. Reference it in YAML

`configs/v4_pipeline_concat.yaml`:

```yaml
fusion:
  type: concat_baseline
  feat_dim: 768
  fused_dim: 1024
  num_classes: 5
  dropout: 0.3
  projections:
    v_semantic: { in_dim: 512,  out_dim: 768 }
    v_textfor:  { in_dim: 3584, out_dim: 768 }
```

Train it:
```bash
python -m v4.cli train --config configs/v4_pipeline_concat.yaml --seed 42 --out-dir outputs/v4/concat_baseline
```

---

## 5. Spec-compliance considerations

If you're swapping the fusion to compare against V3.1 section 5, **do not** change the classifier head. `src/v4/models/classifier/mlp_head.py` is locked to V3.1 section 6 (`1024 -> 512+BN+GELU+Dropout(0.5) -> 256+GELU -> 5`).

Your new fusion's output `fused` tensor MUST be 1024-dim if the classifier head is going to consume it (the trainer wires `fused -> MLPClassifier`). If you want a different fused dim, you must also swap the classifier - see the relevant V3.1 section 6 caveat in [`DECISIONS.md`](DECISIONS.md).

---

## 6. Test

`tests/spec_compliance/test_v3_1_section_5_3_pairwise_sum.py` only tests `V3PairwiseFusion` - your new fusion is exempt from V3.1 section 5 assertions (since by construction it's a different fusion).

Add a shape test in `tests/unit/test_concat_fusion.py`:

```python
import torch
from v4.models.fusion.concat_baseline import ConcatBaselineFusion

def test_concat_baseline_shapes():
    fusion = ConcatBaselineFusion(feat_dim=768, num_classes=5)
    feats = {
        "v_semantic": torch.randn(4, 768),
        "v_imgfor":   torch.randn(4, 768),
        "v_textfor":  torch.randn(4, 768),
    }
    out = fusion(feats)
    assert out["main_logits"].shape == (4, 5)
    assert out["aux_logits"].shape == (4, 2)
    assert out["fused"].shape == (4, 1024)
```

---

## 7. Recording the design decision

Add an entry in [`DECISIONS.md`](DECISIONS.md) under "Alternate fusions explored" describing:
- Motivation (e.g. "baseline for V3 pairwise ablation")
- Result (F1-macro, AUC, transfer numbers)
- Whether it ships in `main` or stays in a feature branch
