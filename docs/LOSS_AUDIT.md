# Loss function audit (Stage A6)

> Required by `multiguard_non_image_pipeline_fix_plan.md` section 6 "Loss function and auxiliary image loss".

## V3.1 spec (section 5.5)

| Component | PDF value |
|---|---|
| Main loss | `CrossEntropyLoss` over 5 classes |
| Auxiliary loss | `BCE` for binary "image is AI/tampered" |
| Total loss | `main_loss + 0.1 * aux_loss` |
| Aux gradient isolation | Aux head uses `v_imgfor.detach()` so it does not destabilize other branches |
| Binary aux labels | classes 0/1/3 -> 0 (real image), classes 2/4 -> 1 (fake image) |
| Aux head at inference | Disabled (softmax on main_logits only) |

## What's actually in the repo

### A. Composition: CE(main) + 0.1 * BCE(aux)

`phases/v4/src/v4/training/trainer.py:59-67`:

```python
if loss_terms is None:
    loss_terms = [
        {"type": "ce", "target": "main_logits", "weight": 1.0},
        {"type": "bce", "target": "aux_logits",
         "weight": float(train_cfg.get("aux_weight", 0.1))},
    ]
```

- Term 1: `CrossEntropyLoss(main_logits, labels)` weight = `1.0`
- Term 2: `BCEWithLogitsLoss(aux_logits, aux_labels_oh)` weight = `0.1` (default; configurable via `train.aux_weight`)

`phases/v4/src/v4/training/losses.py:51-65`:

```python
for spec in self.specs:
    logits = out[spec.target]
    if spec.type == "ce":
        term = self.ce(logits, labels)
    elif spec.type in {"bce", "bce_logits"}:
        ...
        target_oh = F.one_hot(aux_labels, num_classes=logits.shape[-1]).float()
        term = self.bce(logits, target_oh)
    ...
    total = total + spec.weight * term
```

Both terms summed -> `total = ce + 0.1 * bce`. **Matches spec.**

### B. Binary aux-label mapping

`phases/v4/src/v4/core/class_map.py:32-39`:

```python
# V3.1 section 5.5 binary forensic consistency: "1 iff image-side is AI/tampered"
# Class 2 = Manipulated (image tampered) and Class 4 = Double-Fake (AI image)
IMAGE_FAKE_CLASSES: frozenset[int] = frozenset({2, 4})

def binary_image_label(label: int) -> int:
    """Map 5-class label -> binary 'image is AI/tampered' target for aux head."""
    return 1 if int(label) in IMAGE_FAKE_CLASSES else 0
```

| 5-class label | Class name | Image-side | Aux label |
|---|---|---|---|
| 0 | Real | real | 0 |
| 1 | Out-of-Context | real | 0 |
| 2 | Manipulated | fake | **1** |
| 3 | AI-Text | real | 0 |
| 4 | Fully-Fabricated | fake | **1** |

**Matches spec.**

### C. Where aux_label is attached to each batch

`phases/v4/src/v4/data/datasets/cached.py:9,41`:

```python
from v4.core.class_map import binary_image_label
...
"aux_label": torch.tensor(binary_image_label(int(row["label"])), dtype=torch.long),
```

Each cached-feature row gets a fresh `aux_label` derived from the 5-class label. **Matches spec.**

### D. Aux gradient isolation via `v_imgfor.detach()`

`phases/v4/src/v4/models/fusion/v3_pairwise.py:170-179`:

```python
fused = self.fusion(v_semantic, v_imgfor, v_textfor)
main_logits = self.classifier(fused)
# V3.1 section 5.5: detach v_imgfor to isolate aux loss gradients
aux_logits = self.aux_classifier(v_imgfor.detach())

return {
    "main_logits": main_logits,
    "aux_logits": aux_logits,
    "fused": fused,
}
```

- Aux head: `nn.Linear(feat_dim=768, 2)` initialized in `__init__` line 156.
- Input is `v_imgfor.detach()` -> aux gradients do NOT flow back into image-forensic encoder.
- Note: in our deployment v_imgfor is already detached (cached features, no encoder upstream), so the `.detach()` is defensive belt-and-braces. If a future student wires the image encoder live, the contract still holds.

**Matches spec.**

### E. Aux head NOT used at inference

`phases/v4/src/v4/evaluation/evaluator.py:38-43`:

```python
out = self.model(batch_dev)
if not isinstance(out, dict):
    out = {"main_logits": out}
logits = out["main_logits"]
probs = torch.softmax(logits, dim=-1).cpu().numpy()
preds = logits.argmax(dim=-1).cpu().numpy()
```

Only `main_logits` is read at inference. `aux_logits` is never used for prediction. **Matches spec.**

Same contract holds in `phases/v4/app_hf/app.py` (the HF Space): only `main_logits` is consumed for the final probability output.

## One minor implementation note

The aux head outputs `[B, 2]` (2 logits), and BCE is applied against the one-hot encoding of `aux_labels`. This is functionally equivalent to two independent sigmoid heads on the same features, vs. the alternative `[B, 1]` sigmoid + BCEWithLogitsLoss. Both produce valid binary gradients; the `[B, 2]` parameterization has 1538 params (768*2 + 2) vs 769 for `[B, 1]`. Not a deviation -- the spec says "BCE for binary image consistency" without specifying the head's output dim -- but worth flagging that the V4 trainer slightly over-parameterizes here. No effect on correctness of the auxiliary signal.

## Verdict

The composite loss is implemented exactly as V3.1 section 5.5 specifies:

1. Main: `CrossEntropyLoss` on `main_logits` weight 1.0
2. Aux: `BCEWithLogitsLoss` on `aux_logits` weight 0.1 (with `v_imgfor.detach()`)
3. Binary aux labels: classes 0/1/3 -> 0, classes 2/4 -> 1 (centralized in `binary_image_label`)
4. Aux head is **never** consulted at inference
5. Aux weight is configurable via `train.aux_weight` YAML key (defaults to 0.1 if absent)

No deviations from spec. No silent fallbacks. Section 6 of the non-image fix plan checklist is satisfied.

## Provenance

- `phases/v4/src/v4/training/losses.py` (74 lines)
- `phases/v4/src/v4/training/trainer.py` lines 59-67
- `phases/v4/src/v4/core/class_map.py` lines 32-39
- `phases/v4/src/v4/data/datasets/cached.py` lines 9, 41
- `phases/v4/src/v4/models/fusion/v3_pairwise.py` lines 156, 170-179
- `phases/v4/src/v4/evaluation/evaluator.py` lines 38-43
- Spec source: `docs/doctor-briefs/Implementation Guidelines V3_1.pdf` section 5.5
