# Qwen2-7B text-branch audit (Stage A2)

> Required by `multiguard_non_image_pipeline_fix_plan.md` section 3 "Qwen/text branch layer and hidden-size deviation".

## V3.1 PDF spec (what the doctor's brief asks for)

| Param | PDF value |
|---|---|
| Model | Qwen2-7B-Instruct |
| Layer extracted | layer 30 |
| Hidden size | 4096 |
| Pooling | masked-mean pooling |
| Projection | `Linear(4096, 768) + GELU` |

## What's actually in the repo

Source: `phases/v4/app_hf/inline/qwen_text.py` (mirrored in `phases/v4/src/v4/models/encoders/qwen_text.py`).

| Param | Repo value | Reason |
|---|---|---|
| Model | `Qwen/Qwen2-7B-Instruct` | Matches spec |
| Layer extracted | **`hidden_states[-1]`** (last layer) | See note below |
| Hidden size | **3584** | Actual architecture of Qwen2-7B-Instruct |
| Pooling | masked-mean (`(h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)`) | Matches spec |
| Projection | `Linear(3584, 768) + GELU` (inside `V3PairwiseFusion.text_proj`) | Matches spec (different in-dim) |

## Why the deviations are forced by the model

### Hidden size 3584 != PDF's 4096

Qwen2-7B-Instruct's published config: `hidden_size: 3584`, `num_hidden_layers: 28`, `num_attention_heads: 28`. The PDF's "4096" matches an earlier `Qwen1.5-7B-Chat`, not `Qwen2-7B-Instruct`. Confirmable via:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
print(cfg.hidden_size)   # 3584
print(cfg.num_hidden_layers)  # 28
```

The PDF was written before Qwen2 released; the 4096 number is architecturally impossible for the exact model name the PDF specifies. Repo correctly uses 3584.

This deviation was already logged in the V4 plan as **F2** (forced deviation register). This audit consolidates it.

### Layer -1 != PDF's "layer 30"

Qwen2-7B-Instruct has 28 hidden layers (indices 0-27), plus `hidden_states[0]` which is the embedding output. So `hidden_states` is a list of length 29: indices 0 (embedding) through 28 (final hidden state).

PDF says "layer 30" -- impossible since only 28 transformer blocks exist. The repo uses `hidden_states[-1]` (the last entry, equivalent to `hidden_states[28]`).

In the original Qwen1.5-7B-Chat (which had 32 layers + embeddings = 33 entries), "layer 30" would be the third-to-last hidden state. There's no equivalent index for Qwen2-7B-Instruct. Using the last layer is the natural translation.

### Projection 4096->768 became 3584->768

V3PairwiseFusion's `text_proj` is initialized with `proj_dims["v_textfor"] = 3584` in `v4_pipeline_honest.yaml`, producing `Linear(3584, 768) + GELU`. The OUTPUT of the projection matches the spec (768-d common space). Only the INPUT dim differs.

## Pooling sanity check

```python
# from qwen_text.py:33-37
def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
```

- Mask is applied to hidden states BEFORE the sum -> padding tokens contribute 0.
- Division by `clamp(min=1.0)` prevents divide-by-zero for empty inputs (defensive, shouldn't happen with tokenizer's BOS+EOS).
- Output shape: `[B, hidden_size]` = `[B, 3584]`.

Matches the V3.1 spec's "masked mean pooling" literal wording.

## Tokenizer configuration

```python
# from qwen_text.py:21-30
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"
```

- `padding_side="left"` -- correct for causal-LM use (Qwen is decoder-only).
- `pad_token = eos_token` fallback -- standard for Qwen which doesn't ship with a dedicated pad token.
- Tokenizer is cached via `@lru_cache(maxsize=2)` -- instantiated once per session.

## Verdict

The Qwen text branch deviates from the PDF in **two** ways, both **forced by the model architecture** the PDF itself specifies:

1. Hidden size 3584 (not 4096) -- Qwen2-7B-Instruct has `hidden_size=3584`. Documented.
2. Layer -1 (not 30) -- Qwen2-7B-Instruct only has 28 transformer blocks. Using the last layer is the natural translation.

Neither deviation is a bug. Both are unavoidable given the model the PDF asked for. The masked-mean pooling, projection to 768-d common dim, attention-mask handling, and left-padding are all spec-compliant.

## Cross-reference: per-class F1 contribution

The Qwen text branch was the main mover for classes 3 and 4 (AI-Text and Fully-Fabricated). After the BLIP-2 honest-path caption rewrite removed the syntactic-fingerprint shortcut (P9.3), per-class test F1 dropped from `0.997 / 0.998` to `0.989 / 0.990`. The branch ablation (P14.B4) will quantify what's left of this contribution.

## Provenance

- Code: `phases/v4/app_hf/inline/qwen_text.py` (94 lines)
- Spec source: `docs/doctor-briefs/Implementation Guidelines V3_1.pdf` section 4
- Deviation register: V4 plan D-register entry F2
