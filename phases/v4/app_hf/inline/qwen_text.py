"""Qwen2-7B-Instruct text-forensic encoder — inline copy for the HF Space.

Mirrors phases/v4/src/v4/models/encoders/qwen_text.py with v4.* imports
stripped and the @register decorator dropped.

The TextForensicProjection (3584 -> 768) lives INSIDE V3PairwiseFusion via
proj_dims (P5.10a). This module's `encode_text` returns raw 3584-d.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch
from torch import nn

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_qwen_tokenizer(model_name: str = "Qwen/Qwen2-7B-Instruct"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class Qwen2TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        hidden_size: int = 3584,
        layer_index: int = -1,
        max_length: int = 512,
        load_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.layer_index = layer_index
        self.max_length = max_length
        self.output_dim = hidden_size
        self._qwen = None
        self._tokenizer = None
        if load_backbone:
            self._load_backbone()

    def _load_backbone(self) -> None:
        from transformers import AutoModelForCausalLM

        log.info("Loading %s (fp16)", self.model_name)
        self._tokenizer = get_qwen_tokenizer(self.model_name)
        self._qwen = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._qwen.eval()
        for p in self._qwen.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        if self._qwen is None:
            self._load_backbone()
        if isinstance(text, str):
            text = [text]
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        device = next(self._qwen.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self._qwen(**enc, output_hidden_states=True)
        h = out.hidden_states[self.layer_index]
        pooled = masked_mean_pool(h.float(), enc["attention_mask"])
        return pooled  # [B, 3584]
