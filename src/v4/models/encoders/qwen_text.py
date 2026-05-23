"""Qwen2-7B-Instruct text-forensic encoder per V3.1 §4 (Student 2).

Forced deviation F2 (plan §2): spec says "Layer 30, hidden 4096" which matches
Qwen1.5-7B. We use Qwen2-7B-Instruct (28 layers, hidden 3584) per V3-proven
choice + dual-4090 compatibility. The deviation is logged in
docs/SPEC_COMPLIANCE_MAP.md.

Pipeline per spec §4:
  1. Qwen2-7B-Instruct (AutoModelForCausalLM)
  2. AutoTokenizer
  3. output_hidden_states=True -> hidden_states[layer_index]
  4. Masked mean pool (MUST apply attention_mask before averaging)
  5. Linear(hidden, 768) + GELU projection

Input:  v_textfor_qwen [B, 3584]   (pooled hidden state, precomputed cache)
Output: v_textfor      [B, 768]    (projected forensic feature)

Note: For runtime inference (server use), we use the `runtime_forward` variant
that takes raw text -> tokenize -> Qwen forward -> masked-mean -> proj. For
training, we read precomputed [B, 3584] vectors from cache (much faster).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from v4.core.logging import get_logger
from v4.core.registry import ENCODER_REGISTRY, register
from v4.data.preprocessing.tokenizers import get_qwen_tokenizer, masked_mean_pool
from v4.models.encoders.base import EncoderBase

log = get_logger(__name__)


class TextForensicProjection(nn.Module):
    """Linear(hidden_size, 768) + GELU per V3.1 §4 step 5.

    Trainable in Stage 2 — only the projection moves; Qwen weights frozen.
    """

    def __init__(self, input_dim: int = 3584, output_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


@register(ENCODER_REGISTRY, "qwen2_7b")
class Qwen2TextEncoder(EncoderBase):
    """Qwen2-7B-Instruct text-forensic encoder + projection per V3.1 §4."""

    required_inputs = ("v_textfor_qwen",)   # cache key for precomputed Qwen hidden
    modality = "text"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        hidden_size: int = 3584,   # F2: 7B = 3584 (spec was 4096 for Qwen1.5-7B)
        layer_index: int = -1,      # F2: 7B has 28 layers; last is closest to spec's "layer 30"
        max_length: int = 512,
        output_dim: int = 768,
        load_backbone: bool = False,    # False during cached-feature training; True at inference
        gpu_mem_gib: int = 10,
        **_kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.layer_index = layer_index
        self.max_length = max_length
        self.output_dim = output_dim
        self.gpu_mem_gib = gpu_mem_gib

        # The only trainable piece (projection); Qwen weights are frozen
        self.proj = TextForensicProjection(input_dim=hidden_size, output_dim=output_dim)

        # Lazy: only loaded if explicitly requested (e.g. server)
        self._qwen = None
        self._tokenizer = None
        if load_backbone:
            self._load_backbone()

    def _load_backbone(self) -> None:
        """Load Qwen2-7B-Instruct weights into memory (heavy, ~15 GB)."""
        from transformers import AutoModelForCausalLM

        log.info("Loading Qwen2-7B-Instruct (fp16 + device_map=auto, GPU cap %d GiB)",
                 self.gpu_mem_gib)
        self._tokenizer = get_qwen_tokenizer(self.model_name)
        self._qwen = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: f"{self.gpu_mem_gib}GiB", "cpu": "20GiB"},
            low_cpu_mem_usage=True,
        )
        self._qwen.eval()
        for p in self._qwen.parameters():
            p.requires_grad = False
        log.info("Qwen loaded: hidden_size=%d, num_hidden_layers=%d",
                 self._qwen.config.hidden_size, self._qwen.config.num_hidden_layers)

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """Runtime: tokenize -> forward -> hidden_states[layer_index] -> masked mean pool.

        Returns: [B, hidden_size] float32 on the same device as the projection layer.
        """
        if self._qwen is None:
            self._load_backbone()
        if isinstance(text, str):
            text = [text]
        enc = self._tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=self.max_length,
        )
        # accelerate device_map='auto' accepts cuda:0 inputs and routes internally
        device = next(self._qwen.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self._qwen(**enc, output_hidden_states=True)
        h = out.hidden_states[self.layer_index]    # [B, S, H], fp16
        pooled = masked_mean_pool(h.float(), enc["attention_mask"])  # [B, H], fp32
        return pooled

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Project pre-cached Qwen hidden states to 768-d.

        During training, we read v_textfor_qwen [B, 3584] from cache.
        Projection is applied here.
        """
        v_textfor_raw = batch["v_textfor_qwen"]
        return self.proj(v_textfor_raw)

    def runtime_forward(self, text: str | list[str]) -> torch.Tensor:
        """Server-time path: raw text -> Qwen forward -> projection. [B, 768]."""
        pooled = self.encode_text(text)
        return self.proj(pooled.to(self.proj.linear.weight.device))
