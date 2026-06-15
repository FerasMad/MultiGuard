"""semantic.py — pluggable wrapper for the SEMANTIC branch.

One class. The underlying model is INJECTED from the outside (never built
here), and ``out_dim`` is inferred AUTOMATICALLY from that model so no
dimension is ever hardcoded.

Decoupling rule: this file must NOT import any of the other branch files
(forensic_text, image_forensic, fusion). Only ``main_pipeline`` wires them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SemanticEncoder(nn.Module):
    """Wraps any externally-built semantic model.

    Args:
        model:         an ``nn.Module`` built *outside* this class. Its
                       ``forward`` is expected to return a ``Tensor`` of shape
                       ``[B, out_dim]``.
        example_input: an optional sample input (the exact object the model's
                       ``forward`` consumes — e.g. a dict of tensors). When
                       given, ``out_dim`` is read from a probe forward (the
                       source of truth). When omitted, ``out_dim`` falls back to
                       a dimension attribute already exposed by the model.

    Swapping the semantic model = construct a different ``model`` and pass it
    here; nothing else in the pipeline changes.
    """

    # attribute names a model may already expose that declare its output width
    _DIM_ATTRS = ("out_dim", "output_dim", "embed_dim", "hidden_size")

    def __init__(self, model: nn.Module, example_input: Any | None = None) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(
                "SemanticEncoder requires an nn.Module instance injected from "
                "outside; do not build the model inside this class."
            )
        self.model = model
        self._out_dim = self._infer_out_dim(example_input)

    @property
    def out_dim(self) -> int:
        """Output width of the semantic vector — inferred, never hardcoded."""
        return self._out_dim

    def forward(self, batch: Any) -> torch.Tensor:
        return self.model(batch)

    # ------------------------------------------------------------------ #
    # out_dim auto-inference                                              #
    # ------------------------------------------------------------------ #
    def _infer_out_dim(self, example_input: Any | None) -> int:
        if example_input is not None:
            return int(self._probe(example_input).shape[-1])
        for name in self._DIM_ATTRS:
            value = getattr(self.model, name, None)
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError(
            "Cannot infer out_dim for SemanticEncoder: pass example_input=... "
            f"or expose one of {self._DIM_ATTRS} on the injected model."
        )

    def _probe(self, example_input: Any) -> torch.Tensor:
        param = next(self.model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        example_input = self._to_device(example_input, device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(example_input)
        finally:
            self.model.train(was_training)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                "Semantic model must return a Tensor of shape [B, out_dim]; "
                f"got {type(out).__name__}."
            )
        return out

    @staticmethod
    def _to_device(obj: Any, device: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: SemanticEncoder._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(SemanticEncoder._to_device(v, device) for v in obj)
        return obj
