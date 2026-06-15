"""fusion.py — pluggable wrapper for the FUSION stage.

One class. The fusion model is INJECTED from the outside (never built here),
and ``out_dim`` is inferred AUTOMATICALLY so no dimension is hardcoded.

The fusion model consumes a feature dict (one entry per encoder branch) and may
return either a bare ``Tensor`` or a dict (e.g. ``{"main_logits", "aux_logits",
"fused"}``). For a dict, ``out_dim`` is taken from the entry named ``out_key``.

Decoupling rule: this file must NOT import any of the other branch files
(semantic, forensic_text, image_forensic). Only ``main_pipeline`` wires them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FusionModule(nn.Module):
    """Wraps any externally-built fusion model.

    Args:
        model:            an ``nn.Module`` built *outside* this class (the
                          orchestrator builds it using the encoders' ``out_dim``
                          so its input widths are never hardcoded either).
        example_features: optional sample feature dict used once to probe
                          ``out_dim`` (source of truth). Build it with batch
                          size >= 2 so layers like ``BatchNorm1d`` survive the
                          probe. If omitted, falls back to a dimension attribute.
        out_key:          which output entry defines ``out_dim`` when the model
                          returns a dict (default ``"main_logits"``).

    Swapping the fusion model = construct a different ``model`` and pass it here.
    """

    _DIM_ATTRS = ("out_dim", "output_dim", "num_classes")

    def __init__(
        self,
        model: nn.Module,
        example_features: Any | None = None,
        out_key: str = "main_logits",
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(
                "FusionModule requires an nn.Module instance injected from "
                "outside; do not build the model inside this class."
            )
        self.model = model
        self.out_key = out_key
        self._out_dim = self._infer_out_dim(example_features)

    @property
    def out_dim(self) -> int:
        """Width of the fusion's primary output — inferred, never hardcoded."""
        return self._out_dim

    def forward(self, features: Any) -> Any:
        return self.model(features)

    # ------------------------------------------------------------------ #
    # out_dim auto-inference                                              #
    # ------------------------------------------------------------------ #
    def _infer_out_dim(self, example_features: Any | None) -> int:
        if example_features is not None:
            return int(self._select(self._probe(example_features)).shape[-1])
        for name in self._DIM_ATTRS:
            value = getattr(self.model, name, None)
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError(
            "Cannot infer out_dim for FusionModule: pass example_features=... "
            f"or expose one of {self._DIM_ATTRS} on the injected model."
        )

    def _select(self, out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            if self.out_key in out and isinstance(out[self.out_key], torch.Tensor):
                return out[self.out_key]
            for value in out.values():
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError(
            "Fusion model output must be a Tensor or a dict containing at least "
            f"one Tensor (looked for key {self.out_key!r})."
        )

    def _probe(self, example_features: Any) -> Any:
        param = next(self.model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        example_features = self._to_device(example_features, device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(example_features)
        finally:
            self.model.train(was_training)
        return out

    @staticmethod
    def _to_device(obj: Any, device: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: FusionModule._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(FusionModule._to_device(v, device) for v in obj)
        return obj
