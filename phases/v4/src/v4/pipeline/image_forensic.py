"""image_forensic.py — pluggable wrapper for the IMAGE-FORENSIC branch.

One class. The underlying model is INJECTED from the outside (never built
here), and ``out_dim`` is inferred AUTOMATICALLY so no dimension is hardcoded.

Decoupling rule: this file must NOT import any of the other branch files
(semantic, forensic_text, fusion). Only ``main_pipeline`` wires them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ImageForensicEncoder(nn.Module):
    """Wraps any externally-built image-forensic model.

    Args:
        model:         an ``nn.Module`` built *outside* this class returning a
                       ``Tensor`` of shape ``[B, out_dim]``.
        example_input: optional sample input used once to probe ``out_dim``
                       (source of truth). If omitted, falls back to a dimension
                       attribute exposed by the model.

    Swapping the image model = construct a different ``model`` and pass it here;
    nothing else in the pipeline changes.
    """

    _DIM_ATTRS = ("out_dim", "output_dim", "embed_dim", "hidden_size")

    def __init__(self, model: nn.Module, example_input: Any | None = None) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(
                "ImageForensicEncoder requires an nn.Module instance injected "
                "from outside; do not build the model inside this class."
            )
        self.model = model
        self._out_dim = self._infer_out_dim(example_input)

    @property
    def out_dim(self) -> int:
        """Output width of the image-forensic vector — inferred, never hardcoded."""
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
            "Cannot infer out_dim for ImageForensicEncoder: pass example_input=... "
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
                "Image model must return a Tensor of shape [B, out_dim]; "
                f"got {type(out).__name__}."
            )
        return out

    @staticmethod
    def _to_device(obj: Any, device: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: ImageForensicEncoder._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(ImageForensicEncoder._to_device(v, device) for v in obj)
        return obj
