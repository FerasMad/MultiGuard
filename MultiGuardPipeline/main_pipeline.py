"""main_pipeline.py — the ONLY orchestrator.

This is the single file allowed to import the four branch wrappers. It links
them together and manages the data flow. The four wrappers never import each
other; all wiring (including reading each encoder's ``out_dim`` and using those
to build the fusion model) happens here.

Run directly to see an end-to-end demo with dummy models::

    python main_pipeline.py
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

# The orchestrator (and ONLY the orchestrator) imports the four wrappers.
# Relative imports for package use; absolute fallback so the file also runs as
# a standalone script for the demo below.
try:
    from .semantic import SemanticEncoder
    from .forensic_text import ForensicTextEncoder
    from .image_forensic import ImageForensicEncoder
    from .fusion import FusionModule
except ImportError:  # running as a plain script
    from semantic import SemanticEncoder
    from forensic_text import ForensicTextEncoder
    from image_forensic import ImageForensicEncoder
    from fusion import FusionModule


class MainPipeline(nn.Module):
    """Links the four pluggable branches and runs the forward pass.

    Construct it either from ready-made wrappers (canonical) or, for
    convenience, from raw models via :meth:`from_models` (which builds the
    wrappers, reads the encoder ``out_dim`` values, and uses them to build the
    fusion model — so no dimension is ever hardcoded).

    Args:
        semantic / image / text: the three encoder wrappers.
        fusion:                  the fusion wrapper.
        feature_keys:            keys under which the three encoder outputs are
                                 placed in the feature dict handed to fusion.
    """

    def __init__(
        self,
        semantic: SemanticEncoder,
        image: ImageForensicEncoder,
        text: ForensicTextEncoder,
        fusion: FusionModule,
        feature_keys: tuple[str, str, str] = ("v_semantic", "v_imgfor", "v_textfor"),
    ) -> None:
        super().__init__()
        self.semantic = semantic
        self.image = image
        self.text = text
        self.fusion = fusion
        self.k_sem, self.k_img, self.k_txt = feature_keys

    @property
    def out_dim(self) -> int:
        """Final output width of the pipeline (delegates to the fusion)."""
        return self.fusion.out_dim

    @property
    def encoder_dims(self) -> dict[str, int]:
        """Per-branch output widths, read automatically from each wrapper."""
        return {
            self.k_sem: self.semantic.out_dim,
            self.k_img: self.image.out_dim,
            self.k_txt: self.text.out_dim,
        }

    def forward(self, batch: Any, text: Any | None = None) -> Any:
        """Run the three branches, then fuse.

        Each wrapper picks the keys it needs from ``batch``, so a single merged
        batch dict can drive all three. If ``text`` (raw text) is given, the
        text branch uses its runtime path instead of cached features.
        """
        features = {
            self.k_sem: self.semantic(batch),
            self.k_img: self.image(batch),
            self.k_txt: self.text.forward_text(text) if text is not None else self.text(batch),
        }
        return self.fusion(features)

    # ------------------------------------------------------------------ #
    # convenience factory: build wrappers + fusion from raw models        #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_models(
        cls,
        *,
        semantic_model: nn.Module,
        image_model: nn.Module,
        text_model: nn.Module,
        fusion_builder: Callable[[int, int, int], nn.Module],
        semantic_example: Any | None = None,
        image_example: Any | None = None,
        text_example: Any | None = None,
        feature_keys: tuple[str, str, str] = ("v_semantic", "v_imgfor", "v_textfor"),
        probe_batch: int = 2,
    ) -> "MainPipeline":
        """Build the whole pipeline from raw models.

        ``fusion_builder(d_sem, d_img, d_txt) -> nn.Module`` receives the three
        encoder output widths (inferred automatically) and returns the fusion
        model — this is how the fusion's input dims flow from the encoders
        without any hardcoded numbers.
        """
        semantic = SemanticEncoder(semantic_model, semantic_example)
        image = ImageForensicEncoder(image_model, image_example)
        text = ForensicTextEncoder(text_model, text_example)

        fusion_model = fusion_builder(semantic.out_dim, image.out_dim, text.out_dim)
        example_features = {
            feature_keys[0]: torch.zeros(probe_batch, semantic.out_dim),
            feature_keys[1]: torch.zeros(probe_batch, image.out_dim),
            feature_keys[2]: torch.zeros(probe_batch, text.out_dim),
        }
        fusion = FusionModule(fusion_model, example_features)
        return cls(semantic, image, text, fusion, feature_keys)


# ---------------------------------------------------------------------- #
# Real wiring: build a MainPipeline with the actual MultiGuard models.    #
# Matches the architecture diagram exactly:                              #
#   SemanticEncoder      <- FND-CLIP        (registry key "fnd_clip")     #
#   ForensicTextEncoder  <- Qwen2-7B        (registry key "qwen2_7b")     #
#   ImageForensicEncoder <- DCT ResNet50    (registry key "dct_forensic_v1")
#   FusionModule         <- V3PairwiseFusion(registry key "v3_pairwise")  #
# ---------------------------------------------------------------------- #
def build_multiguard_pipeline(
    config_path: str | None = None,
    device: str = "cpu",
    fusion_ckpt: str | None = None,
) -> "MainPipeline":
    """Build a ready ``MainPipeline`` wired with the REAL MultiGuard models.

    The v4 registry is imported lazily (inside this function) so the rest of
    this file stays standalone/pluggable. Mirrors ``app/server.py:_load_server``.

    Args:
        config_path: optional YAML with ``encoders`` + ``fusion`` specs
                     (same schema as ``app/server_config.yaml``). If omitted,
                     the defaults below match the deployed architecture.
        device:      device to place the assembled pipeline on.
        fusion_ckpt: optional path to a trained fusion checkpoint.
    """
    import os
    import sys

    # make the v4 package importable without installing it
    here = os.path.dirname(os.path.abspath(__file__))
    v4_src = os.path.normpath(os.path.join(here, "..", "phases", "v4", "src"))
    if v4_src not in sys.path:
        sys.path.insert(0, v4_src)

    from v4.core.registry import build_encoder, build_fusion, import_all

    import_all()

    # default specs == the architecture diagram (DCT ResNet50 image branch)
    enc_specs = {
        "semantic": {"type": "fnd_clip", "feat_dim": 512, "output_dim": 768},
        "image": {"type": "dct_forensic_v1", "out_dim": 768},
        "text": {"type": "qwen2_7b", "hidden_size": 3584, "output_dim": 768},
    }
    fusion_spec: dict = {"type": "v3_pairwise", "feat_dim": 768, "fused_dim": 1024,
                         "num_classes": 5}

    if config_path is not None:
        import yaml

        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        enc_specs = cfg["encoders"]
        fusion_spec = cfg["fusion"]

    # Layer 3 — build the real backbones OUTSIDE the wrappers
    semantic_model = build_encoder(dict(enc_specs["semantic"]))
    image_model = build_encoder(dict(enc_specs["image"]))
    text_model = build_encoder(dict(enc_specs["text"]))

    # Layer 2 — inject them into the pluggable wrappers (out_dim auto-read)
    semantic = SemanticEncoder(semantic_model)
    image = ImageForensicEncoder(image_model)
    text = ForensicTextEncoder(text_model)

    # fusion input width flows from the encoders — no hardcoded dim
    fspec = dict(fusion_spec)
    fspec.setdefault("feat_dim", semantic.out_dim)
    fspec.pop("ckpt", None)
    fusion_model = build_fusion(fspec)
    if fusion_ckpt and os.path.exists(fusion_ckpt):
        from v4.core.checkpoints import load_checkpoint

        ck = load_checkpoint(fusion_ckpt, map_location=device)
        fusion_model.load_state_dict(ck["model_state"])
    fusion = FusionModule(fusion_model)

    # Layer 1 — the orchestrator that wires everything together
    return MainPipeline(semantic, image, text, fusion).to(device)


# ---------------------------------------------------------------------- #
# Demo: dummy models prove the wiring without any real backbones.         #
# ---------------------------------------------------------------------- #
def _demo() -> None:
    class _DummyEncoder(nn.Module):
        """Reads one key from the batch dict and projects it — no declared dim."""

        def __init__(self, key: str, in_features: int, out_features: int) -> None:
            super().__init__()
            self.key = key
            self.proj = nn.Linear(in_features, out_features)

        def forward(self, batch: dict) -> torch.Tensor:
            return self.proj(batch[self.key])

    class _DummyTextEncoder(_DummyEncoder):
        def runtime_forward(self, text: list[str]) -> torch.Tensor:  # raw-text path
            x = torch.zeros(len(text), self.proj.in_features)
            return self.proj(x)

    class _DummyFusion(nn.Module):
        """Concatenates the three branch features and classifies — dims injected."""

        def __init__(self, d_sem: int, d_img: int, d_txt: int, num_classes: int = 5) -> None:
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(d_sem + d_img + d_txt, 64),
                nn.BatchNorm1d(64),  # needs batch >= 2 at probe time
                nn.GELU(),
                nn.Linear(64, num_classes),
            )
            self.aux = nn.Linear(d_img, 2)

        def forward(self, features: dict) -> dict:
            fused = torch.cat(
                [features["v_semantic"], features["v_imgfor"], features["v_textfor"]], dim=-1
            )
            return {"main_logits": self.head(fused), "aux_logits": self.aux(features["v_imgfor"])}

    # --- build raw models OUTSIDE, with arbitrary (different) widths ---
    sem_model = _DummyEncoder("x_sem", in_features=16, out_features=768)
    img_model = _DummyEncoder("v_imgfor_dct", in_features=8, out_features=512)
    txt_model = _DummyTextEncoder("x_txt", in_features=32, out_features=256)

    # example inputs let each wrapper probe its out_dim automatically
    pipeline = MainPipeline.from_models(
        semantic_model=sem_model,
        image_model=img_model,
        text_model=txt_model,
        fusion_builder=lambda d_sem, d_img, d_txt: _DummyFusion(d_sem, d_img, d_txt),
        semantic_example={"x_sem": torch.zeros(2, 16)},
        image_example={"v_imgfor_dct": torch.zeros(2, 8)},
        text_example={"x_txt": torch.zeros(2, 32)},
    )

    print("encoder_dims (auto-inferred):", pipeline.encoder_dims)
    print("pipeline.out_dim (auto-inferred):", pipeline.out_dim)

    batch = {
        "x_sem": torch.randn(4, 16),
        "v_imgfor_dct": torch.randn(4, 8),
        "x_txt": torch.randn(4, 32),
    }
    pipeline.eval()
    out = pipeline(batch)
    print("main_logits shape:", tuple(out["main_logits"].shape))
    print("aux_logits shape: ", tuple(out["aux_logits"].shape))

    # runtime text path
    out_rt = pipeline(batch, text=["hello", "world", "foo", "bar"])
    print("runtime main_logits shape:", tuple(out_rt["main_logits"].shape))


if __name__ == "__main__":
    _demo()
