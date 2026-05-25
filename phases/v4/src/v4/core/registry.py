"""Registry pattern for plug-and-play swap of encoders, fusion modules, datasets.

Plain module-level dicts + `@register` decorator. No plugin discovery, no
entry-point scanning. To add a new encoder/fusion/dataset, write one file under
the right subpackage and add a `@register(REGISTRY, "name")` decorator.

See docs/ADD_ENCODER.md for the contributor walkthrough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from v4.models.encoders.base import EncoderBase
    from v4.models.fusion.base import FusionBase

ENCODER_REGISTRY: dict[str, type] = {}
FUSION_REGISTRY: dict[str, type] = {}
DATASET_REGISTRY: dict[str, type] = {}


def register(registry: dict[str, type], name: str):
    """Decorator that adds a class to a registry under the given name.

    Usage::

        @register(ENCODER_REGISTRY, "univfd")
        class UnivFDEncoder(EncoderBase):
            ...
    """

    def deco(cls: type) -> type:
        if name in registry:
            raise ValueError(f"Registry already contains key {name!r}: {registry[name]}")
        registry[name] = cls
        # Attach the registered name for introspection.
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return deco


def build_encoder(spec: dict) -> EncoderBase:
    """Instantiate an encoder from a YAML spec dict.

    The spec must contain a `type` key matching an ENCODER_REGISTRY entry.
    All other keys are passed as kwargs to the class constructor.
    """
    if "type" not in spec:
        raise ValueError(f"encoder spec missing 'type' key: {spec!r}")
    cls = ENCODER_REGISTRY.get(spec["type"])
    if cls is None:
        avail = ", ".join(sorted(ENCODER_REGISTRY)) or "<empty>"
        raise KeyError(f"encoder type {spec['type']!r} not registered. Available: {avail}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)


def build_fusion(spec: dict) -> FusionBase:
    """Instantiate a fusion module from a YAML spec dict."""
    if "type" not in spec:
        raise ValueError(f"fusion spec missing 'type' key: {spec!r}")
    cls = FUSION_REGISTRY.get(spec["type"])
    if cls is None:
        avail = ", ".join(sorted(FUSION_REGISTRY)) or "<empty>"
        raise KeyError(f"fusion type {spec['type']!r} not registered. Available: {avail}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)


def build_dataset(spec: dict) -> Dataset:
    """Instantiate a dataset from a YAML spec dict."""
    if "type" not in spec:
        raise ValueError(f"dataset spec missing 'type' key: {spec!r}")
    cls = DATASET_REGISTRY.get(spec["type"])
    if cls is None:
        avail = ", ".join(sorted(DATASET_REGISTRY)) or "<empty>"
        raise KeyError(f"dataset type {spec['type']!r} not registered. Available: {avail}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)


def import_all() -> None:
    """Force-import every encoder/fusion/dataset module so their @register
    decorators run. Call this once near the start of any entrypoint that
    needs the registries populated.
    """
    # Encoders
    # Datasets
    import v4.data.datasets.cached
    import v4.data.datasets.runtime
    import v4.models.encoders.dct_forensic
    import v4.models.encoders.fnd_clip
    import v4.models.encoders.qwen_text
    import v4.models.encoders.univfd

    # Fusion
    import v4.models.fusion.v3_pairwise  # noqa: F401
