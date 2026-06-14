"""Pluggable MultiGuard pipeline (dependency-injection design).

Five decoupled units, one class each:
    - SemanticEncoder        (semantic.py)
    - ForensicTextEncoder    (forensic_text.py)
    - ImageForensicEncoder   (image_forensic.py)
    - FusionModule           (fusion.py)
    - MainPipeline           (main_pipeline.py)  <- the only orchestrator

Every model is injected from outside; every class exposes an auto-inferred
``out_dim``; the four branch files never import each other.

This module is package plumbing (re-exports for convenience), not a sixth
pipeline file.
"""

from __future__ import annotations

from .fusion import FusionModule
from .forensic_text import ForensicTextEncoder
from .image_forensic import ImageForensicEncoder
from .main_pipeline import MainPipeline
from .semantic import SemanticEncoder

__all__ = [
    "SemanticEncoder",
    "ForensicTextEncoder",
    "ImageForensicEncoder",
    "FusionModule",
    "MainPipeline",
]
