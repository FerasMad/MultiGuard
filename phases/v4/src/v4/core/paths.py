"""Centralized path resolution for the V4 project.

All paths are computed relative to PROJECT_ROOT (the directory containing
`pyproject.toml`). This avoids hardcoded absolute paths and makes the code
portable across PCs.
"""
from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (or this file's location) until pyproject.toml found."""
    p = Path(start) if start else Path(__file__).resolve()
    if p.is_file():
        p = p.parent
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume the caller knows what they're doing
    return p


PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE_ROOT = PROJECT_ROOT / "cache" / "v4"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "v4"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
APP_ROOT = PROJECT_ROOT / "app"
DOCS_ROOT = PROJECT_ROOT / "docs"


def ensure_dir(p: Path | str) -> Path:
    """Create directory (and parents) if missing; return Path."""
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def resolve_relative(p: Path | str) -> Path:
    """Resolve a path that may be relative to PROJECT_ROOT."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()


def get_hf_cache() -> Path:
    """Location of HuggingFace's cache (for Qwen2-7B, BERT, CLIP weights)."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
