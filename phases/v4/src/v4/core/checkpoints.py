"""Checkpoint save/load helpers with full provenance metadata.

Every saved checkpoint carries:
    config_hash, data_hash, git_sha, seed, torch_version, cuda_version,
    spec_version, stage, epoch, val_metrics
so future students can reproduce or audit any run via
`python -m v4 reproduce --run <path>`.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from v4.core.logging import get_logger

log = get_logger(__name__)


def file_sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 of a file (for data_hash provenance)."""
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git_sha(repo_root: Path | None = None) -> str:
    """Current git HEAD SHA, or 'no-git' if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def build_provenance(*, config_hash: str, data_hash: str, seed: int, stage: str) -> dict:
    """Build the provenance dict that goes into every checkpoint."""
    import torch

    return {
        "config_hash": config_hash,
        "data_hash": data_hash,
        "git_sha": git_sha(),
        "seed": int(seed),
        "stage": stage,
        "spec_version": "V3.1",
        "torch_version": torch.__version__,
        "cuda_version": (torch.version.cuda or "cpu") if torch.cuda.is_available() else "cpu",
    }


def save_checkpoint(
    path: Path | str,
    *,
    model_state: dict,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    rng_state: dict | None = None,
    epoch: int = 0,
    val_metrics: dict | None = None,
    config: dict | None = None,
    provenance: dict | None = None,
    extras: dict | None = None,
) -> None:
    """Save a checkpoint with all provenance fields.

    Two roles:
      - `best.pt` — written on every val-F1 improvement, no optimizer state needed
      - `latest.pt` — written every epoch, includes optimizer + scheduler + RNG for resume
    """
    import torch

    payload: dict[str, Any] = {
        "model_state": model_state,
        "epoch": int(epoch),
        "val_metrics": dict(val_metrics or {}),
        "config": dict(config or {}),
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state"] = scheduler_state
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if provenance is not None:
        payload.update(provenance)
    if extras:
        payload.update(extras)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(path: Path | str, *, map_location: str | Any = "cpu") -> dict:
    """Load a checkpoint payload dict (model_state + provenance + extras)."""
    import torch

    return torch.load(str(path), map_location=map_location, weights_only=False)


def strip_prefix(state: dict, prefix: str) -> dict:
    """Strip a prefix from every key in a state_dict (e.g. 'encoder.' for UnivFD)."""
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def shape_compat_filter(source_state: dict, target_state: dict) -> dict:
    """Return only the keys from source_state that match shape in target_state.

    Used by FND-CLIP load to skip incompatible classifier-head shapes when
    loading the V1 binary checkpoint into a fresh FND-CLIP architecture.
    """
    return {
        k: v
        for k, v in source_state.items()
        if k in target_state and target_state[k].shape == v.shape
    }
