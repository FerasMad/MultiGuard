"""Reproducibility — seed Python, NumPy, PyTorch, CUDA all at once.

See plan §8 (Reproducibility) for the policy. Strict cudnn determinism is OPT-IN
(via the trainer config `train.deterministic: true`) because it costs ~10%
throughput. Default behavior: seed everything but allow non-deterministic CUDA
kernels for speed.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_all(seed: int, *, deterministic: bool = False) -> None:
    """Seed every relevant RNG.

    Args:
        seed: integer seed (any value; convention: 42, 1337, 2024).
        deterministic: if True, force `torch.backends.cudnn.deterministic=True`
            and `benchmark=False`. ~10% throughput cost. Opt-in for the final
            canonical run, off for sweeps.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch is heavy — import lazily so this module is importable without it.
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Standard: faster, but small run-to-run variance possible.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def snapshot_rng_state() -> dict:
    """Capture current RNG state across all generators. For checkpoint resume.

    Returns a dict that can be round-tripped through `restore_rng_state(...)`.
    """
    out: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch
        out["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            out["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return out


def restore_rng_state(state: dict) -> None:
    """Inverse of `snapshot_rng_state`. Used by trainer resume path."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except ImportError:
        pass
