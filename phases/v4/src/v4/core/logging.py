"""Logging configuration for V4. Replaces V3's `print()` everywhere.

Usage::

    from v4.core.logging import get_logger
    log = get_logger(__name__)
    log.info("Stage 1 training started")
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure(level: LogLevel = "INFO", format_: str | None = None) -> None:
    """One-time root-logger configuration. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(format_ or _DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    # Quiet down chatty libraries
    for noisy in ("urllib3", "filelock", "huggingface_hub", "PIL"):
        logging.getLogger(noisy).setLevel("WARNING")
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger getter. Call once at top of each file."""
    configure()
    return logging.getLogger(name)
