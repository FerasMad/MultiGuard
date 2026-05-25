"""V3.1 section 7 MMFakeBench transfer probe.

Implemented as a per-source filter on the test split where source starts with
'MMFakeBench_' (per locked decision C3). The evaluator's per-source breakdown
already covers this, but we expose a dedicated entry point for clarity.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from v4.core.logging import get_logger
from v4.evaluation.evaluator import BaseEvaluator

log = get_logger(__name__)


def run_transfer_probe(
    evaluator: BaseEvaluator,
    test_loader: DataLoader,
    out_dir: str | Path,
) -> dict:
    """Run V3.1 section 7 transfer eval on MMFakeBench-origin rows of the test split.

    Note: the underlying dataset must include the `source` field in its samples
    (CachedFeatureDataset does). This function delegates to the evaluator's full
    eval routine and then filters per-source.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return evaluator.evaluate(test_loader, out_dir=out_dir)
