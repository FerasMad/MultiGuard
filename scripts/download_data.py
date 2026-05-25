"""Auto-download MMFakeBench from HuggingFace.

NewsCLIPpings, DGM4, GenImage, VisualNews are handled via TeamViewer or manual.
"""

from __future__ import annotations

import sys
from pathlib import Path


def download_mmfakebench(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"downloading liuxuannan/MMFakeBench -> {dest}")
    snapshot_download(
        repo_id="liuxuannan/MMFakeBench",
        repo_type="dataset",
        local_dir=str(dest),
    )


if __name__ == "__main__":
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/MMFakeBench")
    download_mmfakebench(dest)
    print(f"OK: {dest}")
