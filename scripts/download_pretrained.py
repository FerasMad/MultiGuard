"""Download CNNDetection blur_jpg_v0.pth (UnivFD initialization)."""

from __future__ import annotations

import sys
from pathlib import Path

URL = "https://github.com/PeterWang512/CNNDetection/releases/download/v1.0/blur_jpg_prob0.pth"


def download(dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {dest}")
    urllib.request.urlretrieve(URL, dest)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/pretrained/blur_jpg_prob0.pth")
    download(out)
    print(f"OK ({out.stat().st_size} bytes)")
