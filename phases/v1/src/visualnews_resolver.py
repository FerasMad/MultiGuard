"""Resolve NewsCLIPpings-style image_ids into VisualNews on-disk paths.

NewsCLIPpings annotations only carry an integer `image_id`; they don't
contain image paths or even the source publisher. The on-disk layout
that VisualNews ships looks like:

    origin/<source>/images/<group>/<num>.jpg

where `<source>` is one of `{bbc, guardian, washington_post, usa_today}`,
`<group>` is a 4-digit zero-padded directory, and `<num>` is a 3-digit
zero-padded filename. The relationship between NewsCLIPpings' image_id
and that path goes through VisualNews' own metadata pickles in
`articles.tar.gz`:

    image_id (NewsCLIPpings)  -- pickle lookup -->  (source, ori_id)
    (source, ori_id)                              <-->  image_path

where `ori_id = group * 1000 + num`. Verified against ~200 real files
from origin.tar: 171/202 round-tripped cleanly via this mapping.

Usage:

    from src.visualnews_resolver import VisualNewsResolver
    r = VisualNewsResolver(articles_path="data/raw/visualnews/articles.tar.gz")
    path = r.path_for(image_id=1452411)
    # => Path('origin/bbc/images/0102/969.jpg')

The resolver is stateless past initial loading; building the index
takes ~5 seconds and ~100 MB RAM for 1.5M entries. The index is
cached on disk after the first build (`articles.tar.gz.index.json`)
so subsequent calls are <1 second.
"""
from __future__ import annotations

import json
import pickle
import tarfile
from pathlib import Path
from typing import Iterable

# Maps the pickle-file basename (without 'processed_' prefix or '.p' suffix)
# to the on-disk source folder name. NewsCLIPpings rolled bbc into
# bbc_1 + bbc_2 because it was scraped in two batches; both share the
# `bbc/` image folder.
_PICKLE_TO_SOURCE = {
    "washington_post": "washington_post",
    "bbc_1": "bbc",
    "bbc_2": "bbc",
    "guardian": "guardian",
    "usa_today": "usa_today",
}


class VisualNewsResolver:
    """image_id -> on-disk JPG path lookup, backed by VisualNews metadata."""

    def __init__(
        self,
        articles_path: str | Path,
        origin_root: str | Path = "origin",
        cache_index: bool = True,
    ) -> None:
        """Build (or load cached) the image_id index.

        Args:
            articles_path: path to `articles.tar.gz` (the metadata bundle
                that ships with VisualNews).
            origin_root: prefix prepended to every resolved image path.
                Default `"origin"` matches the layout inside the
                `origin.tar` archive.
            cache_index: if True, write the index to
                `<articles_path>.index.json` on first build and reuse it
                on subsequent runs. Set to False for one-off use.
        """
        self.articles_path = Path(articles_path)
        self.origin_root = Path(origin_root)

        cache_path = self.articles_path.with_name(
            self.articles_path.name + ".index.json"
        )
        if cache_index and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                payload = json.load(f)
            # JSON keys are strings; convert back to int.
            self._index: dict[int, tuple[str, int]] = {
                int(k): (v[0], v[1]) for k, v in payload.items()
            }
        else:
            self._index = self._build_index_from_tarball()
            if cache_index:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {str(k): list(v) for k, v in self._index.items()},
                        f,
                    )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, image_id: int) -> bool:
        return image_id in self._index

    def path_for(self, image_id: int) -> Path | None:
        """Return the on-disk relative path for a NewsCLIPpings image_id.

        Returns:
            A `Path` like `origin/bbc/images/0102/969.jpg`, or `None`
            if the image_id isn't in the VisualNews index (the small
            fraction of NewsCLIPpings ids that don't survived a join
            against the released metadata pickles).
        """
        entry = self._index.get(image_id)
        if entry is None:
            return None
        source, ori_id = entry
        group = f"{ori_id // 1000:04d}"
        fname = f"{ori_id % 1000:03d}.jpg"
        return self.origin_root / source / "images" / group / fname

    def source_for(self, image_id: int) -> str | None:
        """Return the publisher source name for an image_id, or None."""
        entry = self._index.get(image_id)
        return entry[0] if entry else None

    def coverage(self, image_ids: Iterable[int]) -> tuple[int, int]:
        """Report how many of `image_ids` resolve to a path.

        Returns `(found, total)`. Useful for sanity-checking a NewsCLIPpings
        annotations file before you start building a balanced CSV.
        """
        ids = list(image_ids)
        found = sum(1 for i in ids if i in self._index)
        return found, len(ids)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_index_from_tarball(self) -> dict[int, tuple[str, int]]:
        """Open articles.tar.gz, load every processed_*.p pickle, build the
        image_id -> (source, ori_id) map. ~5 s + ~100 MB RAM.
        """
        if not self.articles_path.exists():
            raise FileNotFoundError(
                f"articles.tar.gz not found at {self.articles_path}"
            )
        index: dict[int, tuple[str, int]] = {}
        with tarfile.open(self.articles_path, "r:gz") as tf:
            for member in tf.getmembers():
                base = member.name.rsplit("/", 1)[-1]
                if not (base.startswith("processed_") and base.endswith(".p")):
                    continue
                tag = base[len("processed_"):-2]  # strip prefix + '.p'
                if tag not in _PICKLE_TO_SOURCE:
                    continue
                source = _PICKLE_TO_SOURCE[tag]
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = pickle.load(f)
                for image_id, entry in data.items():
                    ori_id = entry.get("ori_id")
                    if ori_id is None:
                        continue
                    index[int(image_id)] = (source, int(ori_id))
        return index


if __name__ == "__main__":
    # Smoke test: build the index, look up a known id, report coverage.
    import sys
    if len(sys.argv) >= 2:
        articles = sys.argv[1]
    else:
        articles = "data/raw/visualnews/articles.tar.gz"

    print(f"Loading index from {articles} ...")
    r = VisualNewsResolver(articles_path=articles)
    print(f"Indexed {len(r):,} image_ids")
    print()

    # Show the path for a known id
    sample_id = next(iter(r._index))
    print(f"Sample resolution: image_id={sample_id} -> {r.path_for(sample_id)}")

    # Coverage check against NewsCLIPpings val annotations, if present
    nclip_val = Path(
        "data/raw/NewsCLIPpings/news_clippings/data/merged_balanced/val.json"
    )
    if nclip_val.exists():
        import json as _json
        with open(nclip_val) as f:
            ann = _json.load(f)["annotations"]
        ids = {a["image_id"] for a in ann} | {a["id"] for a in ann}
        found, total = r.coverage(ids)
        print(f"NewsCLIPpings val coverage: {found}/{total} unique ids resolve "
              f"({found / max(total, 1) * 100:.1f}%)")
