"""Tests for `src.visualnews_resolver.VisualNewsResolver`.

Builds an index from a tiny synthetic articles.tar.gz fixture so the
test suite stays offline + fast. The integration check against the
real articles.tar.gz on disk is exercised by the smoke `__main__`
inside the module itself.
"""
from __future__ import annotations

import io
import pickle
import sys
import tarfile
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visualnews_resolver import VisualNewsResolver  # noqa: E402


def _make_pickle_bytes(entries: dict[int, dict]) -> bytes:
    return pickle.dumps(entries)


@pytest.fixture
def fake_articles(tmp_path: Path) -> Path:
    """Build a 3-source articles.tar.gz with a handful of entries."""
    out = tmp_path / "articles.tar.gz"
    sources = {
        "bbc_1": {
            1324908: {"ori_id": 38392, "image_id": 1324908},
            1325046: {"ori_id": 38435, "image_id": 1325046},
        },
        "guardian": {
            5: {"ori_id": 102969, "image_id": 5},
        },
        "washington_post": {
            7: {"ori_id": 13105, "image_id": 7},
        },
    }

    with tarfile.open(out, "w:gz") as tf:
        for tag, entries in sources.items():
            data = _make_pickle_bytes(entries)
            info = tarfile.TarInfo(f"./articles/processed_{tag}.p")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return out


# ---------------------------------------------------------------------------
# core behaviour
# ---------------------------------------------------------------------------


def test_index_size_and_contains(fake_articles: Path):
    r = VisualNewsResolver(articles_path=fake_articles, cache_index=False)
    assert len(r) == 4
    assert 1324908 in r
    assert 9999999 not in r


def test_path_for_bbc(fake_articles: Path):
    r = VisualNewsResolver(articles_path=fake_articles, cache_index=False)
    p = r.path_for(1324908)
    assert p is not None
    assert p.as_posix() == "origin/bbc/images/0038/392.jpg"


def test_path_for_guardian_three_digit_padding(fake_articles: Path):
    """ori_id 102969 -> group 0102, file 969.jpg."""
    r = VisualNewsResolver(articles_path=fake_articles, cache_index=False)
    assert r.path_for(5).as_posix() == "origin/guardian/images/0102/969.jpg"


def test_path_for_washington_post_zero_pad(fake_articles: Path):
    """ori_id 13105 -> group 0013, file 105.jpg (both zero-padded)."""
    r = VisualNewsResolver(articles_path=fake_articles, cache_index=False)
    assert (
        r.path_for(7).as_posix() == "origin/washington_post/images/0013/105.jpg"
    )


def test_path_for_unknown_returns_none(fake_articles: Path):
    r = VisualNewsResolver(articles_path=fake_articles, cache_index=False)
    assert r.path_for(424242) is None


def test_source_for(fake_articles: Path):
    r = VisualNewsResolver(articles_path=fake_articles, cache_index=False)
    assert r.source_for(1324908) == "bbc"
    assert r.source_for(5) == "guardian"
    assert r.source_for(424242) is None


def test_coverage(fake_articles: Path):
    r = VisualNewsResolver(articles_path=fake_articles, cache_index=False)
    found, total = r.coverage([1324908, 5, 424242, 9999999])
    assert (found, total) == (2, 4)


# ---------------------------------------------------------------------------
# bbc_1 + bbc_2 both map to bbc/
# ---------------------------------------------------------------------------


def test_bbc_1_and_bbc_2_share_bbc_folder(tmp_path: Path):
    out = tmp_path / "articles.tar.gz"
    sources = {
        "bbc_1": {1: {"ori_id": 1000, "image_id": 1}},
        "bbc_2": {2: {"ori_id": 2000, "image_id": 2}},
    }
    with tarfile.open(out, "w:gz") as tf:
        for tag, entries in sources.items():
            data = _make_pickle_bytes(entries)
            info = tarfile.TarInfo(f"./articles/processed_{tag}.p")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    r = VisualNewsResolver(articles_path=out, cache_index=False)
    assert r.source_for(1) == "bbc"
    assert r.source_for(2) == "bbc"
    # Both go under origin/bbc/ regardless of which scrape batch.
    assert r.path_for(1).parts[1] == "bbc"
    assert r.path_for(2).parts[1] == "bbc"


# ---------------------------------------------------------------------------
# index cache
# ---------------------------------------------------------------------------


def test_index_cache_roundtrip(fake_articles: Path):
    """First build writes JSON cache; second build reads it."""
    cache = fake_articles.with_name(fake_articles.name + ".index.json")
    assert not cache.exists()

    r1 = VisualNewsResolver(articles_path=fake_articles, cache_index=True)
    assert cache.exists()
    p1 = r1.path_for(1324908)

    # Rebuild — should hit the cache (no tarfile open needed).
    r2 = VisualNewsResolver(articles_path=fake_articles, cache_index=True)
    p2 = r2.path_for(1324908)
    assert p1 == p2
    assert len(r2) == len(r1)


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_missing_articles_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        VisualNewsResolver(
            articles_path=tmp_path / "nope.tar.gz",
            cache_index=False,
        )


def test_origin_root_override(fake_articles: Path):
    r = VisualNewsResolver(
        articles_path=fake_articles,
        origin_root="data/raw/visualnews/origin",
        cache_index=False,
    )
    p = r.path_for(1324908)
    assert p.as_posix() == "data/raw/visualnews/origin/bbc/images/0038/392.jpg"
