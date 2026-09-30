from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from scrollless.dedup import SeenStore
from scrollless.models import Post


def _post(
    *,
    id_: str = "a1",
    source: str = "reddit",
    external_url: str | None = None,
) -> Post:
    return Post(
        id=id_,
        source=source,  # type: ignore[arg-type]
        channel="ch",
        title="t",
        url=f"https://{source}/{id_}",
        external_url=external_url,
        score=10,
        num_comments=0,
        created_utc=datetime.now(UTC),
    )


def test_filter_new_returns_all_on_empty_store(tmp_path: Path) -> None:
    with SeenStore(tmp_path / "seen.sqlite") as store:
        posts = [_post(id_="a"), _post(id_="b")]
        assert store.filter_new(posts) == posts


def test_mark_then_filter_excludes_seen(tmp_path: Path) -> None:
    with SeenStore(tmp_path / "seen.sqlite") as store:
        p1, p2 = _post(id_="a"), _post(id_="b")
        store.mark([p1])
        assert store.filter_new([p1, p2]) == [p2]


def test_filter_deduplicates_by_external_url_across_sources(tmp_path: Path) -> None:
    with SeenStore(tmp_path / "seen.sqlite") as store:
        reddit_post = _post(
            id_="r1", source="reddit", external_url="https://shared.example/article"
        )
        hn_post = _post(id_="h1", source="hn", external_url="https://shared.example/article")
        store.mark([reddit_post])
        # Different (source, id) but same external_url: must be filtered.
        assert store.filter_new([hn_post]) == []


def test_external_url_normalization(tmp_path: Path) -> None:
    """Trailing slash and case differences should not defeat dedup."""
    with SeenStore(tmp_path / "seen.sqlite") as store:
        a = _post(id_="a", source="reddit", external_url="https://Example.com/foo/")
        b = _post(id_="b", source="hn", external_url="https://example.com/foo")
        store.mark([a])
        assert store.filter_new([b]) == []


def test_missing_external_url_does_not_collide(tmp_path: Path) -> None:
    """Posts without external_url (e.g. self-posts) must not all collapse."""
    with SeenStore(tmp_path / "seen.sqlite") as store:
        a = _post(id_="a", external_url=None)
        b = _post(id_="b", external_url=None)
        store.mark([a])
        assert store.filter_new([b]) == [b]


def test_entries_outside_window_are_pruned_and_ignored(tmp_path: Path) -> None:
    db = tmp_path / "seen.sqlite"
    with SeenStore(db, window_days=7) as store:
        store.mark([_post(id_="old")])

    ancient = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("UPDATE seen SET seen_at = ?", (ancient,))

    old = _post(id_="old")
    fresh = _post(id_="fresh")
    with SeenStore(db, window_days=7) as store:
        # Old entry should not block re-inclusion of the same id.
        assert store.filter_new([old]) == [old]
        # Marking a new post prunes the stale row.
        store.mark([fresh])
        with closing(sqlite3.connect(db)) as conn:
            remaining = {r[0] for r in conn.execute("SELECT post_id FROM seen")}
    assert remaining == {"fresh"}


def test_mark_is_idempotent(tmp_path: Path) -> None:
    with SeenStore(tmp_path / "seen.sqlite") as store:
        p = _post(id_="a")
        store.mark([p, p])
        store.mark([p])
        with closing(sqlite3.connect(store.path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
    assert count == 1


def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deeper" / "still_deeper" / "seen.sqlite"
    with SeenStore(nested):
        pass
    assert nested.exists()


@pytest.mark.parametrize("window", [1, 7, 30])
def test_window_param_respected(tmp_path: Path, window: int) -> None:
    with SeenStore(tmp_path / "seen.sqlite", window_days=window) as store:
        assert store.window_days == window


def test_duplicate_within_one_batch_is_dropped(tmp_path: Path) -> None:
    """Two submissions of the same article in one run must not both ship."""
    with SeenStore(tmp_path / "seen.sqlite") as store:
        first = _post(id_="h1", external_url="https://shared.example/story")
        second = _post(id_="h2", external_url="https://shared.example/story")
        kept = store.filter_new([first, second])
        assert [p.id for p in kept] == ["h1"]


def test_batch_duplicate_keeps_the_busier_thread(tmp_path: Path) -> None:
    """When one run holds the same story twice, keep the higher-scored copy."""
    quiet = _post(id_="quiet", external_url="https://shared.example/story")
    quiet.score = 440
    busy = _post(id_="busy", external_url="https://shared.example/story")
    busy.score = 696
    with SeenStore(tmp_path / "seen.sqlite") as store:
        kept = store.filter_new([quiet, busy])
    assert [p.id for p in kept] == ["busy"]


def test_batch_duplicate_prefers_real_scores_over_estimated(tmp_path: Path) -> None:
    """A feed-derived score is a position, not a number, so it must not win."""
    estimated = _post(id_="rss", external_url="https://shared.example/story")
    estimated.score = 9
    estimated.score_is_estimated = True
    real = _post(id_="api", external_url="https://shared.example/story")
    real.score = 3
    with SeenStore(tmp_path / "seen.sqlite") as store:
        kept = store.filter_new([estimated, real])
    assert [p.id for p in kept] == ["api"]


def test_distinct_posts_in_one_batch_all_survive(tmp_path: Path) -> None:
    with SeenStore(tmp_path / "seen.sqlite") as store:
        a = _post(id_="a", external_url="https://example.com/a")
        b = _post(id_="b", external_url="https://example.com/b")
        c = _post(id_="c", external_url=None)
        assert len(store.filter_new([a, b, c])) == 3
