"""SQLite-backed seen-post log.

A post is identified by ``(source, id)``, since IDs are only unique within a
source. The linked ``external_url`` is stored too, so the same article posted
to several subreddits or to both Reddit and HN appears only once, across runs
and within a single run.

The store keeps rows for ``window_days`` and is pruned on each ``mark()`` call.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scrollless.models import Post

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    source TEXT NOT NULL,
    post_id TEXT NOT NULL,
    external_url TEXT,
    seen_at TEXT NOT NULL,
    PRIMARY KEY (source, post_id)
);
CREATE INDEX IF NOT EXISTS idx_seen_external_url ON seen(external_url);
CREATE INDEX IF NOT EXISTS idx_seen_at ON seen(seen_at);
"""


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    return url.strip().rstrip("/").lower() or None


class SeenStore:
    """Tracks which posts have already been digested.

    Use as a context manager so the underlying connection is closed cleanly.
    """

    def __init__(self, path: str | Path, window_days: int = 7) -> None:
        self.path = Path(path)
        self.window_days = window_days
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)  # autocommit
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(_SCHEMA)

    def __enter__(self) -> SeenStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def filter_new(self, posts: Iterable[Post]) -> list[Post]:
        """Return only posts not seen within the window, and not repeated here.

        The store catches anything sent by a previous run; the in-batch pass
        catches the same article arriving twice in this run. Duplicates are
        resolved in favour of the copy with a real score, then the higher
        score. Output order does not matter: the ranker sorts next.
        """
        cutoff = self._cutoff_iso()
        ordered = sorted(posts, key=lambda p: (not p.score_is_estimated, p.score), reverse=True)
        batch_ids: set[tuple[str, str]] = set()
        batch_urls: set[str] = set()
        with closing(self._conn.cursor()) as cur:
            out: list[Post] = []
            for p in ordered:
                norm_url = _normalize_url(p.external_url)
                if (p.source, p.id) in batch_ids or (norm_url and norm_url in batch_urls):
                    log.debug("Dropping duplicate within this run: %s:%s", p.source, p.id)
                    continue
                row = cur.execute(
                    "SELECT 1 FROM seen "
                    "WHERE seen_at >= ? "
                    "  AND ((source = ? AND post_id = ?) "
                    "       OR (? IS NOT NULL AND external_url = ?)) "
                    "LIMIT 1",
                    (cutoff, p.source, p.id, norm_url, norm_url),
                ).fetchone()
                if row is None:
                    out.append(p)
                    batch_ids.add((p.source, p.id))
                    if norm_url:
                        batch_urls.add(norm_url)
        return out

    def mark(self, posts: Iterable[Post]) -> None:
        """Record posts as seen and prune entries outside the window."""
        now_iso = datetime.now(UTC).isoformat(timespec="seconds")
        rows = [(p.source, p.id, _normalize_url(p.external_url), now_iso) for p in posts]
        with closing(self._conn.cursor()) as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO seen (source, post_id, external_url, seen_at) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            deleted = cur.execute(
                "DELETE FROM seen WHERE seen_at < ?", (self._cutoff_iso(),)
            ).rowcount
        if deleted:
            log.debug("Pruned %d stale entries from seen store", deleted)

    def _cutoff_iso(self) -> str:
        return (datetime.now(UTC) - timedelta(days=self.window_days)).isoformat(timespec="seconds")
