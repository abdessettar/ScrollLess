"""Hacker News collector.

Uses both public HN APIs, since each covers half of what is needed:

- Algolia search lists high-score stories in one request.
- Algolia items returns a story's whole comment tree in one request, but in
  chronological order.
- Firebase returns a story's top-level comment IDs in HN's ranked order, but
  only one item per request.

Each story therefore costs two requests in ``enrich()``: the ranking from
Firebase and the comment bodies from Algolia.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from scrollless.collectors._text import visible_text
from scrollless.collectors.base import Collector
from scrollless.config import HackerNewsConfig
from scrollless.exceptions import CollectorError
from scrollless.models import Comment, Post

log = logging.getLogger(__name__)

_ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
_ALGOLIA_ITEM_URL = "https://hn.algolia.com/api/v1/items/{id}"
_FIREBASE_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
_HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"

_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_MAX_TOP_COMMENTS = 5
_MAX_NESTED_PER_COMMENT = 2


class HackerNewsCollector(Collector):
    SOURCE = "hn"

    def __init__(self, cfg: HackerNewsConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._client = client or httpx.Client(
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": "scrollless/0.1"},
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
            self._owns_client = False

    def collect(self) -> list[Post]:
        if not self.cfg.enabled:
            log.info("HN collector disabled by config; skipping")
            return []
        try:
            hits = self._search()
        except httpx.HTTPError as e:
            raise CollectorError("hn", "search", e) from e

        posts: list[Post] = []
        for hit in hits:
            try:
                posts.append(self._hit_to_post(hit))
            except (KeyError, ValueError, httpx.HTTPError) as e:
                log.warning("Skipping HN item %s: %s", hit.get("objectID", "?"), e)
        log.info("HN: collected %d posts", len(posts))
        return posts

    def _search(self) -> list[dict[str, Any]]:
        tags = f"({','.join(self.cfg.tags)})" if len(self.cfg.tags) > 1 else self.cfg.tags[0]
        # /search sorts by popularity, so without a date filter it returns
        # the most popular stories of all time.
        filters = [f"points>={self.cfg.min_score}"]
        if self.cfg.window_hours > 0:
            cutoff = int((datetime.now(UTC) - timedelta(hours=self.cfg.window_hours)).timestamp())
            filters.append(f"created_at_i>{cutoff}")
        params = {
            "tags": tags,
            "numericFilters": ",".join(filters),
            "hitsPerPage": str(self.cfg.max_posts),
        }
        resp = self._client.get(_ALGOLIA_SEARCH_URL, params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        hits: list[dict[str, Any]] = data.get("hits", [])
        return hits

    def enrich(self, posts: list[Post]) -> list[Post]:
        """Fetch comment trees for the posts that survived ranking."""
        if self.cfg.comment_depth <= 0:
            return posts
        enriched = 0
        for post in posts:
            if post.source != self.SOURCE:
                continue
            post.comments = self._fetch_comments(post.id)
            enriched += 1
        log.info("HN: fetched comment trees for %d posts", enriched)
        return posts

    def _hit_to_post(self, hit: dict[str, Any]) -> Post:
        item_id = hit["objectID"]
        external = hit.get("url") or None
        return Post(
            id=item_id,
            source="hn",
            channel="hn",
            title=hit.get("title") or hit.get("story_title") or "(no title)",
            url=_HN_ITEM_URL.format(id=item_id),
            external_url=external,
            score=int(hit.get("points") or 0),
            num_comments=int(hit.get("num_comments") or 0),
            created_utc=datetime.fromtimestamp(int(hit["created_at_i"]), tz=UTC),
            body=visible_text(hit.get("story_text") or ""),
            comments=[],  # filled by enrich()
            author=hit.get("author") or "[unknown]",
        )

    def _fetch_comments(self, item_id: str) -> list[Comment]:
        """Two requests: HN's ranking, then every body in one go."""
        tree = self._fetch_json(_ALGOLIA_ITEM_URL.format(id=item_id))
        if not tree:
            return []
        children = [c for c in (tree.get("children") or []) if _is_live(c)]
        by_id = {int(c["id"]): c for c in children if c.get("id") is not None}

        ranked = self._ranked_comment_ids(item_id)
        ordered = [by_id[i] for i in ranked if i in by_id]
        if not ordered:
            # No ranking available: fall back to chronological order.
            log.debug("HN %s: no ranking available, using chronological order", item_id)
            ordered = children

        out: list[Comment] = []
        for node in ordered[:_MAX_TOP_COMMENTS]:
            out.append(_to_comment(node, depth=0))
            if self.cfg.comment_depth >= 2:
                nested = [c for c in (node.get("children") or []) if _is_live(c)]
                out.extend(_to_comment(sub, depth=1) for sub in nested[:_MAX_NESTED_PER_COMMENT])
        return out

    def _ranked_comment_ids(self, item_id: str) -> list[int]:
        """HN's own comment ranking, which only Firebase exposes."""
        item = self._fetch_json(_FIREBASE_ITEM_URL.format(id=item_id))
        if not item:
            return []
        return [int(k) for k in (item.get("kids") or [])]

    def _fetch_json(self, url: str) -> dict[str, Any] | None:
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.debug("HN fetch failed for %s: %s", url, e)
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None


def _is_live(node: dict[str, Any]) -> bool:
    """Algolia represents deleted and dead comments with a null body."""
    return node.get("type") == "comment" and bool(node.get("text"))


def _to_comment(node: dict[str, Any], depth: int) -> Comment:
    return Comment(
        author=node.get("author") or "[unknown]",
        body=visible_text(node.get("text") or ""),
        score=0,  # HN doesn't expose comment scores, on either API
        depth=depth,
    )
