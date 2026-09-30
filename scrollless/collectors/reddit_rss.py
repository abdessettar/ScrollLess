"""Reddit collector using the public RSS feeds.

The ``.rss`` feeds are the only Reddit interface still available without
credentials (the ``.json`` endpoints return 403 and new OAuth apps need manual
approval). They allow about one request per minute, so a run makes:

    len(subreddits)            listing requests, in collect()
  + surviving Reddit posts     comment requests, in enrich()

The collector paces itself using Reddit's ``x-ratelimit-*`` headers and retries
a 429 after the advertised reset.

Feeds carry no score or comment count. Reddit's own ordering (``top`` is
best-first) is turned into a descending synthetic score so ranking and quotas
still work, and ``score_is_estimated`` keeps that number out of the digest.
``min_score`` and ``min_scores`` have no effect in this mode.
"""

from __future__ import annotations

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from scrollless.collectors._text import visible_text
from scrollless.collectors.base import Collector
from scrollless.config import RedditConfig
from scrollless.exceptions import CollectorError
from scrollless.models import Comment, Post, domain_of

log = logging.getLogger(__name__)

_BASE_URL = "https://www.reddit.com"
_ATOM = "{http://www.w3.org/2005/Atom}"
_HTTP_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
_DEFAULT_USER_AGENT = "linux:scrollless:0.1.0 (feed reader)"

# Small pad so a retry does not land exactly on the window boundary.
_RATE_LIMIT_PAD_SECONDS = 1.0
_FALLBACK_WAIT_SECONDS = 60.0
_MAX_RATE_LIMIT_RETRIES = 3

# A post with no comments and almost no body leaves the model nothing to
# summarize but the title, so it is dropped.
_MIN_SUMMARIZABLE_BODY_CHARS = 200

# Selftext sits between these markers; after SC_ON comes Reddit's
# "submitted by ... [link] [comments]" footer.
_BODY_RE = re.compile(r"<!-- SC_OFF -->(.*?)<!-- SC_ON -->", re.S)
_LINK_RE = re.compile(r'<a href="([^"]+)">\s*\[link\]\s*</a>')


class RedditRSSCollector(Collector):
    SOURCE = "reddit"

    def __init__(
        self,
        cfg: RedditConfig,
        client: httpx.Client | None = None,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self._sleep = sleep
        self._user_agent = _DEFAULT_USER_AGENT
        self._client = client or httpx.Client(
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": self._user_agent},
            follow_redirects=True,
        )
        self._owns_client = client is None
        # A spent budget schedules a wait paid by the next request, so a run
        # never ends with a pointless sleep.
        self._pending_wait = 0.0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
            self._owns_client = False

    def collect(self) -> list[Post]:
        posts: list[Post] = []
        for sub_name in self.cfg.subreddits:
            try:
                posts.extend(self._collect_sub(sub_name))
            except httpx.HTTPError as e:
                log.warning("Skipping r/%s: %s", sub_name, CollectorError("reddit", sub_name, e))
                continue
        log.info("Reddit (RSS): %d posts across %d subs", len(posts), len(self.cfg.subreddits))
        return posts

    def _collect_sub(self, sub_name: str) -> list[Post]:
        """One request: the listing. Comments wait for enrich()."""
        params = {"limit": str(self.cfg.max_posts_per_sub)}
        if self.cfg.sort == "top":
            params["t"] = self.cfg.time_filter
        feed = self._get(f"/r/{sub_name}/{self.cfg.sort}.rss", params)

        entries = _parse_entries(feed)
        # Filter before scoring so skipped posts do not take a rank position.
        kept = [
            entry
            for entry in entries
            if not self.cfg.should_skip(
                author=entry["author"],
                title=entry["title"],
                domain=domain_of(_external_url(entry["content"], entry["url"])),
            )
        ]
        posts = [
            _to_post(entry, sub_name, position=position, total=len(kept))
            for position, entry in enumerate(kept)
        ]
        skipped = len(entries) - len(kept)
        log.info(
            "r/%s: %d posts%s",
            sub_name,
            len(posts),
            f" ({skipped} skipped as recurring or bot)" if skipped else "",
        )
        return posts

    def enrich(self, posts: list[Post]) -> list[Post]:
        """Fetch one comment feed (about a minute each) per surviving post."""
        if not self.cfg.max_comments:
            return posts
        kept: list[Post] = []
        enriched = dropped = 0
        for post in posts:
            if post.source != self.SOURCE:
                kept.append(post)
                continue
            post.comments = self._fetch_comments(post.channel, post.id)
            enriched += 1
            if not post.comments and len(post.body) < _MIN_SUMMARIZABLE_BODY_CHARS:
                log.debug("r/%s: dropping %s (no comments, no body)", post.channel, post.id)
                dropped += 1
                continue
            kept.append(post)
        log.info(
            "Reddit (RSS): fetched comments for %d posts (%d dropped as unsummarizable)",
            enriched,
            dropped,
        )
        return kept

    def _fetch_comments(self, sub_name: str, post_id: str) -> list[Comment]:
        params = {"sort": "top", "limit": str(self.cfg.max_comments)}
        try:
            feed = self._get(f"/r/{sub_name}/comments/{post_id}.rss", params)
        except httpx.HTTPError as e:
            log.debug("Comments fetch failed for %s/%s: %s", sub_name, post_id, e)
            return []
        out: list[Comment] = []
        for entry in _parse_entries(feed):
            if not entry["id"].startswith("t1_"):
                continue  # the post itself can show up in its own feed
            out.append(
                Comment(
                    author=entry["author"],
                    body=visible_text(entry["content"]),
                    score=0,  # feeds don't carry comment scores
                    depth=0,  # nor the reply tree
                )
            )
            if len(out) >= self.cfg.max_comments:
                break
        return out

    def _get(self, path: str, params: dict[str, str]) -> str:
        """GET a feed, waiting out Reddit's one-request-per-window budget."""
        url = f"{_BASE_URL}{path}"
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            if self._pending_wait > 0:
                self._sleep(self._pending_wait)
                self._pending_wait = 0.0
            resp = self._client.get(url, params=params, headers={"User-Agent": self._user_agent})
            if resp.status_code == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
                wait = _reset_hint(resp)
                log.info(
                    "Reddit rate limit hit on %s; waiting %.0fs (attempt %d/%d)",
                    path,
                    wait,
                    attempt + 1,
                    _MAX_RATE_LIMIT_RETRIES,
                )
                self._sleep(wait)
                continue
            resp.raise_for_status()
            self._pace(resp)
            return resp.text
        # Unreachable: the loop always returns or raises.
        raise CollectorError("reddit", path, RuntimeError("rate limit retries exhausted"))

    def _pace(self, resp: httpx.Response) -> None:
        """Note how long the next request has to wait, per Reddit's headers."""
        raw = resp.headers.get("x-ratelimit-remaining")
        try:
            remaining = float(raw) if raw is not None else None
        except ValueError:
            remaining = None
        if remaining is not None and remaining <= 0:
            self._pending_wait = _reset_hint(resp)
            log.debug(
                "Reddit budget spent; next feed waits %.0fs for the window",
                self._pending_wait,
            )
        elif self.cfg.request_delay_seconds > 0:
            self._pending_wait = self.cfg.request_delay_seconds


def _reset_hint(resp: httpx.Response) -> float:
    """Seconds to wait, from Reddit's own headers, with a sane fallback."""
    for header in ("x-ratelimit-reset", "retry-after"):
        raw = resp.headers.get(header)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            return value + _RATE_LIMIT_PAD_SECONDS
    return _FALLBACK_WAIT_SECONDS


def _parse_entries(xml_text: str) -> list[dict[str, str]]:
    """Flatten an Atom feed into plain dicts. Source is reddit.com over TLS."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.warning("Unparseable feed: %s", e)
        return []
    out: list[dict[str, str]] = []
    for entry in root.findall(f"{_ATOM}entry"):
        link = entry.find(f"{_ATOM}link")
        author = entry.find(f"{_ATOM}author/{_ATOM}name")
        out.append(
            {
                "id": (entry.findtext(f"{_ATOM}id") or "").strip(),
                "title": (entry.findtext(f"{_ATOM}title") or "").strip(),
                "url": (link.get("href") if link is not None else "") or "",
                "published": (
                    entry.findtext(f"{_ATOM}published") or entry.findtext(f"{_ATOM}updated") or ""
                ).strip(),
                "author": ((author.text or "") if author is not None else "")
                .strip()
                .removeprefix("/u/")
                or "[deleted]",
                "content": entry.findtext(f"{_ATOM}content") or "",
            }
        )
    return out


def _to_post(entry: dict[str, Any], sub_name: str, *, position: int, total: int) -> Post:
    permalink = entry["url"]
    content = entry["content"]
    body_match = _BODY_RE.search(content)
    return Post(
        id=entry["id"].removeprefix("t3_"),
        source="reddit",
        channel=sub_name,
        title=entry["title"],
        url=permalink,
        external_url=_external_url(content, permalink),
        # Feeds have no score: derive one from the position in the listing.
        score=max(total - position, 1),
        num_comments=0,  # not in the feed; keeps the engagement factor neutral
        created_utc=_parse_timestamp(entry["published"]),
        body=visible_text(body_match.group(1) if body_match else ""),
        comments=[],
        author=entry["author"],
        score_is_estimated=True,
    )


def _external_url(content: str, permalink: str) -> str | None:
    """The `[link]` anchor points off-site for link posts, home for self-posts."""
    match = _LINK_RE.search(html.unescape(content))
    if not match:
        return None
    href = match.group(1)
    return None if href.rstrip("/") == permalink.rstrip("/") else href


def _parse_timestamp(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw).astimezone(UTC)
    except ValueError:
        log.debug("Unparseable feed timestamp %r; falling back to now", raw)
        return datetime.now(UTC)
