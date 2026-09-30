"""Reddit collector using the official OAuth API.

A "script" app authenticates app-only (``grant_type=client_credentials``),
which gives read-only access to public listings. No account password is used.
One token is requested per run. Reddit allows 100 requests per minute per
client and reports the remaining budget in ``X-Ratelimit-*`` headers.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from scrollless.collectors.base import Collector
from scrollless.config import RedditConfig
from scrollless.exceptions import (
    CollectorError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from scrollless.models import Comment, Post, domain_of

log = logging.getLogger(__name__)

_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_API_BASE = "https://oauth.reddit.com"
_WEB_BASE = "https://www.reddit.com"
_HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

# Reddit asks for a descriptive User-Agent. Set REDDIT_USER_AGENT to your own.
_DEFAULT_USER_AGENT = "linux:scrollless:0.1.0 (by /u/unknown)"

# With fewer requests than this left in the window, wait for the reset.
_RATE_LIMIT_FLOOR = 5


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingCredentialsError(name, source="reddit")
    return value


class RedditCollector(Collector):
    SOURCE = "reddit"

    def __init__(
        self,
        cfg: RedditConfig,
        client: httpx.Client | None = None,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self._sleep = sleep
        # Fail before any network I/O when credentials are missing.
        self._client_id = _require_env("REDDIT_CLIENT_ID")
        self._client_secret = _require_env("REDDIT_CLIENT_SECRET")
        self._user_agent = os.environ.get("REDDIT_USER_AGENT") or _DEFAULT_USER_AGENT
        self._client = client or httpx.Client(
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": self._user_agent},
        )
        self._owns_client = client is None
        self._token: str | None = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
            self._owns_client = False

    def collect(self) -> list[Post]:
        self._authenticate()
        posts: list[Post] = []
        for sub_name in self.cfg.subreddits:
            try:
                posts.extend(self._collect_sub(sub_name))
            except httpx.HTTPError as e:
                log.warning("Skipping r/%s: %s", sub_name, CollectorError("reddit", sub_name, e))
                continue
            self._throttle()
        log.info("Reddit: %d posts across %d subs", len(posts), len(self.cfg.subreddits))
        return posts

    def _authenticate(self) -> None:
        """Exchange the app credentials for an app-only bearer token."""
        try:
            resp = self._client.post(
                _TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
                headers={"User-Agent": self._user_agent},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise InvalidCredentialsError(
                    "reddit",
                    f"Reddit rejected the app credentials (HTTP {e.response.status_code}). "
                    "Check REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET against the script app "
                    "at https://www.reddit.com/prefs/apps.",
                ) from e
            raise CollectorError("reddit", "auth", e) from e
        except httpx.HTTPError as e:
            raise CollectorError("reddit", "auth", e) from e

        payload = resp.json()
        token = payload.get("access_token") if isinstance(payload, dict) else None
        if not token:
            raise InvalidCredentialsError(
                "reddit", f"token response carried no access_token: {payload!r}"
            )
        self._token = str(token)
        log.debug("Reddit: obtained app-only token (expires in %ss)", payload.get("expires_in"))

    def _collect_sub(self, sub_name: str) -> list[Post]:
        listing = self._fetch_listing(sub_name)
        children = listing.get("data", {}).get("children", [])

        threshold = self.cfg.threshold_for(sub_name)
        posts: list[Post] = []
        for child in children:
            data = child.get("data", {})
            if data.get("stickied") or int(data.get("score", 0)) < threshold:
                continue
            if self.cfg.should_skip(
                author=str(data.get("author") or ""),
                title=str(data.get("title", "")),
                domain="" if data.get("is_self") else domain_of(str(data.get("url") or "")),
            ):
                log.debug("r/%s: skipping %s as recurring or bot", sub_name, data.get("id"))
                continue
            posts.append(self._to_post(data, sub_name, comments=[]))
            self._throttle()
        log.info("r/%s: %d posts (>= score %d)", sub_name, len(posts), threshold)
        return posts

    def _fetch_listing(self, sub_name: str) -> dict[str, Any]:
        params = {"limit": str(self.cfg.max_posts_per_sub), "raw_json": "1"}
        if self.cfg.sort == "top":
            params["t"] = self.cfg.time_filter
        data = self._get(f"/r/{sub_name}/{self.cfg.sort}", params)
        return data if isinstance(data, dict) else {}

    def enrich(self, posts: list[Post]) -> list[Post]:
        """One comment request per surviving post, after dedup and ranking."""
        if not self.cfg.max_comments:
            return posts
        enriched = 0
        for post in posts:
            if post.source != self.SOURCE:
                continue
            post.comments = self._fetch_comments(post.channel, post.id)
            enriched += 1
            self._throttle()
        log.info("Reddit: fetched comments for %d posts", enriched)
        return posts

    def _fetch_comments(self, sub_name: str, post_id: str) -> list[Comment]:
        params = {
            "limit": str(self.cfg.max_comments),
            "sort": "top",
            "depth": "1",
            "raw_json": "1",
        }
        try:
            payload = self._get(f"/r/{sub_name}/comments/{post_id}", params)
        except httpx.HTTPError as e:
            log.debug("Comments fetch failed for %s/%s: %s", sub_name, post_id, e)
            return []
        # Comments are the second listing in the response array.
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        children = payload[1].get("data", {}).get("children", [])
        out: list[Comment] = []
        for child in children[: self.cfg.max_comments]:
            if child.get("kind") != "t1":
                continue
            data = child.get("data", {})
            out.append(
                Comment(
                    author=data.get("author") or "[deleted]",
                    body=data.get("body") or "",
                    score=int(data.get("score") or 0),
                    depth=0,
                )
            )
        return out

    def _get(self, path: str, params: dict[str, str]) -> Any:
        """Authenticated GET against the OAuth host, honouring the rate budget."""
        resp = self._client.get(
            f"{_API_BASE}{path}",
            params=params,
            headers={
                "Authorization": f"bearer {self._token}",
                "User-Agent": self._user_agent,
            },
        )
        resp.raise_for_status()
        self._respect_rate_limit(resp)
        return resp.json()

    def _respect_rate_limit(self, resp: httpx.Response) -> None:
        """Wait out the window when the 100/min budget is nearly spent."""
        raw_remaining = resp.headers.get("x-ratelimit-remaining")
        if raw_remaining is None:
            return
        try:
            remaining = float(raw_remaining)
            reset = float(resp.headers.get("x-ratelimit-reset", "0"))
        except ValueError:
            return
        if remaining <= _RATE_LIMIT_FLOOR and reset > 0:
            log.warning(
                "Reddit rate budget nearly spent (%.0f left); sleeping %.0fs for the reset",
                remaining,
                reset,
            )
            self._sleep(reset)

    def _to_post(self, data: dict[str, Any], sub_name: str, comments: list[Comment]) -> Post:
        is_self = bool(data.get("is_self"))
        return Post(
            id=str(data["id"]),
            source="reddit",
            channel=sub_name,
            title=str(data.get("title", "")),
            url=f"{_WEB_BASE}{data['permalink']}",
            external_url=str(data["url"]) if not is_self else None,
            score=int(data.get("score", 0)),
            num_comments=int(data.get("num_comments", 0)),
            created_utc=datetime.fromtimestamp(float(data["created_utc"]), tz=UTC),
            body=data.get("selftext") or "",
            comments=comments,
            author=str(data.get("author") or "[deleted]"),
        )

    def _throttle(self) -> None:
        if self.cfg.request_delay_seconds > 0:
            self._sleep(self.cfg.request_delay_seconds)
