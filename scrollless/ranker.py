"""Ranks posts so the digest surfaces high-signal items first.

The score multiplies three transparent signals:

1. ``score``: raw upvotes or HN points.
2. Recency: up to +20% for a brand new post, decaying linearly to 0 at 24 h.
3. Engagement: up to +30% when comments are numerous relative to score.

Returns the top N posts according to the configured ``digest.max_items``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import groupby

from scrollless.models import Post

log = logging.getLogger(__name__)

_RECENCY_BONUS_AT_ZERO = 0.20
_RECENCY_HOURS = 24.0
_ENGAGEMENT_RATIO_CAP = 0.30


def _recency_factor(age_hours: float) -> float:
    """1.20 for fresh posts, decaying linearly to 1.00 at 24 h, then flat."""
    if age_hours <= 0:
        return 1.0 + _RECENCY_BONUS_AT_ZERO
    if age_hours >= _RECENCY_HOURS:
        return 1.0
    return 1.0 + _RECENCY_BONUS_AT_ZERO * (1.0 - age_hours / _RECENCY_HOURS)


def _engagement_factor(score: int, num_comments: int) -> float:
    """Boost posts where discussion is dense relative to upvotes, capped at +30%."""
    if score <= 0:
        return 1.0
    ratio = num_comments / score
    return 1.0 + min(ratio, _ENGAGEMENT_RATIO_CAP)


def rank_score(post: Post, *, now: datetime | None = None) -> float:
    now = now or datetime.now(UTC)
    age_hours = max(0.0, (now - post.created_utc).total_seconds() / 3600.0)
    return (
        post.score * _recency_factor(age_hours) * _engagement_factor(post.score, post.num_comments)
    )


def rank(
    posts: Iterable[Post],
    *,
    max_items: int,
    per_channel_max: int | None = None,
    channel_quotas: dict[str, int] | None = None,
    now: datetime | None = None,
) -> list[Post]:
    """Sort by composite score (desc) and truncate to ``max_items``.

    When per-channel limits are set, each channel (subreddit name or ``"hn"``)
    contributes at most ``channel_quotas[channel]`` items if listed there,
    otherwise ``per_channel_max``. Both ``None`` means no cap. The merged set
    is re-sorted by composite score globally, so the digest still reads
    "best first" overall.

    Per-channel caps stop large communities, whose scores can be ten times
    higher than elsewhere, from crowding out everything else.
    """
    ts = now or datetime.now(UTC)
    scored = sorted(posts, key=lambda p: rank_score(p, now=ts), reverse=True)
    total = len(scored)
    quotas = channel_quotas or {}

    if per_channel_max is not None or quotas:
        scored = _apply_per_channel_cap(scored, per_channel_max, quotas, now=ts)

    top = scored[:max_items]
    log.info("Ranked %d posts, keeping top %d", total, len(top))
    return top


def _apply_per_channel_cap(
    scored: list[Post],
    default_cap: int | None,
    overrides: dict[str, int],
    *,
    now: datetime,
) -> list[Post]:
    """Keep at most ``overrides[channel]`` (or ``default_cap``) posts per channel."""
    by_channel = sorted(scored, key=lambda p: p.channel)
    capped: list[Post] = []
    for channel, group in groupby(by_channel, key=lambda p: p.channel):
        quota = overrides.get(channel, default_cap)
        ordered = sorted(group, key=lambda p: rank_score(p, now=now), reverse=True)
        if quota is None:
            capped.extend(ordered)
        else:
            capped.extend(ordered[:quota])
    capped.sort(key=lambda p: rank_score(p, now=now), reverse=True)
    return capped
