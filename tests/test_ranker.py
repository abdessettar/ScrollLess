from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from scrollless.models import Post
from scrollless.ranker import rank, rank_score

NOW = datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)


def _post(
    *,
    id_: str,
    score: int,
    num_comments: int = 0,
    age_hours: float = 12,
    source: str = "reddit",
    channel: str = "ch",
) -> Post:
    return Post(
        id=id_,
        source=source,  # type: ignore[arg-type]
        channel=channel,
        title=id_,
        url=f"https://x/{id_}",
        external_url=None,
        score=score,
        num_comments=num_comments,
        created_utc=NOW - timedelta(hours=age_hours),
    )


def test_higher_score_ranks_higher_when_age_and_engagement_equal() -> None:
    a = _post(id_="a", score=100)
    b = _post(id_="b", score=200)
    assert rank_score(a, now=NOW) < rank_score(b, now=NOW)


def test_fresh_post_outranks_older_at_same_raw_score() -> None:
    fresh = _post(id_="fresh", score=100, age_hours=1)
    stale = _post(id_="stale", score=100, age_hours=48)
    assert rank_score(fresh, now=NOW) > rank_score(stale, now=NOW)


def test_engagement_boosts_comment_heavy_post() -> None:
    quiet = _post(id_="q", score=100, num_comments=5, age_hours=48)
    discussed = _post(id_="d", score=100, num_comments=100, age_hours=48)
    assert rank_score(discussed, now=NOW) > rank_score(quiet, now=NOW)


def test_engagement_bonus_is_capped() -> None:
    """Comment ratio above 0.30 stops adding to the score."""
    moderate = _post(id_="m", score=100, num_comments=30, age_hours=48)
    extreme = _post(id_="e", score=100, num_comments=10_000, age_hours=48)
    assert rank_score(extreme, now=NOW) == pytest.approx(rank_score(moderate, now=NOW))


def test_recency_factor_decays_to_one_past_window() -> None:
    """A post 24 h+ old gets no recency multiplier."""
    edge = _post(id_="edge", score=100, age_hours=24)
    way_past = _post(id_="past", score=100, age_hours=100)
    assert rank_score(edge, now=NOW) == pytest.approx(rank_score(way_past, now=NOW))


def test_zero_score_post_does_not_divide_by_zero() -> None:
    p = _post(id_="z", score=0, num_comments=50, age_hours=1)
    assert rank_score(p, now=NOW) == 0.0


def test_negative_age_treated_as_brand_new() -> None:
    """Clock skew between collector and ranker shouldn't crash or penalize."""
    future = _post(id_="f", score=100, age_hours=-1)
    assert rank_score(future, now=NOW) > 100  # gets full recency bonus


def test_rank_truncates_to_max_items() -> None:
    posts = [_post(id_=str(i), score=10 + i) for i in range(20)]
    top = rank(posts, max_items=5, now=NOW)
    assert len(top) == 5
    # Should be the highest-scored ones, in descending order.
    assert [p.id for p in top] == ["19", "18", "17", "16", "15"]


def test_rank_max_items_greater_than_input() -> None:
    posts = [_post(id_="a", score=10), _post(id_="b", score=20)]
    top = rank(posts, max_items=50, now=NOW)
    assert [p.id for p in top] == ["b", "a"]


def test_rank_empty_input() -> None:
    assert rank([], max_items=5, now=NOW) == []


def test_per_channel_max_caps_each_subreddit() -> None:
    """A high-volume subreddit can't crowd out the rest of the digest."""
    nba = [
        _post(id_=f"n{i}", source="reddit", channel="nba", score=5000 - i, age_hours=2)
        for i in range(15)
    ]
    france = [
        _post(id_=f"f{i}", source="reddit", channel="france", score=600 - i, age_hours=2)
        for i in range(15)
    ]

    # Without the cap, NBA's higher scores would dominate.
    no_cap = rank(nba + france, max_items=10, now=NOW)
    assert all(p.channel == "nba" for p in no_cap)

    # With per_channel_max=5, each subreddit gets at most 5.
    capped = rank(nba + france, max_items=10, per_channel_max=5, now=NOW)
    assert sum(1 for p in capped if p.channel == "nba") == 5
    assert sum(1 for p in capped if p.channel == "france") == 5


def test_channel_quotas_override_default() -> None:
    """Specific channels can have a higher quota than the default."""
    nba = [_post(id_=f"n{i}", channel="nba", score=5000 - i, age_hours=2) for i in range(10)]
    hn = [
        _post(id_=f"h{i}", source="hn", channel="hn", score=500 - i, age_hours=2) for i in range(15)
    ]

    top = rank(
        nba + hn,
        max_items=30,
        per_channel_max=5,
        channel_quotas={"hn": 10},
        now=NOW,
    )
    assert sum(1 for p in top if p.channel == "nba") == 5
    assert sum(1 for p in top if p.channel == "hn") == 10


def test_channel_quotas_picks_best_within_channel() -> None:
    """Each channel's quota is filled by composite score within the channel."""
    posts = [
        _post(id_="n_high", channel="nba", score=9000, age_hours=2),
        _post(id_="n_low", channel="nba", score=100, age_hours=2),
        _post(id_="f_high", channel="france", score=700, age_hours=2),
        _post(id_="f_low", channel="france", score=80, age_hours=2),
    ]
    top = rank(posts, max_items=4, per_channel_max=1, now=NOW)
    ids = [p.id for p in top]
    assert "n_low" not in ids
    assert "f_low" not in ids
    assert ids == ["n_high", "f_high"]  # globally sorted by composite score


def test_per_channel_under_quota_does_not_underfill_others() -> None:
    """Quiet channels just contribute fewer items; others aren't bumped up to compensate."""
    nba = [_post(id_=f"n{i}", channel="nba", score=1000 - i, age_hours=2) for i in range(5)]
    france = [_post(id_="f0", channel="france", score=500, age_hours=2)]

    top = rank(nba + france, max_items=10, per_channel_max=3, now=NOW)
    # NBA caps at 3, France has only 1, so 4 in total.
    assert sum(1 for p in top if p.channel == "nba") == 3
    assert sum(1 for p in top if p.channel == "france") == 1


def test_no_quota_keeps_global_ranking() -> None:
    """No per-channel limits: pure global ranking."""
    posts = [_post(id_=f"r{i}", channel="ch", score=1000 - i) for i in range(5)]
    top = rank(posts, max_items=3, now=NOW)
    assert [p.id for p in top] == ["r0", "r1", "r2"]


def test_channel_quotas_alone_means_others_uncapped() -> None:
    """If only channel_quotas is set (no per_channel_max), other channels keep all items."""
    nba = [_post(id_=f"n{i}", channel="nba", score=5000 - i, age_hours=2) for i in range(10)]
    france = [_post(id_=f"f{i}", channel="france", score=600 - i, age_hours=2) for i in range(10)]

    top = rank(nba + france, max_items=20, channel_quotas={"nba": 3}, now=NOW)
    assert sum(1 for p in top if p.channel == "nba") == 3
    assert sum(1 for p in top if p.channel == "france") == 10
