from __future__ import annotations

from datetime import UTC, datetime
from xml.sax.saxutils import escape

import httpx
import pytest
from scrollless.collectors.reddit_rss import RedditRSSCollector
from scrollless.config import RedditConfig
from scrollless.models import Post

SUB = "dataengineering"


@pytest.fixture
def cfg() -> RedditConfig:
    return RedditConfig(
        subreddits=[SUB],
        sort="top",
        time_filter="day",
        max_posts_per_sub=10,
        max_comments=5,
        request_delay_seconds=0.0,  # no real sleeping in tests
    )


def _feed(entries: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<title>r/{SUB}</title>" + "".join(entries) + "</feed>"
    )


def _post_entry(
    *,
    id_: str = "1abc",
    title: str = "Duckdb moving away from postgres parser",
    author: str = "/u/Gators1992",
    body: str | None = None,
    external: str | None = None,
    published: str = "2026-08-20T15:18:23+00:00",
) -> str:
    """An Atom entry shaped like Reddit's real listing feed."""
    permalink = f"https://www.reddit.com/r/{SUB}/comments/{id_}/slug/"
    inner = f'<!-- SC_OFF --><div class="md"><p>{body}</p></div><!-- SC_ON -->' if body else ""
    # Reddit's footer: [link] points off-site for link posts, home for self-posts.
    content = (
        f'{inner} &#32; submitted by <a href="https://www.reddit.com/user/x">{author}</a>'
        f'<span><a href="{external or permalink}">[link]</a></span>'
        f'<span><a href="{permalink}">[comments]</a></span>'
    )
    return (
        f"<entry><author><name>{author}</name></author>"
        f'<category term="{SUB}" label="r/{SUB}"/>'
        f'<content type="html">{escape(content)}</content>'
        f"<id>t3_{id_}</id>"
        f'<link href="{permalink}" />'
        f"<updated>{published}</updated><published>{published}</published>"
        f"<title>{escape(title)}</title></entry>"
    )


def _comment_entry(*, id_: str = "c1", author: str = "/u/bob", body: str = "Nice") -> str:
    content = f'<div class="md"><p>{body}</p></div>'
    return (
        f"<entry><author><name>{author}</name></author>"
        f'<content type="html">{escape(content)}</content>'
        f"<id>t1_{id_}</id>"
        f'<link href="https://www.reddit.com/r/{SUB}/comments/1abc/slug/{id_}/" />'
        f"<updated>2026-08-20T16:00:00+00:00</updated>"
        f"<title>comment by {author}</title></entry>"
    )


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_collect_parses_entries_into_posts(cfg: RedditConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/r/{SUB}/top.rss":
            assert request.url.params["t"] == "day"
            return httpx.Response(200, text=_feed([_post_entry(body="This is big since …")]))
        return httpx.Response(200, text=_feed([_comment_entry(body="Finally JSON_TABLE!")]))

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    posts = collector.enrich(collector.collect())
    assert len(posts) == 1
    p = posts[0]
    assert p.id == "1abc"  # t3_ prefix stripped
    assert p.source == "reddit"
    assert p.channel == SUB
    assert p.title == "Duckdb moving away from postgres parser"
    assert p.url == f"https://www.reddit.com/r/{SUB}/comments/1abc/slug/"
    assert p.author == "Gators1992"  # /u/ prefix stripped, not char-stripped
    assert p.body == "This is big since …"
    assert p.created_utc.year == 2026
    assert [c.body for c in p.comments] == ["Finally JSON_TABLE!"]


def test_author_prefix_strip_keeps_leading_u(cfg: RedditConfig) -> None:
    """Regression: lstrip('/u/') would eat the 'u' of a name like /u/user123."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(200, text=_feed([_post_entry(author="/u/user123")]))
        return httpx.Response(200, text=_feed([_comment_entry()]))

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert posts[0].author == "user123"


def test_score_is_synthetic_and_flagged(cfg: RedditConfig) -> None:
    """No score in the feed: position becomes a descending stand-in."""
    entries = [_post_entry(id_=f"p{i}") for i in range(3)]
    cfg = cfg.model_copy(update={"max_comments": 0})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_feed(entries))

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    scores = [p.score for p in posts]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 3
    assert all(p.score_is_estimated for p in posts)
    assert all(p.num_comments == 0 for p in posts)


def test_link_post_exposes_external_url(cfg: RedditConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(
                200, text=_feed([_post_entry(external="https://duckdb.org/2026/08/20/peg")])
            )
        return httpx.Response(200, text=_feed([_comment_entry()]))

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    posts = collector.enrich(collector.collect())
    assert posts[0].external_url == "https://duckdb.org/2026/08/20/peg"


def test_self_post_has_no_external_url(cfg: RedditConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(200, text=_feed([_post_entry(body="text only")]))
        return httpx.Response(200, text=_feed([_comment_entry()]))

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    posts = collector.enrich(collector.collect())
    assert posts[0].external_url is None


def test_collect_never_fetches_comments(cfg: RedditConfig) -> None:
    """The whole point of the split: listings are cheap, threads are not.

    Fetching comments during collection meant paying a 60-second window for
    posts that dedup and the quotas were about to throw away.
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text=_feed([_post_entry(id_=f"p{i}") for i in range(6)]))

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert len(posts) == 6  # every listing entry, not a pre-trimmed slice
    assert calls == [f"/r/{SUB}/top.rss"]
    assert all(not p.comments for p in posts)


def test_enrich_fetches_one_feed_per_post_it_is_given(cfg: RedditConfig) -> None:
    """Only the posts that survived ranking reach enrich, and each costs one."""
    comment_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(200, text=_feed([_post_entry(id_=f"p{i}") for i in range(6)]))
        comment_calls.append(request.url.path)
        return httpx.Response(200, text=_feed([_comment_entry()]))

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    survivors = collector.collect()[:2]  # what rank() would have handed back
    enriched = collector.enrich(survivors)

    assert [p.id for p in enriched] == ["p0", "p1"]
    assert len(comment_calls) == 2  # not six
    assert all(p.comments for p in enriched)


def test_enrich_leaves_other_sources_untouched(cfg: RedditConfig) -> None:
    """A mixed digest passes through: only this collector's posts are its work."""
    hn_post = Post(
        id="42",
        source="hn",
        channel="hn",
        title="An HN story",
        url="https://news.ycombinator.com/item?id=42",
        external_url=None,
        score=500,
        num_comments=100,
        created_utc=datetime.now(UTC),
    )
    comment_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(200, text=_feed([_post_entry(id_="r1")]))
        comment_calls.append(request.url.path)
        return httpx.Response(200, text=_feed([_comment_entry()]))

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    reddit_posts = collector.collect()
    out = collector.enrich([hn_post, *reddit_posts])

    assert [p.id for p in out] == ["42", "r1"]  # order preserved, nothing dropped
    assert out[0].comments == []  # the HN post was not touched
    assert len(comment_calls) == 1


def test_max_comments_zero_skips_comment_fetch(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"max_comments": 0})
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text=_feed([_post_entry(id_="a"), _post_entry(id_="b")]))

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    posts = collector.enrich(collector.collect())
    assert [p.id for p in posts] == ["a", "b"]  # nothing dropped when comments are off
    assert calls == [f"/r/{SUB}/top.rss"]


def test_post_entry_in_comment_feed_is_ignored(cfg: RedditConfig) -> None:
    """A comment feed leads with the post itself; only t1_ entries are comments."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(200, text=_feed([_post_entry()]))
        return httpx.Response(
            200, text=_feed([_post_entry(), _comment_entry(body="an actual comment")])
        )

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    posts = collector.enrich(collector.collect())
    assert [c.body for c in posts[0].comments] == ["an actual comment"]


def test_rate_limited_request_waits_the_window_then_retries(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"max_comments": 0})
    sleeps: list[float] = []
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"x-ratelimit-reset": "30"}, text="")
        return httpx.Response(200, text=_feed([_post_entry()]))

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=sleeps.append).collect()
    assert len(posts) == 1
    assert sleeps == [31.0]  # Reddit's own reset hint, plus a one-second pad


def test_spent_budget_paces_before_the_next_request(cfg: RedditConfig) -> None:
    """Reddit reports remaining=0 on success too: wait rather than earn a 429."""
    cfg = cfg.model_copy(update={"max_comments": 0, "subreddits": [SUB, "startups"]})
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-ratelimit-remaining": "0.0", "x-ratelimit-reset": "55"},
            text=_feed([_post_entry()]),
        )

    RedditRSSCollector(cfg, client=_client(handler), sleep=sleeps.append).collect()
    # One wait, paid before the second sub's request, not a trailing sleep
    # after the last one, which would add a dead minute to every run.
    assert sleeps == [56.0]


def test_persistent_rate_limit_gives_up_and_skips_the_sub(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"max_comments": 0, "subreddits": [SUB, "startups"]})
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if SUB in request.url.path:
            return httpx.Response(429, headers={"x-ratelimit-reset": "10"}, text="")
        return httpx.Response(200, text=_feed([_post_entry(id_="ok")]))

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=sleeps.append).collect()
    assert [p.id for p in posts] == ["ok"]  # the healthy sub still lands
    assert len(sleeps) == 3  # bounded retries, not an infinite wait


def test_skips_sub_on_http_error(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"max_comments": 0, "subreddits": ["gone", "good"]})

    def handler(request: httpx.Request) -> httpx.Response:
        if "/r/gone/" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, text=_feed([_post_entry(id_="g")]))

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert [p.id for p in posts] == ["g"]


def test_unparseable_feed_yields_no_posts(cfg: RedditConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not a feed")

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert posts == []


def test_hot_sort_omits_time_filter(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"sort": "hot", "max_comments": 0})
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, text=_feed([]))

    RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert captured["path"] == f"/r/{SUB}/hot.rss"
    assert "t" not in captured["params"]  # time_filter only applies to top


def test_post_with_no_comments_and_no_body_is_dropped(cfg: RedditConfig) -> None:
    """A top-of-day link post nobody replied to has no thread to summarize."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(
                200,
                text=_feed(
                    [
                        _post_entry(id_="silent", external="https://example.com/x"),
                        _post_entry(id_="discussed", external="https://example.com/y"),
                    ]
                ),
            )
        if "silent" in request.url.path:
            return httpx.Response(200, text=_feed([_post_entry(id_="silent")]))  # post only
        return httpx.Response(200, text=_feed([_comment_entry(body="worth reading because …")]))

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    posts = collector.enrich(collector.collect())
    assert [p.id for p in posts] == ["discussed"]


def test_long_self_post_survives_without_comments(cfg: RedditConfig) -> None:
    """No replies is fine when the post itself is the content."""
    essay = "word " * 60  # comfortably over the body threshold

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("top.rss"):
            return httpx.Response(200, text=_feed([_post_entry(id_="essay", body=essay)]))
        return httpx.Response(200, text=_feed([_post_entry(id_="essay")]))  # no comments

    collector = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None)
    posts = collector.enrich(collector.collect())
    assert [p.id for p in posts] == ["essay"]


def test_bot_and_recurring_posts_are_skipped(cfg: RedditConfig) -> None:
    """The `stickied` flag vanished with the .json endpoints; this replaces it."""
    cfg = cfg.model_copy(
        update={
            "max_comments": 0,
            "skip_title_patterns": [r"^(daily|weekly|monthly)\b.*\b(thread|discussion)\b"],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_feed(
                [
                    _post_entry(
                        id_="mega",
                        author="/u/AutoModerator",
                        title="Monthly General Discussion - Aug 2026",
                    ),
                    _post_entry(id_="recurring", author="/u/mod", title="Daily Discussion Thread"),
                    _post_entry(id_="real", author="/u/alice", title="A genuine post"),
                ]
            ),
        )

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert [p.id for p in posts] == ["real"]


def test_skipped_posts_do_not_consume_the_top_score(cfg: RedditConfig) -> None:
    """Filtering happens before scoring, so a megathread can't demote the rest."""
    cfg = cfg.model_copy(update={"max_comments": 0})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_feed(
                [
                    _post_entry(id_="bot", author="/u/AutoModerator", title="Weekly thread"),
                    _post_entry(id_="first", author="/u/alice", title="Actually the top post"),
                    _post_entry(id_="second", author="/u/bob", title="Runner-up"),
                ]
            ),
        )

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert [p.id for p in posts] == ["first", "second"]
    assert posts[0].score > posts[1].score
    assert posts[0].score == 2  # scored against the kept entries, not all three


def test_clip_hosts_are_skipped(cfg: RedditConfig) -> None:
    """Sports clips are titled by a convention half of them ignore, but they
    all link to the same handful of video hosts."""
    cfg = cfg.model_copy(update={"max_comments": 0, "skip_domains": ["streamable.com"]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_feed(
                [
                    _post_entry(
                        id_="clip",
                        title="VJ Edgecombe dunks on a kid",
                        external="https://streamable.com/abc123",
                    ),
                    _post_entry(
                        id_="news",
                        title="Klay Thompson gave back $9.8M",
                        external="https://www.espn.com/story",
                    ),
                    _post_entry(id_="selfpost", title="Discussion of the trade", body="text"),
                ]
            ),
        )

    posts = RedditRSSCollector(cfg, client=_client(handler), sleep=lambda _s: None).collect()
    assert [p.id for p in posts] == ["news", "selfpost"]
