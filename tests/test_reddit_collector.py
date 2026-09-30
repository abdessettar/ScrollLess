from __future__ import annotations

from typing import Any

import httpx
import pytest
from scrollless.collectors.reddit import RedditCollector
from scrollless.config import RedditConfig
from scrollless.exceptions import InvalidCredentialsError, MissingCredentialsError


@pytest.fixture(autouse=True)
def creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "linux:scrollless:test (by /u/tester)")


@pytest.fixture
def cfg() -> RedditConfig:
    return RedditConfig(
        subreddits=["dataengineering"],
        sort="hot",
        min_score=50,
        max_posts_per_sub=5,
        max_comments=3,
        request_delay_seconds=0.0,  # no real sleeping in tests
    )


def _child(
    *,
    id_: str = "abc",
    title: str = "Sample",
    score: int = 100,
    stickied: bool = False,
    is_self: bool = False,
    selftext: str = "",
    url: str = "https://example.com/article",
) -> dict[str, Any]:
    return {
        "kind": "t3",
        "data": {
            "id": id_,
            "title": title,
            "permalink": f"/r/dataengineering/comments/{id_}/",
            "url": url,
            "score": score,
            "num_comments": 10,
            "created_utc": 1_700_000_000,
            "selftext": selftext,
            "author": "alice",
            "stickied": stickied,
            "is_self": is_self,
        },
    }


def _listing(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"data": {"children": children}}


def _comment_response(comment_bodies: list[str]) -> list[dict[str, Any]]:
    """The /comments/{id} endpoint returns [post_listing, comments_listing]."""
    comments = [
        {"kind": "t1", "data": {"author": f"u{i}", "body": body, "score": 5}}
        for i, body in enumerate(comment_bodies)
    ]
    return [
        _listing([]),  # post (unused)
        _listing(comments),
    ]


_TOKEN_OK = {"access_token": "tok-123", "token_type": "bearer", "expires_in": 86400}


def _routing_handler(
    data_handler: Any, token_response: httpx.Response | None = None
) -> httpx.MockTransport:
    """MockTransport that serves the token exchange, delegating the rest."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/access_token":
            return token_response or httpx.Response(200, json=_TOKEN_OK)
        return data_handler(request)

    return httpx.MockTransport(handler)


def _client(transport: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=transport)


def test_collect_parses_listing_into_posts(cfg: RedditConfig) -> None:
    def data(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "bearer tok-123"
        assert request.url.host == "oauth.reddit.com"
        if request.url.path == "/r/dataengineering/hot":
            assert request.url.params["limit"] == "5"
            return httpx.Response(200, json=_listing([_child(id_="x", score=200)]))
        if request.url.path == "/r/dataengineering/comments/x":
            return httpx.Response(200, json=_comment_response(["first!", "second"]))
        return httpx.Response(404)

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    posts = collector.enrich(collector.collect())
    assert len(posts) == 1
    assert posts[0].id == "x"
    assert posts[0].source == "reddit"
    assert posts[0].channel == "dataengineering"
    assert posts[0].url == "https://www.reddit.com/r/dataengineering/comments/x/"
    assert posts[0].external_url == "https://example.com/article"
    assert [c.body for c in posts[0].comments] == ["first!", "second"]


def test_token_exchange_uses_basic_auth_and_client_credentials(cfg: RedditConfig) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/access_token":
            captured["host"] = request.url.host
            captured["auth"] = request.headers.get("authorization", "")
            captured["body"] = request.content.decode()
            captured["ua"] = request.headers.get("user-agent", "")
            return httpx.Response(200, json=_TOKEN_OK)
        return httpx.Response(200, json=_listing([]))

    RedditCollector(cfg, client=_client(httpx.MockTransport(handler))).collect()
    assert captured["host"] == "www.reddit.com"
    assert captured["auth"].startswith("Basic ")
    assert "grant_type=client_credentials" in captured["body"]
    assert captured["ua"] == "linux:scrollless:test (by /u/tester)"


def test_token_fetched_once_per_run() -> None:
    cfg = RedditConfig(
        subreddits=["a", "b", "c"], min_score=0, max_comments=0, request_delay_seconds=0.0
    )
    token_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        if request.url.path == "/api/v1/access_token":
            token_calls += 1
            return httpx.Response(200, json=_TOKEN_OK)
        return httpx.Response(200, json=_listing([]))

    RedditCollector(cfg, client=_client(httpx.MockTransport(handler))).collect()
    assert token_calls == 1


def test_missing_credentials_raise_before_any_request(
    cfg: RedditConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_SECRET")
    with pytest.raises(MissingCredentialsError, match="REDDIT_CLIENT_SECRET"):
        RedditCollector(cfg)


def test_rejected_credentials_raise_invalid_credentials(cfg: RedditConfig) -> None:
    transport = _routing_handler(
        lambda request: httpx.Response(404),
        token_response=httpx.Response(401, json={"message": "Unauthorized", "error": 401}),
    )
    with pytest.raises(InvalidCredentialsError, match="prefs/apps"):
        RedditCollector(cfg, client=_client(transport)).collect()


def test_token_response_without_access_token_is_rejected(cfg: RedditConfig) -> None:
    transport = _routing_handler(
        lambda request: httpx.Response(404),
        token_response=httpx.Response(200, json={"error": "invalid_grant"}),
    )
    with pytest.raises(InvalidCredentialsError, match=r"access_token"):
        RedditCollector(cfg, client=_client(transport)).collect()


def test_collect_filters_low_score_and_stickied(cfg: RedditConfig) -> None:
    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r/dataengineering/hot":
            return httpx.Response(
                200,
                json=_listing(
                    [
                        _child(id_="a", score=200),
                        _child(id_="b", score=10),  # below min_score
                        _child(id_="c", score=999, stickied=True),  # stickied
                    ]
                ),
            )
        return httpx.Response(200, json=_comment_response([]))

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    posts = collector.enrich(collector.collect())
    assert [p.id for p in posts] == ["a"]


def test_min_scores_overrides_default_per_subreddit() -> None:
    """Sub-specific threshold beats the global default."""
    cfg = RedditConfig(
        subreddits=["loud", "quiet"],
        sort="hot",
        min_score=100,
        min_scores={"loud": 5000, "quiet": 5},
        max_posts_per_sub=5,
        max_comments=0,
        request_delay_seconds=0.0,
    )

    def data(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/r/loud/hot":
            return httpx.Response(
                200,
                json=_listing(
                    [
                        _child(id_="big", score=10_000),
                        _child(id_="medium", score=500),  # below loud's 5000 floor
                    ]
                ),
            )
        if path == "/r/quiet/hot":
            return httpx.Response(
                200,
                json=_listing(
                    [
                        _child(id_="small", score=10),  # above quiet's 5 floor
                        _child(id_="tiny", score=2),  # below quiet's 5 floor
                    ]
                ),
            )
        return httpx.Response(404)

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    posts = collector.enrich(collector.collect())
    assert {p.id for p in posts} == {"big", "small"}


def test_subreddit_without_override_falls_back_to_default() -> None:
    """A sub not listed in min_scores uses the default min_score."""
    cfg = RedditConfig(
        subreddits=["nooverride"],
        sort="hot",
        min_score=100,
        min_scores={"loud": 5000},  # nooverride is not listed
        max_posts_per_sub=5,
        max_comments=0,
        request_delay_seconds=0.0,
    )

    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r/nooverride/hot":
            return httpx.Response(
                200,
                json=_listing(
                    [
                        _child(id_="a", score=150),
                        _child(id_="b", score=50),  # below default 100
                    ]
                ),
            )
        return httpx.Response(404)

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    posts = collector.enrich(collector.collect())
    assert [p.id for p in posts] == ["a"]


def test_threshold_for_helper() -> None:
    cfg = RedditConfig(
        subreddits=["a", "b"],
        min_score=10,
        min_scores={"a": 100},
    )
    assert cfg.threshold_for("a") == 100
    assert cfg.threshold_for("b") == 10  # falls back to default
    assert cfg.threshold_for("never-listed") == 10


def test_self_post_has_no_external_url(cfg: RedditConfig) -> None:
    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r/dataengineering/hot":
            return httpx.Response(
                200,
                json=_listing(
                    [_child(id_="s", is_self=True, selftext="discussion text", score=100)]
                ),
            )
        return httpx.Response(200, json=_comment_response([]))

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    posts = collector.enrich(collector.collect())
    assert posts[0].external_url is None
    assert posts[0].body == "discussion text"


def test_skips_sub_on_http_error(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"subreddits": ["bad", "good"]})

    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/r/bad/"):
            return httpx.Response(404)
        if request.url.path == "/r/good/hot":
            return httpx.Response(200, json=_listing([_child(id_="g", score=100)]))
        if "/comments/" in request.url.path:
            return httpx.Response(200, json=_comment_response([]))
        return httpx.Response(404)

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    posts = collector.enrich(collector.collect())
    assert [p.id for p in posts] == ["g"]


def test_max_comments_zero_skips_comment_fetch(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"max_comments": 0})
    calls: list[str] = []

    def data(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/r/dataengineering/hot":
            return httpx.Response(200, json=_listing([_child(id_="x", score=100)]))
        return httpx.Response(500)

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    collector.enrich(collector.collect())
    # Only the listing call should have happened (the token call is filtered out).
    assert calls == ["/r/dataengineering/hot"]


def test_throttle_called_between_requests(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"request_delay_seconds": 0.5})
    sleeps: list[float] = []

    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r/dataengineering/hot":
            return httpx.Response(200, json=_listing([_child(id_="x", score=100)]))
        return httpx.Response(200, json=_comment_response(["a"]))

    collector = RedditCollector(
        cfg,
        client=_client(_routing_handler(data)),
        sleep=sleeps.append,
    )
    collector.enrich(collector.collect())
    assert sleeps and all(s == 0.5 for s in sleeps)


def test_backs_off_when_rate_budget_nearly_spent(cfg: RedditConfig) -> None:
    """Reddit reports the remaining budget; we wait out the window near zero."""

    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r/dataengineering/hot":
            return httpx.Response(
                200,
                json=_listing([]),
                headers={"x-ratelimit-remaining": "2.0", "x-ratelimit-reset": "37"},
            )
        return httpx.Response(404)

    sleeps: list[float] = []
    RedditCollector(cfg, client=_client(_routing_handler(data)), sleep=sleeps.append).collect()
    assert 37.0 in sleeps


def test_healthy_rate_budget_does_not_back_off(cfg: RedditConfig) -> None:
    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r/dataengineering/hot":
            return httpx.Response(
                200,
                json=_listing([]),
                headers={"x-ratelimit-remaining": "94.0", "x-ratelimit-reset": "37"},
            )
        return httpx.Response(404)

    sleeps: list[float] = []
    RedditCollector(cfg, client=_client(_routing_handler(data)), sleep=sleeps.append).collect()
    assert 37.0 not in sleeps


def test_top_sort_passes_time_filter(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(update={"sort": "top", "time_filter": "day"})
    captured: dict[str, Any] = {}

    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/r/dataengineering/top":
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json=_listing([]))
        return httpx.Response(404)

    collector = RedditCollector(cfg, client=_client(_routing_handler(data)))
    collector.enrich(collector.collect())
    assert captured["params"]["t"] == "day"
    assert captured["params"]["raw_json"] == "1"


def test_bot_and_recurring_posts_are_skipped(cfg: RedditConfig) -> None:
    cfg = cfg.model_copy(
        update={
            "max_comments": 0,
            "skip_title_patterns": [r"^weekly\b"],
        }
    )

    def data(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/r/{'dataengineering'}/hot":
            children = [
                _child(id_="bot", score=500),
                _child(id_="recurring", title="Weekly hiring thread", score=500),
                _child(id_="real", title="A genuine post", score=500),
            ]
            children[0]["data"]["author"] = "AutoModerator"
            return httpx.Response(200, json=_listing(children))
        return httpx.Response(404)

    posts = RedditCollector(cfg, client=_client(_routing_handler(data))).collect()
    assert [p.id for p in posts] == ["real"]
