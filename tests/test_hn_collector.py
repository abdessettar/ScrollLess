from __future__ import annotations

from typing import Any

import httpx
import pytest
from scrollless.collectors.hn import HackerNewsCollector
from scrollless.config import HackerNewsConfig
from scrollless.exceptions import CollectorError


def _algolia_response(hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {"hits": hits, "nbHits": len(hits)}


def _hit(
    *,
    id_: str = "1",
    title: str = "Sample",
    points: int = 100,
    url: str | None = "https://example.com",
    comments: int = 10,
) -> dict[str, Any]:
    return {
        "objectID": id_,
        "title": title,
        "url": url,
        "points": points,
        "num_comments": comments,
        "author": "pg",
        "created_at_i": 1_700_000_000,
        "story_text": None,
    }


def _comment(
    *, id_: int, author: str = "alice", text: str = "a take", children: list[Any] | None = None
) -> dict[str, Any]:
    """A node as the Algolia items endpoint returns it."""
    return {
        "id": id_,
        "type": "comment",
        "author": author,
        "text": text,
        "children": children or [],
    }


def _algolia_item(story_id: int, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": story_id, "type": "story", "title": "Sample", "children": children}


def _is_search(request: httpx.Request) -> bool:
    return request.url.path == "/api/v1/search"


def _is_tree(request: httpx.Request) -> bool:
    return request.url.path.startswith("/api/v1/items/")


def _is_ranking(request: httpx.Request) -> bool:
    return request.url.host == "hacker-news.firebaseio.com"


@pytest.fixture
def hn_cfg() -> HackerNewsConfig:
    return HackerNewsConfig(
        enabled=True,
        min_score=50,
        max_posts=10,
        tags=["story", "ask_hn"],
        comment_depth=2,
        window_hours=0,  # disable date filter in most tests for simpler assertions
    )


def _client_with(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_collect_then_enrich_returns_posts_with_comments(hn_cfg: HackerNewsConfig) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if _is_search(request):
            assert request.url.params["tags"] == "(story,ask_hn)"
            assert request.url.params["numericFilters"] == "points>=50"
            return httpx.Response(200, json=_algolia_response([_hit(id_="42")]))
        if _is_tree(request):
            return httpx.Response(
                200,
                json=_algolia_item(
                    42,
                    [
                        _comment(
                            id_=100,
                            author="alice",
                            text="top",
                            children=[_comment(id_=200, author="carol", text="nested")],
                        ),
                        _comment(id_=101, author="bob", text="second"),
                    ],
                ),
            )
        if _is_ranking(request):
            return httpx.Response(200, json={"id": 42, "kids": [100, 101]})
        return httpx.Response(404)

    client = _client_with(httpx.MockTransport(handler))
    collector = HackerNewsCollector(hn_cfg, client=client)
    posts = collector.enrich(collector.collect())

    assert len(posts) == 1
    post = posts[0]
    assert post.id == "42"
    assert post.source == "hn"
    assert post.url == "https://news.ycombinator.com/item?id=42"
    assert post.external_url == "https://example.com"
    assert post.score == 100
    assert post.author == "pg"

    # depth 2: alice (depth 0) + carol nested under alice (depth 1) + bob (depth 0)
    assert [c.author for c in post.comments] == ["alice", "carol", "bob"]
    assert [c.depth for c in post.comments] == [0, 1, 0]
    # One search, then exactly two requests for the story's comments.
    assert len(calls) == 3


def test_comments_follow_hn_ranking_not_the_clock(hn_cfg: HackerNewsConfig) -> None:
    """The reason two APIs are used at all.

    Algolia returns a story's children in chronological order, so its first
    comments are the earliest ones ('first!' noise). Firebase's `kids` array
    is HN's own ranked order but costs a request per comment. We take the
    ranking from one and the bodies from the other.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_search(request):
            return httpx.Response(200, json=_algolia_response([_hit(id_="7")]))
        if _is_tree(request):
            return httpx.Response(
                200,
                json=_algolia_item(
                    7,
                    [  # chronological: earliest first
                        _comment(id_=1, author="earliest", text="first!"),
                        _comment(id_=2, author="middle", text="a thought"),
                        _comment(id_=3, author="latest", text="the best analysis"),
                    ],
                ),
            )
        if _is_ranking(request):  # HN ranks the last one top
            return httpx.Response(200, json={"id": 7, "kids": [3, 1, 2]})
        return httpx.Response(404)

    collector = HackerNewsCollector(hn_cfg, client=_client_with(httpx.MockTransport(handler)))
    posts = collector.enrich(collector.collect())
    assert [c.author for c in posts[0].comments] == ["latest", "earliest", "middle"]


def test_falls_back_to_chronological_when_the_ranking_is_unavailable(
    hn_cfg: HackerNewsConfig,
) -> None:
    """A Firebase outage should cost ordering, not the whole thread."""

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_search(request):
            return httpx.Response(200, json=_algolia_response([_hit(id_="7")]))
        if _is_tree(request):
            return httpx.Response(
                200,
                json=_algolia_item(7, [_comment(id_=1, author="a"), _comment(id_=2, author="b")]),
            )
        return httpx.Response(503)  # Firebase down

    collector = HackerNewsCollector(hn_cfg, client=_client_with(httpx.MockTransport(handler)))
    posts = collector.enrich(collector.collect())
    assert [c.author for c in posts[0].comments] == ["a", "b"]


def test_comment_html_is_stripped(hn_cfg: HackerNewsConfig) -> None:
    """HN serves markup; the model shouldn't have to read tags."""

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_search(request):
            hit = _hit(id_="9")
            hit["story_text"] = "<p>Story <i>body</i> here.</p>"
            return httpx.Response(200, json=_algolia_response([hit]))
        if _is_tree(request):
            return httpx.Response(
                200,
                json=_algolia_item(
                    9,
                    [
                        _comment(
                            id_=1,
                            text='<p>See <a href="https://x.test">this</a> &#x27;quote&#x27;.</p>',
                        )
                    ],
                ),
            )
        if _is_ranking(request):
            return httpx.Response(200, json={"id": 9, "kids": [1]})
        return httpx.Response(404)

    collector = HackerNewsCollector(hn_cfg, client=_client_with(httpx.MockTransport(handler)))
    posts = collector.enrich(collector.collect())
    assert posts[0].comments[0].body == "See this 'quote'."
    assert posts[0].body == "Story body here."


def test_collect_skips_disabled(hn_cfg: HackerNewsConfig) -> None:
    cfg = hn_cfg.model_copy(update={"enabled": False})
    client = _client_with(httpx.MockTransport(lambda _r: httpx.Response(500)))
    assert HackerNewsCollector(cfg, client=client).collect() == []


def test_search_http_error_raises_collector_error(hn_cfg: HackerNewsConfig) -> None:
    client = _client_with(httpx.MockTransport(lambda _r: httpx.Response(503)))
    with pytest.raises(CollectorError, match="hn:search"):
        HackerNewsCollector(hn_cfg, client=client).collect()


def test_deleted_comments_are_filtered(hn_cfg: HackerNewsConfig) -> None:
    """Algolia represents deleted and dead comments with a null body."""

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_search(request):
            return httpx.Response(200, json=_algolia_response([_hit(id_="1")]))
        if _is_tree(request):
            return httpx.Response(
                200,
                json=_algolia_item(
                    1,
                    [
                        {"id": 10, "type": "comment", "author": None, "text": None},
                        {"id": 11, "type": "comment", "author": "gone", "text": ""},
                        _comment(id_=12, author="ok", text="fine"),
                    ],
                ),
            )
        if _is_ranking(request):
            return httpx.Response(200, json={"id": 1, "kids": [10, 11, 12]})
        return httpx.Response(404)

    collector = HackerNewsCollector(hn_cfg, client=_client_with(httpx.MockTransport(handler)))
    posts = collector.enrich(collector.collect())
    assert [c.author for c in posts[0].comments] == ["ok"]


def test_window_hours_adds_created_at_filter(hn_cfg: HackerNewsConfig) -> None:
    cfg = hn_cfg.model_copy(update={"window_hours": 24})
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hn.algolia.com":
            captured["filters"] = request.url.params["numericFilters"]
            return httpx.Response(200, json=_algolia_response([]))
        return httpx.Response(404)

    client = _client_with(httpx.MockTransport(handler))
    HackerNewsCollector(cfg, client=client).collect()

    assert "points>=50" in captured["filters"]
    assert "created_at_i>" in captured["filters"]


def test_window_hours_zero_omits_created_at_filter(hn_cfg: HackerNewsConfig) -> None:
    """window_hours=0 means 'no date constraint', useful for back-filling."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hn.algolia.com":
            captured["filters"] = request.url.params["numericFilters"]
            return httpx.Response(200, json=_algolia_response([]))
        return httpx.Response(404)

    client = _client_with(httpx.MockTransport(handler))
    HackerNewsCollector(hn_cfg, client=client).collect()

    assert captured["filters"] == "points>=50"


def test_single_tag_not_wrapped_in_parens(hn_cfg: HackerNewsConfig) -> None:
    cfg = hn_cfg.model_copy(update={"tags": ["story"]})
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hn.algolia.com":
            captured["tags"] = request.url.params["tags"]
            return httpx.Response(200, json=_algolia_response([]))
        return httpx.Response(404)

    client = _client_with(httpx.MockTransport(handler))
    HackerNewsCollector(cfg, client=client).collect()
    assert captured["tags"] == "story"
