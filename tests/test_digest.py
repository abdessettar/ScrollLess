from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from scrollless.digest import (
    RenderedDigest,
    build_digest_items,
    build_intro,
    format_subject,
    render_digest,
)
from scrollless.exceptions import SummarizerError
from scrollless.models import DigestItem, ItemSummary, Post

NOW = datetime(2026, 5, 17, 9, 0, 0, tzinfo=UTC)


def _post(
    *,
    id_: str = "a",
    source: str = "reddit",
    channel: str = "dataengineering",
    title: str = "Postgres 17 released",
    score: int = 420,
    external_url: str | None = "https://postgresql.org/release/17",
    score_is_estimated: bool = False,
) -> Post:
    return Post(
        id=id_,
        source=source,  # type: ignore[arg-type]
        channel=channel,
        title=title,
        url=f"https://reddit.com/r/{channel}/comments/{id_}/",
        external_url=external_url,
        score=score,
        num_comments=80,
        created_utc=NOW,
        score_is_estimated=score_is_estimated,
    )


def _summary(**kwargs: object) -> ItemSummary:
    base: dict[str, object] = {
        "gist": "A new release landed.",
        "points": ["One maintainer explains the rewrite", "Several ask about dialect drift"],
    }
    base.update(kwargs)
    return ItemSummary(**base)  # type: ignore[arg-type]


def _item(summary: ItemSummary | None = None, **post_kwargs: object) -> DigestItem:
    return DigestItem(
        post=_post(**post_kwargs),  # type: ignore[arg-type]
        summary=summary or _summary(),
    )


# --------------------------------------------------------------------------
# item building
# --------------------------------------------------------------------------


def test_build_digest_items_calls_summarizer_per_post() -> None:
    summarizer = MagicMock()
    summarizer.summarize.side_effect = lambda p: ItemSummary(gist=f"summary for {p.id}")

    items = build_digest_items([_post(id_="a"), _post(id_="b")], summarizer)

    assert [i.summary.gist for i in items] == ["summary for a", "summary for b"]
    assert summarizer.summarize.call_count == 2


def test_build_digest_items_skips_failures() -> None:
    summarizer = MagicMock()
    summarizer.summarize.side_effect = [
        ItemSummary(gist="ok"),
        SummarizerError("deepseek", RuntimeError("rate limit")),
        ItemSummary(gist="ok too"),
    ]

    items = build_digest_items([_post(id_="a"), _post(id_="b"), _post(id_="c")], summarizer)
    assert [i.post.id for i in items] == ["a", "c"]


def test_build_intro_returns_the_synthesis() -> None:
    summarizer = MagicMock()
    summarizer.synthesize.return_value = "Two themes today."
    assert build_intro([_item()], summarizer) == "Two themes today."


def test_build_intro_failure_does_not_cost_the_digest() -> None:
    summarizer = MagicMock()
    summarizer.synthesize.side_effect = SummarizerError("deepseek", RuntimeError("boom"))
    assert build_intro([_item()], summarizer) == ""


def test_build_intro_skips_the_call_when_empty() -> None:
    summarizer = MagicMock()
    assert build_intro([], summarizer) == ""
    summarizer.synthesize.assert_not_called()


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_render_digest_returns_both_formats() -> None:
    out = render_digest([_item()], subject="Test Digest", now=NOW)
    assert isinstance(out, RenderedDigest)
    assert out.item_count == 1
    assert "Postgres 17 released" in out.html
    assert "Postgres 17 released" in out.text
    assert "<!doctype html>" in out.html
    assert "2026-05-17 09:00 UTC" in out.html
    assert "2026-05-17 09:00 UTC" in out.text


def test_render_digest_escapes_html_in_titles() -> None:
    item = _item(title="<script>alert(1)</script>")
    out = render_digest([item], subject="Test", now=NOW)
    assert "<script>alert(1)</script>" not in out.html
    assert "&lt;script&gt;" in out.html
    # Plain text shows the raw string (no escaping needed there).
    assert "<script>alert(1)</script>" in out.text


def test_item_carries_both_the_article_and_the_discussion() -> None:
    out = render_digest([_item()], subject="Test", now=NOW)
    assert "https://postgresql.org/release/17" in out.html  # title links to the article
    assert "https://reddit.com/r/dataengineering/comments/a/" in out.html  # and the thread
    assert "postgresql.org" in out.html  # domain shown as the byline
    assert "https://postgresql.org/release/17" in out.text
    assert "https://reddit.com/r/dataengineering/comments/a/" in out.text


def test_self_post_links_the_discussion_only() -> None:
    out = render_digest([_item(external_url=None)], subject="Test", now=NOW)
    assert "https://reddit.com/r/dataengineering/comments/a/" in out.html
    assert "discussion on r/dataengineering" in out.html


def test_summary_sections_render_and_empty_ones_vanish() -> None:
    full = render_digest(
        [_item(summary=_summary(dissent="One reader objects.", takeaway="Try it on 2.0."))],
        subject="T",
        now=NOW,
    )
    assert "One reader objects." in full.html
    assert "Try it on 2.0." in full.html
    assert "One maintainer explains the rewrite" in full.html

    bare = render_digest([_item(summary=ItemSummary(gist="Just a gist."))], subject="T", now=NOW)
    assert "Just a gist." in bare.html
    assert "<ul" not in bare.html  # no empty bullet list
    assert "border-left:3px solid #d97706" not in bare.html  # no empty dissent block


def test_unparseable_summary_still_renders() -> None:
    out = render_digest(
        [_item(summary=ItemSummary(fallback_text="- raw model text"))], subject="T", now=NOW
    )
    assert "- raw model text" in out.html
    assert "- raw model text" in out.text


def test_intro_renders_when_present() -> None:
    out = render_digest([_item()], subject="T", intro="Two themes today.", now=NOW)
    assert "Two themes today." in out.html
    assert "Two themes today." in out.text


def test_render_digest_groups_by_reader_facing_labels() -> None:
    items = [
        _item(id_="a", source="reddit", channel="dataengineering"),
        _item(id_="b", source="hn", channel="hn"),
        _item(id_="c", source="reddit", channel="france"),
        _item(id_="d", source="reddit", channel="dataengineering"),
    ]
    out = render_digest(items, subject="Test", group_by_source=True, now=NOW)
    assert "r/dataengineering" in out.html
    assert "r/france" in out.html
    assert "Hacker News" in out.html
    assert "reddit / dataengineering" not in out.html  # not the internal form


def test_groups_appear_in_rank_order_not_alphabetically() -> None:
    """The strongest channel of the day should lead the email."""
    items = [
        _item(id_="a", source="hn", channel="hn"),
        _item(id_="b", source="reddit", channel="alpha"),
    ]
    out = render_digest(items, subject="Test", group_by_source=True, now=NOW)
    assert out.html.index("Hacker News") < out.html.index("r/alpha")


def test_render_digest_flat_mode_omits_section_headers() -> None:
    items = [
        _item(id_="a", source="reddit", channel="dataengineering"),
        _item(id_="b", source="hn", channel="hn"),
    ]
    out = render_digest(items, subject="Test", group_by_source=False, now=NOW)
    # The labels only exist as section headers; permalinks still contain "r/…".
    assert 'class="section-label"' not in out.html
    assert "Hacker News" not in out.html


def test_empty_digest_says_so_in_one_line() -> None:
    out = render_digest([], subject="Empty", now=NOW)
    assert out.item_count == 0
    assert "Nothing new today" in out.html
    assert "Nothing new today" in out.text
    assert "0 items" in out.html


def test_real_metrics_are_rendered() -> None:
    out = render_digest([_item(score=420)], subject="S", now=NOW)
    assert "420 points" in out.html
    assert "80 comments" in out.html
    assert "420 points" in out.text


def test_estimated_scores_are_never_printed() -> None:
    """RSS-sourced posts carry a position-derived score, so showing it would mislead."""
    out = render_digest([_item(score=7, score_is_estimated=True)], subject="S", now=NOW)
    assert "7 points" not in out.html
    assert "80 comments" not in out.html
    assert "80 comments" not in out.text


# --------------------------------------------------------------------------
# subject line
# --------------------------------------------------------------------------


def test_subject_carries_date_count_and_lead() -> None:
    subject = format_subject(
        "ScrollLess Morning · {date} · {count} items · {lead}",
        count=12,
        lead="Duckdb moving away from postgres parser",
        now=NOW,
    )
    assert subject == (
        "ScrollLess Morning · 2026-05-17 · 12 items · Duckdb moving away from postgres parser"
    )


def test_subject_trims_a_long_lead() -> None:
    subject = format_subject("{lead}", count=1, lead="x" * 200, now=NOW)
    assert subject.endswith("…")
    assert len(subject) <= 61


def test_subject_says_nothing_new_when_empty() -> None:
    assert format_subject("{count} · {lead}", count=0, lead="", now=NOW) == "0 · nothing new"


def test_subject_without_placeholders_still_works() -> None:
    assert format_subject("My Digest", count=3, lead="x", now=NOW) == "My Digest"


def test_subject_with_unknown_placeholder_is_left_alone() -> None:
    assert format_subject("{nope}", count=1, lead="x", now=NOW) == "{nope}"


def test_format_subject_substitutes_date() -> None:
    assert format_subject("Digest {date}", count=1, lead="x", now=NOW) == "Digest 2026-05-17"
