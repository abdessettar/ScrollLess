from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from scrollless import main as main_module
from scrollless.exceptions import EmailDeliveryError, MissingCredentialsError, ScrollLessError
from scrollless.logging_config import configure_logging
from scrollless.models import ItemSummary, Post


def _post(id_: str, score: int = 100) -> Post:
    return Post(
        id=id_,
        source="reddit",
        channel="ch",
        title=f"title-{id_}",
        url=f"https://reddit/{id_}",
        external_url=None,
        score=score,
        num_comments=0,
        created_utc=datetime.now(UTC),
    )


def _fake_collectors(posts: list[Post], *, collect_error: Exception | None = None) -> Any:
    """Stand in for _open_collectors with a single scripted collector."""
    collector = MagicMock()
    if collect_error is not None:
        collector.collect.side_effect = collect_error
    else:
        collector.collect.return_value = posts
    collector.enrich.side_effect = lambda ps: ps

    @contextmanager
    def opener(cfg: object, source: str) -> Iterator[list[MagicMock]]:
        yield [collector]

    opener.collector = collector  # type: ignore[attr-defined]
    return opener


def test_configure_logging_sets_level() -> None:
    configure_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG
    configure_logging(verbose=False)
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING


def test_main_full_pipeline_with_summarization_and_send(config_file: Path, tmp_path: Path) -> None:
    db = tmp_path / "seen.sqlite"
    out_html = tmp_path / "digest.html"

    summarizer = MagicMock()
    summarizer.summarize.return_value = ItemSummary(gist="one", points=["two"])
    summarizer.synthesize.return_value = "today in one line"
    sender = MagicMock()

    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a"), _post("b")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender", return_value=sender) as MockSender,
    ):
        rc = main_module.main(
            ["--config", str(config_file), "--db", str(db), "--output", str(out_html)]
        )

    assert rc == 0
    assert summarizer.summarize.call_count == 2
    assert out_html.exists()
    assert "<!doctype html>" in out_html.read_text()
    MockSender.assert_called_once()
    sender.send.assert_called_once()


def test_main_no_send_skips_email_delivery_and_does_not_mark_seen(
    config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--no-send is a preview mode: produces output but does NOT mark posts seen."""
    db = tmp_path / "seen.sqlite"
    summarizer = MagicMock()
    summarizer.summarize.return_value = ItemSummary(gist="summary")
    summarizer.synthesize.return_value = "intro"

    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender") as MockSender,
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db), "--no-send"])

    assert rc == 0
    MockSender.assert_not_called()
    assert "title-a" in capsys.readouterr().out

    # Second run should still see the same post as new.
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender"),
    ):
        main_module.main(["--config", str(config_file), "--db", str(db), "--no-send"])
    # Post appeared in both runs' output.
    assert capsys.readouterr().out.count("title-a") >= 1


def test_email_failure_does_not_mark_posts_as_seen(config_file: Path, tmp_path: Path) -> None:
    """Regression guard: a failed SMTP send must not burn posts from future digests."""
    db = tmp_path / "seen.sqlite"
    summarizer = MagicMock()
    summarizer.summarize.return_value = ItemSummary(gist="summary")
    summarizer.synthesize.return_value = "intro"
    sender = MagicMock()
    sender.send.side_effect = EmailDeliveryError(OSError("auth failed"))

    # First run: email fails, posts should NOT be marked.
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a"), _post("b")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender", return_value=sender),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db)])
    assert rc == 1

    # Second run: same posts, email succeeds, so both posts should reach the summarizer.
    sender2 = MagicMock()
    summarizer2 = MagicMock()
    summarizer2.summarize.return_value = ItemSummary(gist="summary")
    summarizer2.synthesize.return_value = "intro"
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a"), _post("b")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer2),
        patch("scrollless.main.EmailSender", return_value=sender2),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db)])
    assert rc == 0
    assert summarizer2.summarize.call_count == 2  # both posts re-processed


def test_main_returns_two_on_missing_smtp_credentials(config_file: Path, tmp_path: Path) -> None:
    db = tmp_path / "seen.sqlite"
    summarizer = MagicMock()
    summarizer.summarize.return_value = ItemSummary(gist="summary")
    summarizer.synthesize.return_value = "intro"

    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch(
            "scrollless.main.EmailSender",
            side_effect=MissingCredentialsError("SMTP_PASSWORD", "smtp"),
        ),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db)])
    assert rc == 2


def test_main_returns_one_on_email_delivery_failure(config_file: Path, tmp_path: Path) -> None:
    db = tmp_path / "seen.sqlite"
    summarizer = MagicMock()
    summarizer.summarize.return_value = ItemSummary(gist="summary")
    summarizer.synthesize.return_value = "intro"
    sender = MagicMock()
    sender.send.side_effect = EmailDeliveryError(OSError("network down"))

    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender", return_value=sender),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db)])
    assert rc == 1


def test_main_dry_run_skips_summarizer_and_persistence(
    config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "seen.sqlite"

    summarizer = MagicMock()
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer) as build,
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db), "--dry-run"])

    assert rc == 0
    summarizer.summarize.assert_not_called()
    build.assert_not_called()
    # Same post should still be considered new on the next run.
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
    ):
        main_module.main(["--config", str(config_file), "--db", str(db), "--dry-run"])
    out = capsys.readouterr().out
    # The preview line for the post should appear twice (once per run).
    assert out.count("title-a") == 2


def test_main_no_posts_after_dedup_sends_the_quiet_day_note(
    config_file: Path, tmp_path: Path
) -> None:
    """Silence is ambiguous: a quiet day and a broken pipeline look the same."""
    db = tmp_path / "seen.sqlite"
    sender = MagicMock()
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([])),
        patch("scrollless.main.build_summarizer") as build,
        patch("scrollless.main.EmailSender", return_value=sender),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db)])

    assert rc == 0
    build.assert_not_called()  # nothing to summarize, so no API spend
    sender.send.assert_called_once()
    rendered = sender.send.call_args.args[0]
    assert rendered.item_count == 0
    assert "Nothing new today" in rendered.text
    assert "nothing new" in rendered.subject


def test_all_summaries_failing_does_not_mark_posts_seen(config_file: Path, tmp_path: Path) -> None:
    """A digest nobody could summarize must not burn its posts."""
    db = tmp_path / "seen.sqlite"
    summarizer = MagicMock()
    summarizer.summarize.side_effect = ScrollLessError("model down")
    sender = MagicMock()

    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender", return_value=sender),
        patch("scrollless.main.build_digest_items", return_value=[]),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db)])
    assert rc == 0

    # Second run: the post is still considered new.
    summarizer2 = MagicMock()
    summarizer2.summarize.return_value = ItemSummary(gist="worked this time")
    summarizer2.synthesize.return_value = "intro"
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch("scrollless.main.build_summarizer", return_value=summarizer2),
        patch("scrollless.main.EmailSender", return_value=MagicMock()),
    ):
        main_module.main(["--config", str(config_file), "--db", str(db)])
    assert summarizer2.summarize.call_count == 1


def test_main_returns_two_on_missing_config(tmp_path: Path) -> None:
    rc = main_module.main(["--config", str(tmp_path / "missing.yaml")])
    assert rc == 2


def test_main_returns_two_on_missing_credentials(config_file: Path, tmp_path: Path) -> None:
    with patch(
        "scrollless.main._open_collectors",
        _fake_collectors([], collect_error=MissingCredentialsError("REDDIT_CLIENT_ID", "reddit")),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(tmp_path / "seen.sqlite")])
    assert rc == 2


def test_main_returns_two_when_summarizer_credentials_missing(
    config_file: Path, tmp_path: Path
) -> None:
    db = tmp_path / "seen.sqlite"
    with (
        patch("scrollless.main._open_collectors", _fake_collectors([_post("a")])),
        patch(
            "scrollless.main.build_summarizer",
            side_effect=MissingCredentialsError("DEEPSEEK_API_KEY", "deepseek"),
        ),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(db)])
    assert rc == 2


def test_main_returns_one_on_other_scrollless_error(config_file: Path, tmp_path: Path) -> None:
    with patch(
        "scrollless.main._open_collectors",
        _fake_collectors([], collect_error=ScrollLessError("boom")),
    ):
        rc = main_module.main(["--config", str(config_file), "--db", str(tmp_path / "seen.sqlite")])
    assert rc == 1


def test_comments_are_fetched_only_for_posts_that_survive_dedup(
    config_file: Path, tmp_path: Path
) -> None:
    """The reason collection is two-phase.

    On Reddit's feeds a comment thread costs a full 60-second window. Fetching
    during collection meant paying that for posts dedup was about to discard.
    """
    db = tmp_path / "seen.sqlite"
    summarizer = MagicMock()
    summarizer.summarize.return_value = ItemSummary(gist="summary")
    summarizer.synthesize.return_value = "intro"

    first = _fake_collectors([_post("a"), _post("b")])
    with (
        patch("scrollless.main._open_collectors", first),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender", return_value=MagicMock()),
    ):
        main_module.main(["--config", str(config_file), "--db", str(db)])
    # Set, not list: both posts share a score, so the recency factor decides
    # their order and it depends on which was constructed first.
    assert {p.id for p in first.collector.enrich.call_args.args[0]} == {"a", "b"}

    # Second run offers the same two posts; both were sent, so nothing reaches
    # enrichment and no comment feed is fetched at all.
    second = _fake_collectors([_post("a"), _post("b")])
    with (
        patch("scrollless.main._open_collectors", second),
        patch("scrollless.main.build_summarizer", return_value=summarizer),
        patch("scrollless.main.EmailSender", return_value=MagicMock()),
    ):
        main_module.main(["--config", str(config_file), "--db", str(db)])
    second.collector.collect.assert_called_once()
    second.collector.enrich.assert_not_called()


def test_dry_run_never_fetches_comments(config_file: Path, tmp_path: Path) -> None:
    """--dry-run now costs one listing request per source, nothing more."""
    opener = _fake_collectors([_post("a")])
    with patch("scrollless.main._open_collectors", opener):
        rc = main_module.main(
            ["--config", str(config_file), "--db", str(tmp_path / "seen.sqlite"), "--dry-run"]
        )
    assert rc == 0
    opener.collector.collect.assert_called_once()
    opener.collector.enrich.assert_not_called()
