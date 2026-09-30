"""Collect, rank, summarize and email one digest edition."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from dotenv import load_dotenv

from scrollless.collectors.base import Collector
from scrollless.collectors.hn import HackerNewsCollector
from scrollless.collectors.reddit import RedditCollector
from scrollless.collectors.reddit_rss import RedditRSSCollector
from scrollless.config import Config, load_config
from scrollless.dedup import SeenStore
from scrollless.digest import build_digest_items, build_intro, format_subject, render_digest
from scrollless.email_sender import EmailSender
from scrollless.exceptions import (
    ConfigError,
    EmailDeliveryError,
    InvalidCredentialsError,
    MissingCredentialsError,
    ScrollLessError,
)
from scrollless.logging_config import configure_logging
from scrollless.models import DigestItem, Post
from scrollless.ranker import rank
from scrollless.summarizer import build_summarizer

log = logging.getLogger("scrollless")

_DEFAULT_DB_PATH = Path("scrollless/db/seen.sqlite")


def _reddit_collector(cfg: Config) -> Collector:
    """Pick the Reddit access route. See RedditConfig.access."""
    if cfg.sources.reddit.access == "oauth":
        return RedditCollector(cfg.sources.reddit)
    return RedditRSSCollector(cfg.sources.reddit)


@contextmanager
def _open_collectors(cfg: Config, source: str) -> Iterator[list[Collector]]:
    """Hold the collectors open across dedup and ranking.

    They are used twice (listings first, then comments for whatever
    survives) and own HTTP clients plus, for the RSS collector, rate-limit
    state that has to carry over between the two phases.
    """
    with ExitStack() as stack:
        collectors: list[Collector] = []
        if source in ("reddit", "all"):
            collectors.append(stack.enter_context(_reddit_collector(cfg)))
        if source in ("hn", "all"):
            collectors.append(stack.enter_context(HackerNewsCollector(cfg.sources.hacker_news)))
        yield collectors


def _run_pipeline(args: argparse.Namespace, cfg: Config) -> None:
    """Run the full pipeline. Raises on errors; main() maps to exit codes.

    Comment threads are fetched only for posts that survive dedup and
    ranking: on Reddit's public feeds each one costs about a minute.

    Posts are marked as seen only after a successful send, so --dry-run and
    --no-send can be repeated without consuming content. An empty digest
    still sends a one-line note so a quiet day is not mistaken for a failure.
    """
    with (
        _open_collectors(cfg, args.source) as collectors,
        SeenStore(args.db, window_days=cfg.dedup.window_days) as store,
    ):
        raw_posts: list[Post] = []
        for collector in collectors:
            raw_posts.extend(collector.collect())

        fresh = store.filter_new(raw_posts)
        log.info("Dedup: %d -> %d posts", len(raw_posts), len(fresh))
        top = rank(
            fresh,
            max_items=cfg.digest.max_items,
            per_channel_max=cfg.digest.per_channel_max,
            channel_quotas=cfg.digest.channel_quotas,
        )

        if args.dry_run:
            log.info("Dry-run: skipping summarization (%d posts would be summarized)", len(top))
            for p in top:
                print(f"  [{p.score:>5}] {p.source}/{p.channel}: {p.title}")
            return

        # A collector may drop some of its posts here once it sees the
        # thread is empty.
        if top:
            for collector in collectors:
                top = collector.enrich(top)

        items: list[DigestItem] = []
        intro = ""
        if top:
            summarizer = build_summarizer(cfg.summarization)
            items = build_digest_items(top, summarizer)
            intro = build_intro(items, summarizer)
        else:
            log.info("Nothing new after dedup; sending the quiet-day note")

        subject = format_subject(
            cfg.email.subject_template,
            count=len(items),
            lead=items[0].post.title if items else "",
        )
        rendered = render_digest(
            items,
            subject=subject,
            group_by_source=cfg.digest.group_by_source,
            intro=intro,
        )

        if args.output:
            args.output.write_text(rendered.html, encoding="utf-8")
            log.info("Wrote HTML digest to %s", args.output)

        if args.no_send:
            log.info("--no-send: skipping SMTP delivery (posts NOT marked as seen)")
            print(rendered.text)
            return

        EmailSender(cfg.email).send(rendered)
        if items:
            # If every summary failed nothing was delivered, so nothing is marked.
            store.mark(top)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scrollless", description=__doc__)
    p.add_argument(
        "--config", default="config.yaml", help="Path to the YAML config (default: config.yaml)"
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    p.add_argument(
        "--source",
        choices=["reddit", "hn", "all"],
        default="all",
        help="Limit collection to a single source",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB_PATH,
        help="Path to the dedup SQLite database",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip marking posts as seen and skip LLM calls (no API spend)",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Write the rendered HTML digest to this file (for inspection)",
    )
    p.add_argument(
        "--no-send",
        action="store_true",
        help="Skip SMTP delivery (still runs collect/dedup/rank/summarize)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_dotenv()
    configure_logging(verbose=args.verbose)

    try:
        cfg = load_config(args.config)
        _run_pipeline(args, cfg)
    except (ConfigError, MissingCredentialsError, InvalidCredentialsError) as e:
        log.error("%s", e)
        return 2
    except EmailDeliveryError as e:
        log.error("Email delivery failed: %s (posts NOT marked as seen)", e)
        return 1
    except ScrollLessError as e:
        log.error("Unrecoverable error: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
