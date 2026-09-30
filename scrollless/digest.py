"""Build summaries and render the digest in HTML and plain-text formats.

- ``build_digest_items`` makes one LLM call per post.
- ``build_intro`` makes one call for the opening paragraph.
- ``render_digest`` is pure templating, testable without an LLM.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from scrollless.exceptions import SummarizerError
from scrollless.models import DigestItem, Post
from scrollless.summarizer import Summarizer

log = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_SUBJECT_LEAD_CHARS = 60


def _autoescape_for(template_name: str | None) -> bool:
    """Autoescape HTML templates (``*.html.j2``) but never the text variant."""
    return bool(template_name and ".html" in template_name)


_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=_autoescape_for,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


@dataclass
class RenderedDigest:
    """A rendered digest in both HTML and plain-text forms.

    The caller picks which one(s) to use based on ``email.format`` config.
    """

    subject: str
    html: str
    text: str
    item_count: int


def build_digest_items(posts: Iterable[Post], summarizer: Summarizer) -> list[DigestItem]:
    """Summarize each post. Posts that fail summarization are dropped with a warning."""
    items: list[DigestItem] = []
    for post in posts:
        try:
            summary = summarizer.summarize(post)
        except SummarizerError as e:
            log.warning("Skipping %s:%s, summarization failed: %s", post.source, post.id, e)
            continue
        items.append(DigestItem(post=post, summary=summary))
    log.info("Built %d digest items", len(items))
    return items


def build_intro(items: list[DigestItem], summarizer: Summarizer) -> str:
    """The digest's opening lines. A failure here only drops the intro."""
    if not items:
        return ""
    try:
        return summarizer.synthesize(items)
    except SummarizerError as e:
        log.warning("Digest intro failed, sending without one: %s", e)
        return ""


def render_digest(
    items: list[DigestItem],
    *,
    subject: str,
    group_by_source: bool = True,
    intro: str = "",
    now: datetime | None = None,
) -> RenderedDigest:
    ts = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    grouped = _group_items(items) if group_by_source else []
    context = {
        "subject": subject,
        "items": items,
        "grouped": grouped,
        "group_by_source": group_by_source,
        "intro": intro,
        "generated_at": ts,
    }
    html = _env.get_template("digest.html.j2").render(**context)
    text = _env.get_template("digest.txt.j2").render(**context)
    return RenderedDigest(subject=subject, html=html, text=text, item_count=len(items))


def format_subject(
    template: str,
    *,
    count: int = 0,
    lead: str = "",
    now: datetime | None = None,
) -> str:
    """Fill the configured subject template.

    Placeholders: ``{date}``, ``{count}``, ``{lead}`` (the top item's title,
    trimmed). A template that uses none of them still works.
    """
    today = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
    headline = _trim_lead(lead) if count else "nothing new"
    try:
        return template.format(date=today, count=count, lead=headline)
    except KeyError as e:
        log.warning("Unknown placeholder %s in subject_template; using it verbatim", e)
        return template


def _trim_lead(lead: str) -> str:
    lead = " ".join(lead.split())
    if len(lead) <= _SUBJECT_LEAD_CHARS:
        return lead
    return lead[:_SUBJECT_LEAD_CHARS].rstrip(" ,;:-") + "…"


def _group_items(items: list[DigestItem]) -> list[tuple[str, list[DigestItem]]]:
    """Group by reader-facing channel label, best group first.

    Groups appear in the order of their best item, so the strongest channel
    of the day comes first.
    """
    groups: dict[str, list[DigestItem]] = {}
    for item in items:
        groups.setdefault(item.post.channel_label, []).append(item)
    return list(groups.items())
