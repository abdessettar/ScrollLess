from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

Source = Literal["reddit", "hn"]


def domain_of(url: str | None) -> str:
    """Bare hostname of a link without "www.", or "" when there is no link."""
    if not url:
        return ""
    return urlparse(url).netloc.removeprefix("www.")


@dataclass
class Comment:
    author: str
    body: str
    score: int
    depth: int = 0


@dataclass
class Post:
    id: str
    source: Source
    channel: str  # subreddit name, or "hn"
    title: str
    url: str  # canonical link to discussion thread
    external_url: str | None  # link the post points to (None for self-posts)
    score: int
    num_comments: int
    created_utc: datetime
    body: str = ""  # selftext / HN story text (often empty)
    comments: list[Comment] = field(default_factory=list)
    author: str = ""
    # True when `score` is a stand-in derived from a source's own ordering
    # rather than a real number (Reddit's RSS feeds carry no score). The
    # ranker still uses it; the digest templates hide it.
    score_is_estimated: bool = False

    @property
    def permalink(self) -> str:
        return self.url

    @property
    def channel_label(self) -> str:
        """Human-readable channel name. Add a case here for each new source."""
        if self.source == "hn":
            return "Hacker News"
        return f"r/{self.channel}"

    @property
    def external_domain(self) -> str:
        """Bare domain of the linked article, for the byline under a title."""
        return domain_of(self.external_url)


@dataclass
class ItemSummary:
    """A summary broken into the pieces the digest lays out.

    The templates render these as a lead sentence, bullets, a counterpoint
    and a closing line, and omit any section that is empty.

    ``fallback_text`` holds the raw answer when the model returns something
    unparseable; the renderer shows it as-is rather than dropping the item.
    """

    gist: str = ""
    points: list[str] = field(default_factory=list)
    dissent: str = ""
    takeaway: str = ""
    fallback_text: str = ""

    @property
    def has_content(self) -> bool:
        """True when the structured fields carry something worth rendering."""
        return bool(self.gist or self.points or self.dissent or self.takeaway)


@dataclass
class DigestItem:
    """A post plus its LLM-generated summary, ready for rendering."""

    post: Post
    summary: ItemSummary
