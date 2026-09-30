from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from scrollless.exceptions import ConfigError


def _host_matches(domain: str, skip: str) -> bool:
    domain, skip = domain.casefold(), skip.casefold().removeprefix("www.")
    return domain == skip or domain.endswith(f".{skip}")


class RedditConfig(BaseModel):
    # "rss": public feeds, no credentials, no scores, about 1 request/minute.
    # "oauth": official API, needs a script app approved by Reddit.
    access: Literal["rss", "oauth"] = "rss"
    subreddits: list[str]
    sort: Literal["hot", "top", "new"] = "hot"
    time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "day"
    min_score: int = 50
    # Per-subreddit overrides for min_score. Typical scores differ by orders
    # of magnitude between large and niche communities.
    min_scores: dict[str, int] = Field(default_factory=dict)
    max_posts_per_sub: int = 10
    max_comments: int = 15
    # Bot posts and recurring megathreads summarize to nothing but still cost
    # a request, a digest slot and an LLM call. RSS feeds have no `stickied`
    # flag, so they are matched by author and title instead.
    skip_authors: list[str] = Field(default_factory=lambda: ["AutoModerator"])
    skip_title_patterns: list[str] = Field(default_factory=list)
    # Link hosts whose posts rarely carry a discussion worth summarizing,
    # typically video clip sites. Subdomains match too.
    skip_domains: list[str] = Field(default_factory=list)
    # Pause between requests. Both collectors also follow Reddit's own
    # rate-limit headers, so this is only a courtesy delay.
    request_delay_seconds: float = 0.2

    @field_validator("skip_title_patterns")
    @classmethod
    def _patterns_must_compile(cls, patterns: list[str]) -> list[str]:
        """Fail at startup, not mid-run, on a bad regex."""
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"invalid skip_title_pattern {pattern!r}: {e}") from e
        return patterns

    def should_skip(self, *, author: str, title: str, domain: str = "") -> bool:
        """True for posts that are never worth a digest slot.

        Authors match case-insensitively and exactly; titles match as
        case-insensitive regular expressions; domains match the host or any
        subdomain of it.
        """
        if any(author.casefold() == skip.casefold() for skip in self.skip_authors):
            return True
        if domain and any(_host_matches(domain, skip) for skip in self.skip_domains):
            return True
        return any(re.search(pattern, title, re.IGNORECASE) for pattern in self.skip_title_patterns)

    def threshold_for(self, subreddit: str) -> int:
        """Return the effective min_score for a given subreddit.

        Only meaningful in oauth mode, since RSS feeds carry no score.
        """
        return self.min_scores.get(subreddit, self.min_score)


class HackerNewsConfig(BaseModel):
    enabled: bool = True
    min_score: int = 50
    max_posts: int = 20
    tags: list[str] = Field(default_factory=lambda: ["story"])
    comment_depth: int = 2
    window_hours: int = 24  # only consider stories posted in the last N hours


class SourcesConfig(BaseModel):
    reddit: RedditConfig
    hacker_news: HackerNewsConfig


class DigestConfig(BaseModel):
    max_items: int = 20
    group_by_source: bool = True
    # Default cap per channel (subreddit name, or "hn" for Hacker News).
    # None = no default cap. Specific channels can override via `channel_quotas`.
    per_channel_max: int | None = None
    channel_quotas: dict[str, int] = Field(default_factory=dict)


class SummarizationConfig(BaseModel):
    provider: Literal["deepseek", "openai"] = "deepseek"
    model: str = "deepseek-chat"
    max_tokens_per_item: int = 300
    language: str = "auto"
    style: str = ""
    # Per-channel prompt overrides. A channel not listed here uses `style`.
    style_overrides: dict[str, str] = Field(default_factory=dict)

    def style_for(self, channel: str) -> str:
        """Return the style prompt to use for posts from this channel."""
        return self.style_overrides.get(channel, self.style)


class DedupConfig(BaseModel):
    window_days: int = 7


class EmailConfig(BaseModel):
    from_: str = Field(alias="from")
    to: list[str]
    subject_template: str = "ScrollLess Digest {date}"
    format: Literal["html", "text", "both"] = "html"

    model_config = {"populate_by_name": True}


class Config(BaseModel):
    sources: SourcesConfig
    digest: DigestConfig
    summarization: SummarizationConfig
    dedup: DedupConfig
    email: EmailConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {p}: {e}") from e
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Invalid config in {p}:\n{e}") from e
