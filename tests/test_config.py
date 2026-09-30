from __future__ import annotations

from pathlib import Path

import pytest
from scrollless.config import RedditConfig, load_config
from scrollless.exceptions import ConfigError


def test_load_config_valid(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.sources.reddit.subreddits == ["dataengineering", "france"]
    assert cfg.sources.reddit.sort == "hot"
    assert cfg.sources.hacker_news.comment_depth == 2
    assert cfg.email.from_ == "ScrollLess <test@example.com>"
    assert cfg.summarization.provider == "deepseek"


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parent.parent / "examples").glob("config.*.yaml"))
)
def test_example_configs_are_valid(path: Path) -> None:
    load_config(path)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(p)


def test_load_config_validation_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("dedup:\n  window_days: 7\n", encoding="utf-8")  # most sections missing
    with pytest.raises(ConfigError, match="Invalid config"):
        load_config(p)


def test_unknown_sort_rejected(tmp_path: Path, valid_config_yaml: str) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(valid_config_yaml.replace("sort: hot", "sort: random"), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_skip_rules_match_bot_authors_case_insensitively() -> None:
    cfg = RedditConfig(subreddits=["x"])
    assert cfg.skip_authors == ["AutoModerator"]  # the default, since RSS lost `stickied`
    assert cfg.should_skip(author="AutoModerator", title="Monthly thread")
    assert cfg.should_skip(author="automoderator", title="Monthly thread")
    assert not cfg.should_skip(author="alice", title="Monthly thread")


def test_skip_rules_match_title_patterns() -> None:
    cfg = RedditConfig(
        subreddits=["x"],
        skip_authors=[],
        skip_title_patterns=[r"^(daily|weekly|monthly)\b.*\b(thread|discussion)\b"],
    )
    assert cfg.should_skip(author="alice", title="Monthly General Discussion - Aug 2026")
    assert cfg.should_skip(author="alice", title="daily discussion thread")  # case-insensitive
    assert not cfg.should_skip(author="alice", title="Duckdb moving away from postgres parser")


def test_invalid_skip_pattern_is_rejected_at_startup(
    tmp_path: Path, valid_config_yaml: str
) -> None:
    """A bad regex should fail the config, not blow up mid-collection."""
    p = tmp_path / "config.yaml"
    p.write_text(
        valid_config_yaml.replace(
            "    max_comments: 10",
            "    max_comments: 10\n    skip_title_patterns: ['(unclosed']",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="skip_title_pattern"):
        load_config(p)


def test_skip_rules_match_link_hosts_including_subdomains() -> None:
    cfg = RedditConfig(subreddits=["x"], skip_domains=["streamable.com", "v.redd.it"])
    assert cfg.should_skip(author="a", title="t", domain="streamable.com")
    assert cfg.should_skip(author="a", title="t", domain="clips.streamable.com")
    assert cfg.should_skip(author="a", title="t", domain="v.redd.it")
    assert not cfg.should_skip(author="a", title="t", domain="lemonde.fr")


def test_self_posts_are_never_skipped_by_domain() -> None:
    """A self-post has no host; an empty domain must not match a skip entry."""
    cfg = RedditConfig(subreddits=["x"], skip_domains=["streamable.com"])
    assert not cfg.should_skip(author="a", title="t", domain="")
