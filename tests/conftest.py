from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def valid_config_yaml() -> str:
    return dedent(
        """\
        sources:
          reddit:
            subreddits: [dataengineering, france]
            sort: hot
            min_score: 50
            max_posts_per_sub: 5
            max_comments: 10
          hacker_news:
            enabled: true
            min_score: 50
            max_posts: 10
            tags: [story]
            comment_depth: 2
        digest:
          max_items: 20
          group_by_source: true
        summarization:
          provider: deepseek
          model: deepseek-chat
          max_tokens_per_item: 300
          language: auto
          style: "Be concise."
        dedup:
          window_days: 7
        email:
          from: "ScrollLess <test@example.com>"
          to: [test@example.com]
          subject_template: "Digest · {date} · {count} items · {lead}"
          format: html
        """
    )


@pytest.fixture
def config_file(tmp_path: Path, valid_config_yaml: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(valid_config_yaml, encoding="utf-8")
    return p
