# Contributing

Issues and pull requests are welcome. For larger changes, open an issue first to discuss the idea.

## Setup

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy scrollless
uv run pytest
```

CI runs the same checks. Tests never touch the network: HTTP goes through
`httpx.MockTransport`, and the LLM and SMTP clients are injected mocks.

## Layout

```
scrollless/
  main.py            CLI and pipeline
  config.py          YAML schema (pydantic)
  collectors/        one module per source: reddit_rss.py, reddit.py (OAuth), hn.py
  dedup.py           SQLite store of sent posts
  ranker.py          scoring and per-channel caps
  summarizer.py      LLM prompts and JSON parsing
  digest.py          rendering, with templates/ for HTML and text
  email_sender.py    SMTP delivery
examples/            sample edition configs
scripts/             run script and systemd installer
tests/
```

## Adding a source

1. Add the source name to `Source` in `models.py` and a label in `Post.channel_label`.
2. Add a config model in `config.py` and reference it from `SourcesConfig`.
3. Subclass `Collector` in `collectors/`. `collect()` returns posts from cheap listing requests.
   `enrich()` fetches comments and only runs for posts that survived dedup and ranking.
4. Register it in `_open_collectors()` and the `--source` choices in `main.py`.
5. Add tests with mocked HTTP responses, and an example in `examples/`.

## Guidelines

- Keep the checks green and add tests for behaviour changes.
- Never commit `.env`, personal configs or the dedup database (all git-ignored).
- Be polite to upstream APIs: honour their rate limits and use a descriptive User-Agent.
