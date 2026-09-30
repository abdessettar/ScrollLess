# ScrollLess

A daily email digest of the best Reddit and Hacker News discussions, summarized by an LLM.

Instead of scrolling feeds, you get one email per edition. For each thread you get a
one-line gist, the strongest points from the comments, the main disagreement, and a
practical takeaway. Sections that have nothing to say are left out.

## How it works

1. **Collect** top posts from your subreddits (public RSS feeds, no account needed) and Hacker News.
2. **Deduplicate** against everything sent in the last 7 days, including the same link posted in several places.
3. **Rank** by score, recency and comment activity, with a cap per channel.
4. **Fetch comments** only for the posts that made the cut.
5. **Summarize** each thread with an OpenAI-compatible LLM (DeepSeek by default), using a prompt per channel.
6. **Email** an HTML digest with a plain-text fallback over SMTP.

Posts are only marked as seen after the email is sent, so previews and failed runs do not consume content.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- An API key for [DeepSeek](https://platform.deepseek.com/) or OpenAI
- An SMTP account to send from (for Gmail, an [App Password](https://myaccount.google.com/apppasswords))

A typical run costs well under one cent with `deepseek-chat`. Reddit's public feeds allow about
one request per minute, so a run with a few subreddits takes 15 to 25 minutes.

## Quick start

Fork the repository so you can keep your own changes, then:

```bash
git clone https://github.com/<your-username>/ScrollLess.git
cd ScrollLess
uv sync

cp .env.example .env                                   # add your API key and SMTP credentials
cp examples/config.evening.yaml config.evening.yaml    # set your subreddits and email address
```

Try it without sending anything:

```bash
# Rank posts only: no LLM calls, no email, no cost
uv run scrollless --config config.evening.yaml --dry-run

# Full pipeline, but save the email to a file instead of sending it
uv run scrollless --config config.evening.yaml --no-send --output digest.html
```

Send it for real:

```bash
uv run scrollless --config config.evening.yaml
```

## Configuration

Secrets go in `.env`. Everything else goes in a YAML config file. Both are git-ignored, and
the config is validated at startup.

Each config file is one **edition**: a set of sources and prompts sent as one email. The
examples define a morning edition (sports and community) and an evening edition (tech and
Hacker News), but you can have as many as you like: `config.<name>.yaml`. All editions share
the dedup database, so a thread is never sent twice.

The settings you will most likely change:

| Setting | What it does |
|---|---|
| `sources.reddit.subreddits` | Subreddits to read. |
| `sources.reddit.skip_authors`, `skip_domains`, `skip_title_patterns` | Filter out bots, video clips and recurring threads. |
| `sources.hacker_news.enabled`, `min_score`, `tags` | Hacker News stories to consider. |
| `digest.max_items`, `per_channel_max`, `channel_quotas` | How many items in total and per channel (`hn` for Hacker News). |
| `summarization.style` | Default prompt: what matters in a thread and what to ignore. |
| `summarization.style_overrides` | Prompt per channel, keyed by subreddit name or `hn`. |
| `summarization.language` | `auto` answers in the thread's language, or name one, e.g. `English`. |
| `email.from`, `email.to` | Sender (normally your SMTP user) and recipients. |
| `email.subject_template` | Placeholders: `{lead}` (top story), `{count}`, `{date}`. |

The example configs are commented and cover every option. Prompts only describe what matters
in a channel. The output format is fixed in `scrollless/summarizer.py`.

**LLM provider.** Set `summarization.provider` to `deepseek` or `openai`, `model` to a model
name, and the matching `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` in `.env`.

**Reddit access.** The default `access: rss` needs no credentials but has no vote scores, so
posts are ranked by Reddit's own order and `min_score` is ignored. If you have an approved
Reddit script app, `access: oauth` uses the official API (with scores) and the `REDDIT_*`
variables from `.env.example`.

**Email.** Any SMTP server with STARTTLS works (port 587 by default). Microsoft personal
accounts (Outlook.com, Hotmail) no longer accept password login over SMTP, so send through
another provider. You can still deliver to an Outlook address.

## Scheduling

### Linux (systemd)

Install one timer per edition. Each argument is `<edition>=<schedule>`, where the schedule is
any systemd [calendar expression](https://www.freedesktop.org/software/systemd/man/systemd.time.html#Calendar%20Events):

```bash
scripts/install-systemd.sh morning=08:00 evening=18:00
scripts/install-systemd.sh weekdays="Mon..Fri 07:30"
```

The script writes user units to `~/.config/systemd/user/` and enables them. Missed runs
happen at the next boot or wake-up. To run timers while you are logged out:
`sudo loginctl enable-linger $USER`.

```bash
systemctl --user list-timers 'scrollless-*'           # next run times
systemctl --user start scrollless@evening.service     # run an edition now
journalctl --user -u scrollless@evening.service -f    # logs
scripts/install-systemd.sh --uninstall                # remove everything
```

Re-run the installer after moving the project or to change a schedule.

### cron (Linux or macOS)

`scripts/run-digest.sh <edition>` works from any directory, so it can be used directly in
`crontab -e`:

```cron
0 8  * * *  /path/to/ScrollLess/scripts/run-digest.sh morning >> /tmp/scrollless.log 2>&1
0 18 * * *  /path/to/ScrollLess/scripts/run-digest.sh evening >> /tmp/scrollless.log 2>&1
```

cron skips runs while the machine is off or asleep.

### Keep the database

The list of sent posts is stored in `scrollless/db/seen.sqlite` (or at the path given by
`--db`). Scheduled runs need a persistent disk, which is why ScrollLess runs on your own
machine or a small server rather than in a stateless CI job.

## Command line

```
uv run scrollless [--config FILE] [--dry-run] [--no-send] [--output FILE]
                  [--source {reddit,hn,all}] [--db FILE] [--verbose]
```

| Option | Effect |
|---|---|
| `--config` | Config file to use (default `config.yaml`). |
| `--dry-run` | Print the ranked posts. No LLM calls, no email. |
| `--no-send` | Run everything except the email and print the plain-text digest. |
| `--output` | Also write the HTML email to a file. |
| `--source` | Only collect from one source. |
| `--db` | Dedup database path. Use a temporary file to see already-sent posts again. |
| `--verbose` | Debug logging. |

Exit codes: `0` success (including a "nothing new" email), `1` temporary failure such as a
network or SMTP error, `2` configuration or credential problem.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy scrollless
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project layout and how to add a source.

## License

[MIT](LICENSE)
