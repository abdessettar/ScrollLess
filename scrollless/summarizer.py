"""LLM-backed summarization.

Any OpenAI-compatible chat API (DeepSeek, OpenAI) sits behind the small
``Summarizer`` interface. ``summarize(post)`` runs once per digest item and
``synthesize(items)`` once per digest for the opening paragraph.

The model answers in JSON so each part of a summary (gist, points, dissent,
takeaway) can be length-limited and left out when empty.

Per-call cost is bounded by a character budget per prompt section (so a long
post cannot crowd out the comments) and by ``max_tokens`` from config.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI, OpenAIError

from scrollless.config import SummarizationConfig
from scrollless.exceptions import MissingCredentialsError, SummarizerError
from scrollless.models import Comment, DigestItem, ItemSummary, Post

log = logging.getLogger(__name__)

# Input budget per item: about 1,000 tokens at roughly 4 characters per token.
_PROMPT_CHAR_BUDGET = 4_000
# The post body gets at most this much; the rest is kept for the comments.
_BODY_CHAR_BUDGET = 1_400
_COMMENT_PREVIEW_CHARS = 500
_MAX_POINTS = 4

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s")

_PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": None,  # SDK default
}

_PROVIDER_ENV_VAR = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# The output format lives in code; config prompts only describe what matters
# in a channel. A template change never requires editing every prompt.
_JSON_CONTRACT = """\
Answer with one JSON object and nothing around it:

{
  "gist": "One sentence: what this thread actually is. Plain prose. Never restate the title.",
  "points": ["2-4 entries, one or two sentences each, 40 words maximum. The strongest arguments, findings or reactions in the discussion. Attribute by stance or role ('one commenter', 'several freelancers', 'the author replies'), never by username. Stay concrete: numbers, tools, trade-offs."],
  "dissent": "The notable disagreement or correction, one or two sentences. Empty string if the thread genuinely has none.",
  "takeaway": "One practical line: the thing to try, the trap to avoid, the claim to distrust. If the thread carries no practical implication, return an empty string. Never describe the thread instead. 'The thread shows that opinions are divided' is not a takeaway; leave it empty."
}

Keep the whole object under 200 words: a summary that runs long gets cut off
mid-sentence and is worth less than a short one. Never pad a field to fill it:
an empty string beats a platitude.

Some threads have nothing to summarize. A highlight clip, a screenshot, a
trivia post, a joke: the comments are reactions, not arguments. For those,
write the gist and leave "points", "dissent" and "takeaway" as empty strings
and an empty list. Do not turn reactions into bullets, and do not manufacture
a lesson from them. "Even elite defense can't stop a well-executed stepback,
so defenders should force tougher shots" is filler, not an insight, and
"avoid provoking Kareem Abdul-Jabbar in public discourse" is a joke the digest
does not need. A one-line gist is the correct, complete answer for such a
thread."""

_SYNTHESIS_PROMPT = """\
You write the opening of a personal daily reading digest. Given the day's
items, write 2-3 sentences on what the day actually holds: the themes worth
the reader's attention, and anything several sources are circling at once.

Address the reader directly and plainly. No greeting, no sign-off, no
"in today's digest", no bullet points, no headings. If the day is thin, say
that instead of inflating it."""


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, post: Post) -> ItemSummary: ...

    @abstractmethod
    def synthesize(self, items: list[DigestItem]) -> str: ...


class OpenAICompatibleSummarizer(Summarizer):
    """Works against any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, cfg: SummarizationConfig, client: OpenAI | None = None) -> None:
        if cfg.provider not in _PROVIDER_ENV_VAR:
            raise ValueError(
                f"OpenAICompatibleSummarizer does not support provider {cfg.provider!r}"
            )
        self.cfg = cfg
        if client is not None:
            self._client = client
        else:
            var = _PROVIDER_ENV_VAR[cfg.provider]
            api_key = os.environ.get(var)
            if not api_key:
                raise MissingCredentialsError(var, source=cfg.provider)
            base_url = _PROVIDER_BASE_URLS.get(cfg.provider)
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def summarize(self, post: Post) -> ItemSummary:
        messages = self._build_messages(post)
        content = self._complete(
            messages,
            max_tokens=self.cfg.max_tokens_per_item,
            temperature=0.3,
            json_mode=True,
        )
        return _parse_summary(content)

    def synthesize(self, items: list[DigestItem]) -> str:
        """Two or three sentences across the whole digest."""
        if not items:
            return ""
        inventory = "\n".join(
            f"- [{item.post.channel}] {item.post.title}"
            + (f": {item.summary.gist}" if item.summary.gist else "")
            for item in items
        )
        content = self._complete(
            [
                {"role": "system", "content": _SYNTHESIS_PROMPT + self._language_rule()},
                {"role": "user", "content": inventory},
            ],
            max_tokens=200,
            temperature=0.4,
            json_mode=False,
        )
        return content.strip()

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        extra: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
        except OpenAIError as e:
            raise SummarizerError(self.cfg.provider, e) from e

        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            raise SummarizerError(self.cfg.provider, RuntimeError("empty completion"))
        return content

    def _build_messages(self, post: Post) -> list[dict[str, Any]]:
        # The channel prompt says what matters; the JSON contract and language
        # rule are appended so every channel gets the same output shape.
        style = self.cfg.style_for(post.channel).strip() or "Summarize the discussion."
        system = f"{style}\n\n{_JSON_CONTRACT}{self._language_rule()}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _render_user_prompt(post)},
        ]

    def _language_rule(self) -> str:
        if self.cfg.language == "auto":
            return (
                "\n\nWrite in the language of the source material: a French thread "
                "gets a French summary, an English one gets English."
            )
        return f"\n\nWrite in {self.cfg.language}."


def _parse_summary(raw: str) -> ItemSummary:
    """Turn the model's JSON into an ItemSummary, degrading gracefully.

    Three outcomes: clean JSON; JSON cut off by the token limit, which is
    closed and trimmed here; or unreadable output, kept whole in
    ``fallback_text`` so the item is still shown.
    """
    text = _strip_code_fence(raw.strip())
    truncated = False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = _close_truncated_json(text)
        if repaired is None:
            log.debug("Summary was not JSON; keeping it as plain text")
            return ItemSummary(fallback_text=text)
        log.info("Summary was cut off mid-JSON; recovered the complete fields")
        data, truncated = repaired, True
    if not isinstance(data, dict):
        return ItemSummary(fallback_text=text)

    points = [str(p).strip() for p in (data.get("points") or []) if str(p).strip()]
    summary = ItemSummary(
        gist=str(data.get("gist") or "").strip(),
        points=points[:_MAX_POINTS],
        dissent=str(data.get("dissent") or "").strip(),
        takeaway=str(data.get("takeaway") or "").strip(),
    )
    if truncated:
        _drop_unfinished(summary)
    if not summary.has_content:
        # Valid but empty JSON: show the raw answer rather than a blank item.
        return ItemSummary(fallback_text=text)
    return summary


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[-1]
    return body.rsplit("```", 1)[0].strip()


def _close_truncated_json(text: str) -> dict[str, Any] | None:
    """Close an object the model stopped writing mid-way, or give up.

    Walks the text tracking string state and open brackets, shuts whatever is
    still open, and re-parses. Returns None when the result is still not
    valid JSON.
    """
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string and ch in "{[":
            stack.append(ch)
        elif not in_string and ch in "}]" and stack:
            stack.pop()
    if not stack and not in_string:
        return None  # balanced already; it failed to parse for another reason

    repaired = text
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip().rstrip(",")
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _drop_unfinished(summary: ItemSummary) -> None:
    """Discard the sentence the model was mid-way through when it was cut."""
    if summary.points and not _looks_finished(summary.points[-1]):
        summary.points.pop()
    if not _looks_finished(summary.takeaway):
        summary.takeaway = ""
    if not _looks_finished(summary.dissent):
        summary.dissent = ""
    if not _looks_finished(summary.gist):
        summary.gist = ""


def _looks_finished(text: str) -> bool:
    """Empty is fine; a sentence stopping mid-word is not."""
    stripped = text.rstrip()
    return not stripped or stripped.endswith((".", "!", "?", "…", "»", '"', "'", ")", ":"))


def _render_user_prompt(post: Post) -> str:
    header = f"Source: {post.source} / {post.channel}\nTitle: {post.title}"
    if not post.score_is_estimated:
        # RSS posts carry a position-derived score. Leave it out so the model
        # never quotes it as a real number.
        header += f"\nScore: {post.score}  |  Comments: {post.num_comments}"
    if post.external_url:
        header += f"\nLinks to: {post.external_url}"

    sections = [header]
    body = _truncate(post.body.strip(), _BODY_CHAR_BUDGET)
    if body:
        sections.append(f"--- Post body ---\n{body}")

    # Whatever the header and body didn't spend belongs to the thread.
    spent = sum(len(s) + 2 for s in sections)
    comments_text = _render_comments(post.comments, budget=max(_PROMPT_CHAR_BUDGET - spent, 0))
    if comments_text:
        ordering = "top-voted first" if not post.score_is_estimated else "in the source's own order"
        sections.append(f"--- Comments ({ordering}) ---\n{comments_text}")

    return "\n\n".join(sections)


def _render_comments(comments: list[Comment], *, budget: int) -> str:
    lines: list[str] = []
    used = 0
    for c in comments:
        body = _preview(c.body.strip().replace("\n", " "), _COMMENT_PREVIEW_CHARS)
        if not body:
            continue
        indent = "  " * c.depth
        score_tag = f"[{c.score}] " if c.score else ""
        line = f"{indent}- {score_tag}{c.author}: {body}"
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _preview(text: str, budget: int) -> str:
    """Trim to `budget`, backing up to the last sentence end so takes keep their point."""
    if len(text) <= budget:
        return text
    window = text[:budget]
    sentences = list(_SENTENCE_END.finditer(window))
    if sentences and sentences[-1].start() > budget // 2:
        return window[: sentences[-1].start() + 1]
    return window.rstrip() + "…"


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    log.debug("Truncating section from %d to %d chars", len(text), budget)
    return _preview(text, budget)


def build_summarizer(cfg: SummarizationConfig) -> Summarizer:
    """Return the summarizer for the configured provider."""
    if cfg.provider in {"deepseek", "openai"}:
        return OpenAICompatibleSummarizer(cfg)
    raise ValueError(f"Unsupported summarization provider: {cfg.provider}")
