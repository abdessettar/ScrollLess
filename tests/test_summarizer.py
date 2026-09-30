from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import OpenAIError
from pydantic import ValidationError
from scrollless.config import SummarizationConfig
from scrollless.exceptions import MissingCredentialsError, SummarizerError
from scrollless.models import Comment, DigestItem, ItemSummary, Post
from scrollless.summarizer import (
    OpenAICompatibleSummarizer,
    _parse_summary,
    _render_user_prompt,
    _truncate,
)

_GOOD_JSON = json.dumps(
    {
        "gist": "DuckDB is replacing the Postgres parser it inherited.",
        "points": [
            "One maintainer explains the PEG rewrite",
            "Several ask about SQL dialect drift",
        ],
        "dissent": "One commenter calls the rewrite premature.",
        "takeaway": "Watch for dialect changes in 2.0.",
    }
)


def _post(
    *,
    body: str = "",
    comments: list[Comment] | None = None,
    score_is_estimated: bool = False,
) -> Post:
    return Post(
        id="x",
        source="reddit",
        channel="dataengineering",
        title="Postgres 17 released",
        url="https://reddit/x",
        external_url="https://postgresql.org/release/17",
        score=420,
        num_comments=80,
        created_utc=datetime(2026, 5, 17, 12, 0, tzinfo=UTC),
        body=body,
        comments=comments or [],
        author="alice",
        score_is_estimated=score_is_estimated,
    )


def _cfg(
    provider: str = "deepseek", style: str = "Be concise.", language: str = "auto"
) -> SummarizationConfig:
    return SummarizationConfig(
        provider=provider,  # type: ignore[arg-type]
        model="deepseek-chat",
        max_tokens_per_item=200,
        language=language,
        style=style,
    )


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client.chat.completions.create.return_value = response
    return client


# --------------------------------------------------------------------------
# structured output
# --------------------------------------------------------------------------


def test_summarize_returns_structured_summary() -> None:
    s = OpenAICompatibleSummarizer(_cfg(), client=_mock_client(_GOOD_JSON))
    summary = s.summarize(_post())
    assert summary.gist.startswith("DuckDB is replacing")
    assert len(summary.points) == 2
    assert summary.dissent == "One commenter calls the rewrite premature."
    assert summary.takeaway == "Watch for dialect changes in 2.0."
    assert summary.fallback_text == ""


def test_summarize_requests_json_mode() -> None:
    client = _mock_client(_GOOD_JSON)
    OpenAICompatibleSummarizer(_cfg(), client=client).summarize(_post())
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["max_tokens"] == 200


def test_non_json_answer_degrades_to_plain_text() -> None:
    """A model that ignores the contract must not cost us the item."""
    s = OpenAICompatibleSummarizer(_cfg(), client=_mock_client("- just some bullets"))
    summary = s.summarize(_post())
    assert summary.fallback_text == "- just some bullets"
    assert not summary.has_content


def test_empty_json_object_degrades_to_plain_text() -> None:
    s = OpenAICompatibleSummarizer(_cfg(), client=_mock_client('{"gist": "", "points": []}'))
    assert s.summarize(_post()).fallback_text != ""


def test_parse_summary_drops_blank_points_and_caps_them() -> None:
    raw = json.dumps({"gist": "g", "points": ["a", "", "  ", "b", "c", "d", "e"]})
    summary = _parse_summary(raw)
    assert summary.points == ["a", "b", "c", "d"]  # blanks gone, capped at four


def test_parse_summary_handles_a_json_list() -> None:
    assert _parse_summary('["not", "an object"]').fallback_text != ""


def test_missing_optional_fields_are_empty_not_none() -> None:
    summary = _parse_summary(json.dumps({"gist": "g", "points": ["p"]}))
    assert summary.dissent == ""
    assert summary.takeaway == ""
    assert summary.has_content


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------


def test_user_prompt_renders_metadata_and_comments() -> None:
    post = _post(
        body="Long-awaited release.",
        comments=[
            Comment(author="bob", body="Finally JSON_TABLE!", score=42, depth=0),
            Comment(
                author="carol", body="Logical replication improvements too.", score=10, depth=1
            ),
        ],
    )
    prompt = _render_user_prompt(post)
    assert "Source: reddit / dataengineering" in prompt
    assert "Title: Postgres 17 released" in prompt
    assert "Score: 420" in prompt
    assert "Links to: https://postgresql.org/release/17" in prompt
    assert "Long-awaited release." in prompt
    assert "bob: Finally JSON_TABLE!" in prompt
    assert "  - [10] carol: Logical replication improvements too." in prompt


def test_estimated_scores_are_kept_out_of_the_prompt() -> None:
    """Feeding a position-derived score to the model invites it to quote it."""
    prompt = _render_user_prompt(_post(score_is_estimated=True))
    assert "Score:" not in prompt
    assert "420" not in prompt


def test_long_body_cannot_starve_the_comments() -> None:
    """A self-post longer than the budget must still leave room for comments."""
    post = _post(
        body="essay. " * 2_000,  # ~14k chars, far over any budget
        comments=[Comment(author="bob", body="the thread's actual insight", score=9, depth=0)],
    )
    prompt = _render_user_prompt(post)
    assert "the thread's actual insight" in prompt
    assert len(prompt) <= 4_200  # still bounded


def test_comment_preview_cuts_at_a_sentence_boundary() -> None:
    body = "First sentence is short. " + "padding " * 200
    post = _post(comments=[Comment(author="a", body=body, score=1, depth=0)])
    prompt = _render_user_prompt(post)
    assert "First sentence is short." in prompt
    assert "padding " * 200 not in prompt


def test_truncate_respects_budget() -> None:
    out = _truncate("a" * 10_000, 100)
    assert len(out) <= 101
    assert out.endswith("…")


def test_truncate_passthrough_when_under_budget() -> None:
    assert _truncate("short", 100) == "short"


# --------------------------------------------------------------------------
# system prompt / config plumbing
# --------------------------------------------------------------------------


def test_system_message_forces_language_when_set() -> None:
    client = _mock_client(_GOOD_JSON)
    OpenAICompatibleSummarizer(_cfg(language="fr"), client=client).summarize(_post())
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "Write in fr." in messages[0]["content"]


def test_system_message_follows_the_source_language_when_auto() -> None:
    client = _mock_client(_GOOD_JSON)
    OpenAICompatibleSummarizer(_cfg(language="auto"), client=client).summarize(_post())
    sys_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "language of the source material" in sys_msg


def test_system_message_carries_the_json_contract() -> None:
    client = _mock_client(_GOOD_JSON)
    OpenAICompatibleSummarizer(_cfg(), client=client).summarize(_post())
    sys_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    for field in ("gist", "points", "dissent", "takeaway"):
        assert field in sys_msg


def test_style_overrides_used_for_matching_channel() -> None:
    cfg = SummarizationConfig(
        provider="deepseek",
        model="deepseek-chat",
        max_tokens_per_item=100,
        language="auto",
        style="default style",
        style_overrides={"dataengineering": "tech-specific style"},
    )
    client = _mock_client(_GOOD_JSON)
    OpenAICompatibleSummarizer(cfg, client=client).summarize(_post())  # channel=dataengineering
    sys_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "tech-specific style" in sys_msg
    assert "default style" not in sys_msg


def test_default_style_used_when_no_override() -> None:
    cfg = SummarizationConfig(
        provider="deepseek",
        model="deepseek-chat",
        max_tokens_per_item=100,
        language="auto",
        style="default style",
        style_overrides={"hn": "hn-specific"},
    )
    client = _mock_client(_GOOD_JSON)
    OpenAICompatibleSummarizer(cfg, client=client).summarize(_post())  # channel=dataengineering
    sys_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "default style" in sys_msg
    assert "hn-specific" not in sys_msg


def test_style_for_helper() -> None:
    cfg = SummarizationConfig(
        provider="deepseek",
        model="deepseek-chat",
        max_tokens_per_item=100,
        language="auto",
        style="default",
        style_overrides={"nba": "nba-style"},
    )
    assert cfg.style_for("nba") == "nba-style"
    assert cfg.style_for("dataengineering") == "default"


# --------------------------------------------------------------------------
# digest-level synthesis
# --------------------------------------------------------------------------


def test_synthesize_sends_every_item_and_returns_prose() -> None:
    client = _mock_client("  Two themes today: DuckDB internals and hiring.  ")
    s = OpenAICompatibleSummarizer(_cfg(), client=client)
    items = [
        DigestItem(post=_post(), summary=ItemSummary(gist="DuckDB swaps its parser")),
        DigestItem(post=_post(), summary=ItemSummary(gist="Freelance rates are flat")),
    ]
    out = s.synthesize(items)
    assert out == "Two themes today: DuckDB internals and hiring."
    kwargs = client.chat.completions.create.call_args.kwargs
    user_msg = kwargs["messages"][1]["content"]
    assert "DuckDB swaps its parser" in user_msg
    assert "Freelance rates are flat" in user_msg
    assert "response_format" not in kwargs  # prose, not JSON


def test_synthesize_without_items_makes_no_call() -> None:
    client = _mock_client("unused")
    assert OpenAICompatibleSummarizer(_cfg(), client=client).synthesize([]) == ""
    client.chat.completions.create.assert_not_called()


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------


def test_empty_completion_raises() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
    )
    s = OpenAICompatibleSummarizer(_cfg(), client=client)
    with pytest.raises(SummarizerError, match="empty completion"):
        s.summarize(_post())


def test_openai_error_wrapped_in_summarizer_error() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIError("rate limited")
    s = OpenAICompatibleSummarizer(_cfg(), client=client)
    with pytest.raises(SummarizerError, match="deepseek"):
        s.summarize(_post())


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError, match="DEEPSEEK_API_KEY"):
        OpenAICompatibleSummarizer(_cfg())


def test_config_rejects_unknown_provider() -> None:
    # Unsupported providers fail at config load, before any API call.
    with pytest.raises(ValidationError):
        SummarizationConfig(provider="unknown")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# truncated answers: a long summary is usually a good one, so salvage it
# --------------------------------------------------------------------------

# What DeepSeek actually returned when max_tokens cut a French thread short.
_TRUNCATED = """{
  "gist": "Le fil discute d'une tribune du Monde sur le climat.",
  "points": [
    "Plusieurs commentateurs soulignent l'évidence du propos.",
    "Un commentaire critique l'hypocrisie du journal."
  ],
  "dissent": "Certains lui reprochent de caricaturer.",
  "takeaway": "Le fil montre un scepticisme envers les médias et une frustration face à l'inaction climat"""


def test_truncated_json_is_recovered_not_dumped() -> None:
    summary = _parse_summary(_TRUNCATED)
    assert summary.fallback_text == ""  # the braces never reach the reader
    assert summary.gist.startswith("Le fil discute")
    assert len(summary.points) == 2
    assert summary.dissent == "Certains lui reprochent de caricaturer."
    # The sentence the model was mid-way through is dropped rather than shown
    # ending in "l'inaction climat".
    assert summary.takeaway == ""


def test_truncation_inside_a_point_drops_only_that_point() -> None:
    raw = '{"gist": "A gist.", "points": ["Finished sentence.", "Half a sen'
    summary = _parse_summary(raw)
    assert summary.points == ["Finished sentence."]
    assert summary.gist == "A gist."


def test_truncation_between_fields_keeps_everything_complete() -> None:
    raw = '{"gist": "A gist.", "points": ["One.", "Two."],'
    summary = _parse_summary(raw)
    assert summary.gist == "A gist."
    assert summary.points == ["One.", "Two."]


def test_code_fenced_json_is_parsed() -> None:
    fenced = '```json\n{"gist": "Fenced.", "points": ["A."]}\n```'
    assert _parse_summary(fenced).gist == "Fenced."


def test_unsalvageable_text_still_falls_back() -> None:
    summary = _parse_summary("I'm sorry, I can't summarize that thread.")
    assert summary.fallback_text.startswith("I'm sorry")
    assert not summary.has_content


def test_balanced_but_invalid_json_falls_back() -> None:
    """Salvage is for truncation, not for malformed output in general."""
    assert _parse_summary('{"gist": "x" "points": []}').fallback_text != ""


def test_recovered_summary_that_lost_everything_falls_back() -> None:
    raw = '{"gist": "Half a gi'
    assert _parse_summary(raw).fallback_text != ""


def test_a_gist_alone_is_a_complete_summary() -> None:
    """Threads with no discussion (clips, trivia) should return only a gist."""
    summary = _parse_summary(
        json.dumps(
            {"gist": "A highlight clip of a dunk.", "points": [], "dissent": "", "takeaway": ""}
        )
    )
    assert summary.has_content
    assert summary.fallback_text == ""
    assert summary.gist == "A highlight clip of a dunk."
    assert summary.points == []


def test_contract_tells_the_model_when_to_stop_at_a_gist() -> None:
    client = _mock_client(_GOOD_JSON)
    OpenAICompatibleSummarizer(_cfg(), client=client).summarize(_post())
    sys_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "nothing to summarize" in sys_msg
    assert "A one-line gist is the correct, complete answer" in sys_msg
