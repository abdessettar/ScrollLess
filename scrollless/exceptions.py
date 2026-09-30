"""Domain-specific exception hierarchy.

Catching these at boundaries lets us distinguish recoverable per-source failures
(one subreddit erroring out) from fatal misconfiguration (missing credentials).
"""

from __future__ import annotations


class ScrollLessError(Exception):
    """Base class for all ScrollLess exceptions."""


class ConfigError(ScrollLessError):
    """Configuration file is missing, malformed, or invalid."""


class MissingCredentialsError(ScrollLessError):
    """A required environment variable / API key is not set."""

    def __init__(self, var_name: str, source: str) -> None:
        super().__init__(
            f"Missing required environment variable {var_name!r} for {source}. See .env.example."
        )
        self.var_name = var_name
        self.source = source


class InvalidCredentialsError(ScrollLessError):
    """Credentials were present but the service rejected them.

    Unlike :class:`MissingCredentialsError`, the variable is set but its
    value is wrong (typo, revoked key, deleted app).
    """

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"[{source}] {detail}")
        self.source = source


class SummarizerError(ScrollLessError):
    """LLM call failed in a way the caller should know about."""

    def __init__(self, provider: str, original: Exception) -> None:
        super().__init__(f"[{provider}] {type(original).__name__}: {original}")
        self.provider = provider
        self.original = original


class EmailDeliveryError(ScrollLessError):
    """SMTP send failed in a way the caller should know about."""

    def __init__(self, original: Exception) -> None:
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original


class CollectorError(ScrollLessError):
    """Recoverable failure while collecting from a single source/channel."""

    def __init__(self, source: str, channel: str, original: Exception) -> None:
        super().__init__(f"[{source}:{channel}] {type(original).__name__}: {original}")
        self.source = source
        self.channel = channel
        self.original = original
