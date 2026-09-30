from __future__ import annotations

import smtplib
from email.message import EmailMessage
from unittest.mock import MagicMock

import pytest
from scrollless.config import EmailConfig
from scrollless.digest import RenderedDigest
from scrollless.email_sender import EmailSender
from scrollless.exceptions import EmailDeliveryError, MissingCredentialsError


def _digest(subject: str = "Test Digest") -> RenderedDigest:
    return RenderedDigest(
        subject=subject,
        html="<html><body><h1>Hello</h1></body></html>",
        text="Hello\n=====\n\nplain text",
        item_count=3,
    )


def _cfg(format_: str = "html") -> EmailConfig:
    return EmailConfig.model_validate(
        {
            "from": "ScrollLess <digest@example.com>",
            "to": ["reader@example.com", "ops@example.com"],
            "subject_template": "Digest {date}",
            "format": format_,
        }
    )


@pytest.fixture
def smtp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pa55word")


def test_send_uses_starttls_login_and_send_message(smtp_env: None) -> None:
    client = MagicMock()
    EmailSender(_cfg(), smtp_factory=lambda h, p: client).send(_digest())

    client.starttls.assert_called_once()
    client.login.assert_called_once_with("user@example.com", "pa55word")
    client.send_message.assert_called_once()
    client.quit.assert_called_once()


def test_message_contains_html_and_text_alternatives(smtp_env: None) -> None:
    captured: list[EmailMessage] = []
    client = MagicMock()
    client.send_message.side_effect = captured.append

    EmailSender(_cfg(format_="html"), smtp_factory=lambda h, p: client).send(_digest())
    msg = captured[0]
    assert msg["Subject"] == "Test Digest"
    assert msg["From"] == "ScrollLess <digest@example.com>"
    assert "reader@example.com" in msg["To"]
    assert "ops@example.com" in msg["To"]

    # multipart/alternative with both text and html parts
    parts = list(msg.iter_parts())
    subtypes = {p.get_content_subtype() for p in parts}
    assert {"plain", "html"}.issubset(subtypes)


def test_format_text_only_sends_plain_only(smtp_env: None) -> None:
    captured: list[EmailMessage] = []
    client = MagicMock()
    client.send_message.side_effect = captured.append

    EmailSender(_cfg(format_="text"), smtp_factory=lambda h, p: client).send(_digest())
    msg = captured[0]
    assert msg.get_content_type() == "text/plain"
    assert not msg.is_multipart()


def test_quit_called_even_on_send_failure(smtp_env: None) -> None:
    client = MagicMock()
    client.send_message.side_effect = smtplib.SMTPException("relay denied")

    with pytest.raises(EmailDeliveryError, match="SMTPException"):
        EmailSender(_cfg(), smtp_factory=lambda h, p: client).send(_digest())

    client.quit.assert_called_once()


def test_oserror_during_connection_wrapped(smtp_env: None) -> None:
    def factory(_h: str, _p: int) -> MagicMock:
        raise OSError("connection refused")

    with pytest.raises(EmailDeliveryError, match="OSError"):
        EmailSender(_cfg(), smtp_factory=factory).send(_digest())


def test_missing_smtp_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingCredentialsError, match="SMTP_HOST"):
        EmailSender(_cfg())


def test_smtp_port_defaults_to_587(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "user")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.delenv("SMTP_PORT", raising=False)

    captured_port: list[int] = []

    def factory(_h: str, port: int) -> MagicMock:
        captured_port.append(port)
        return MagicMock()

    EmailSender(_cfg(), smtp_factory=factory).send(_digest())
    assert captured_port == [587]
