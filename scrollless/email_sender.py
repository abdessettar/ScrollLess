"""SMTP delivery for the rendered digest.

Sends a multipart/alternative email with plain-text and HTML parts over
STARTTLS (port 587 by default).

The smtplib factory is injected so tests don't open real connections.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from collections.abc import Callable
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import Any

from scrollless.config import EmailConfig
from scrollless.digest import RenderedDigest
from scrollless.exceptions import EmailDeliveryError, MissingCredentialsError

log = logging.getLogger(__name__)

# Typed as Any to accept both smtplib.SMTP and test mocks without a Protocol
# that has to mirror smtplib's exact (and overloaded) method signatures.
SMTPFactory = Callable[[str, int], Any]


def _default_factory(host: str, port: int) -> smtplib.SMTP:
    return smtplib.SMTP(host, port, timeout=30)


class EmailSender:
    """Sends a rendered digest via SMTP using credentials from the environment."""

    def __init__(
        self,
        cfg: EmailConfig,
        *,
        smtp_factory: SMTPFactory = _default_factory,
    ) -> None:
        self.cfg = cfg
        self._factory = smtp_factory

        for var in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
            if not os.environ.get(var):
                raise MissingCredentialsError(var, source="smtp")

        self._host = os.environ["SMTP_HOST"]
        self._port = int(os.environ.get("SMTP_PORT", "587"))
        self._user = os.environ["SMTP_USER"]
        self._password = os.environ["SMTP_PASSWORD"]

    def send(self, digest: RenderedDigest) -> None:
        msg = self._build_message(digest)
        try:
            client = self._factory(self._host, self._port)
            try:
                client.starttls(context=ssl.create_default_context())
                client.login(self._user, self._password)
                client.send_message(msg)
            finally:
                client.quit()
        except (smtplib.SMTPException, OSError) as e:
            raise EmailDeliveryError(e) from e
        log.info(
            "Sent digest '%s' (%d items) to %s",
            digest.subject,
            digest.item_count,
            ", ".join(self.cfg.to),
        )

    def _build_message(self, digest: RenderedDigest) -> EmailMessage:
        msg = EmailMessage()
        msg["From"] = self.cfg.from_
        msg["To"] = ", ".join(formataddr((None, addr)) for addr in self.cfg.to)
        msg["Subject"] = digest.subject
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="scrollless")

        fmt = self.cfg.format
        if fmt == "text":
            msg.set_content(digest.text)
        elif fmt == "html":
            # Always include a plain-text fallback.
            msg.set_content(digest.text)
            msg.add_alternative(digest.html, subtype="html")
        else:  # both
            msg.set_content(digest.text)
            msg.add_alternative(digest.html, subtype="html")
        return msg
