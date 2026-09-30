"""Convert source HTML into plain text for the summarizer.

Reddit's Atom feeds escape their HTML twice (once for the document, once
inside ``<content>``) and Hacker News returns inline ``<p>`` and ``<a>`` tags.
Tags waste prompt budget and tend to be quoted back by the model.
"""

from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"[ \t]+")


def visible_text(fragment: str) -> str:
    """Strip markup and unescape entities, collapsing runs of spaces."""
    if not fragment:
        return ""
    text = html.unescape(html.unescape(fragment))
    text = _TAG_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", html.unescape(text)).strip()
