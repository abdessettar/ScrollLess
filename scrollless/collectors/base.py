from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import ClassVar

from scrollless.models import Post, Source


class Collector(ABC):
    """Fetches posts from a single source.

    Collection happens in two phases because comments are the expensive part
    and most posts are discarded:

    1. ``collect()`` returns posts from the cheap listing requests.
    2. ``enrich()`` fetches comment threads, and runs after dedup and ranking
       so only posts that will be summarized cost anything.

    Implementations may own resources such as an HTTP client. Use them as
    context managers:

        with SomeCollector(cfg) as c:
            posts = c.collect()
            ...
            posts = c.enrich(posts)
    """

    #: Value of ``Post.source`` for the posts this collector produces, so it
    #: can pick its own out of a mixed list in ``enrich()``.
    SOURCE: ClassVar[Source]

    @abstractmethod
    def collect(self) -> list[Post]: ...

    def enrich(self, posts: list[Post]) -> list[Post]:
        """Fill in what ``collect()`` deferred, for the posts that survived.

        Returns the list minus any of *this collector's* posts that turned out
        not to be worth summarizing after all. Posts from other sources pass
        through untouched. Default: nothing to add.
        """
        return posts

    def close(self) -> None:
        """Release any owned resources. Override in subclasses that own state."""
        return

    def __enter__(self) -> Collector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
