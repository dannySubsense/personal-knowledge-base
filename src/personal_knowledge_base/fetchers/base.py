"""Abstract base class for all fetchers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FetchResult:
    """Result of fetching content from a source.

    Attributes:
        url: The original URL that was fetched.
        title: Title of the content.
        content: The extracted text content.
        content_type: Type of content (youtube, pdf, article, etc.).
        author: Optional author name.
        published_date: Optional publication date.
        metadata: Additional metadata specific to the content type.
        success: Whether the fetch was successful.
        error_message: Error message if fetch failed.
        fetched_at: Timestamp when the content was fetched.

    """

    url: str
    title: str = ""
    content: str = ""
    content_type: str = ""
    author: str | None = None
    published_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: str | None = None
    fetched_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate the result after initialization."""
        if not self.success and not self.error_message:
            raise ValueError("Failed fetch must have an error message.")


class Fetcher(ABC):
    """Abstract base class for content fetchers.

    All fetchers must implement the `fetch` method to retrieve content
    from their respective sources.

    """

    @abstractmethod
    def fetch(self, url: str) -> FetchResult:
        """Fetch content from the given URL.

        Args:
            url: The URL to fetch content from.

        Returns:
            FetchResult containing the extracted content or error information.

        """
        ...

    @abstractmethod
    def can_fetch(self, url: str) -> bool:
        """Check if this fetcher can handle the given URL.

        Args:
            url: The URL to check.

        Returns:
            True if this fetcher can handle the URL, False otherwise.

        """
        ...
