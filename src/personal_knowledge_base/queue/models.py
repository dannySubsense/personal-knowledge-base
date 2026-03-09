"""Job dataclass for the queue system."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Job:
    """Represents a job in the ingestion queue.

    Attributes:
        id: Unique identifier (UUID).
        url: URL or path to the content to be ingested.
        priority: Priority level (1=immediate, 2=normal).
        status: Current status (pending, processing, done, failed).
        content_type: Type of content (youtube, pdf, article, image, code).
        kb_name: Target knowledge base name.
        created_at: Timestamp when the job was created.
        updated_at: Timestamp when the job was last updated.
        retry_count: Number of retry attempts.
        error_message: Error message if the job failed.

    """

    id: str
    url: str
    priority: int = 2
    status: str = "pending"
    content_type: str | None = None
    kb_name: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validate job attributes after initialization."""
        if self.priority not in (1, 2):
            raise ValueError(f"Invalid priority: {self.priority}. Must be 1 or 2.")
        valid_statuses = ("pending", "processing", "done", "failed")
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}.")
        valid_types = (None, "youtube", "pdf", "article", "image", "code", "code_repo")
        if self.content_type not in valid_types:
            raise ValueError(
                f"Invalid content_type: {self.content_type}. "
                f"Must be one of {valid_types[1:]} or None."
            )
