"""WhatsApp message handler for the Personal Knowledge Base.

Detects URLs in WhatsApp messages and queues them for processing.
Handles duplicates and trusted sender filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from personal_knowledge_base.queue.models import Job
from personal_knowledge_base.queue.operations import add_job, list_jobs

# Robust URL regex: matches http/https URLs with paths, query strings, fragments
_URL_RE = re.compile(
    r"https?://" r"(?:[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)" r"(?<![.,;:!?\)])",
    re.IGNORECASE,
)


@dataclass
class WhatsAppConfig:
    """Configuration for the WhatsApp handler.

    Attributes:
        db_path: Path to the queue SQLite database.
        duplicate_window_days: Consider URL a duplicate if queued within this many days.
        default_priority: Default job priority (1=immediate, 2=normal).
        trusted_senders: Phone numbers allowed to queue URLs. Empty list = open mode.

    """

    db_path: str = "~/pkb-data/queue.db"
    duplicate_window_days: int = 30
    default_priority: int = 2
    trusted_senders: list[str] = field(default_factory=list)


@dataclass
class HandleResult:
    """Result of processing a WhatsApp message.

    Attributes:
        urls_found: Total number of URLs extracted from the message.
        queued: Number of URLs successfully queued.
        duplicates: Number of URLs skipped due to duplicate detection.
        rejected: Number of URLs rejected (untrusted sender).
        job_ids: List of job IDs for queued URLs.
        message: Human-readable summary of the result.

    """

    urls_found: int = 0
    queued: int = 0
    duplicates: int = 0
    rejected: int = 0
    job_ids: list[str] = field(default_factory=list)
    message: str = ""


class WhatsAppHandler:
    """Handle incoming WhatsApp messages, extracting and queuing URLs.

    Example::

        config = WhatsAppConfig(trusted_senders=["+16039885837"])
        handler = WhatsAppHandler(config)
        result = handler.handle_message("+16039885837", "Check https://example.com")

    """

    def __init__(self, config: WhatsAppConfig | None = None) -> None:
        """Initialise the handler with optional config.

        Args:
            config: Configuration dataclass. Defaults to WhatsAppConfig().

        """
        self.config = config if config is not None else WhatsAppConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_message(self, sender: str, message: str) -> HandleResult:
        """Process an incoming WhatsApp message.

        Extracts URLs, enforces trusted-sender policy, deduplicates, and
        queues new URLs for processing.

        Args:
            sender: Sender phone number (e.g. "+16039885837").
            message: Raw message text.

        Returns:
            HandleResult summarising what happened.

        """
        result = HandleResult()

        urls = self.extract_urls(message)
        result.urls_found = len(urls)

        if not urls:
            result.message = "No URLs found"
            return result

        # Trusted-sender check (open mode when trusted_senders is empty)
        if self.config.trusted_senders and sender not in self.config.trusted_senders:
            result.rejected = result.urls_found
            result.message = "Untrusted sender"
            return result

        for url in urls:
            if self.is_duplicate(url):
                result.duplicates += 1
            else:
                job = self.queue_url(url)
                result.queued += 1
                result.job_ids.append(job)

        result.message = self._build_summary(result)
        return result

    def extract_urls(self, text: str) -> list[str]:
        """Extract all HTTP/HTTPS URLs from *text*.

        Args:
            text: Arbitrary message text.

        Returns:
            Ordered list of unique URLs found (preserves first occurrence order).

        """
        raw = _URL_RE.findall(text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        urls: list[str] = []
        for url in raw:
            if url not in seen:
                seen.add(url)
                urls.append(url)
        return urls

    def is_duplicate(self, url: str) -> bool:
        """Return True if *url* was queued within the configured duplicate window.

        Args:
            url: The URL to check.

        Returns:
            True if a recent job exists for this URL, False otherwise.

        """
        cutoff = datetime.now(UTC) - timedelta(days=self.config.duplicate_window_days)
        jobs: list[Job] = list_jobs()
        return any(job.url == url and job.created_at >= cutoff for job in jobs)

    def queue_url(self, url: str, priority: int | None = None) -> str:
        """Add *url* to the processing queue.

        Args:
            url: The URL to enqueue.
            priority: Job priority override. Falls back to config default.

        Returns:
            The job ID string for the newly created job.

        """
        effective_priority = priority if priority is not None else self.config.default_priority
        job = add_job(url=url, priority=effective_priority)
        return job.id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(result: HandleResult) -> str:
        """Build a human-readable summary string from *result*."""
        parts: list[str] = []

        if result.queued:
            noun = "URL" if result.queued == 1 else "URLs"
            parts.append(f"Queued {result.queued} {noun}")

        if result.duplicates:
            noun = "duplicate" if result.duplicates == 1 else "duplicates"
            parts.append(f"{result.duplicates} {noun} skipped")

        if not parts:
            return "No URLs queued"

        return ", ".join(parts)
