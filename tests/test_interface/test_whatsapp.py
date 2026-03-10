"""Tests for the WhatsApp handler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from personal_knowledge_base.interface.whatsapp import (
    WhatsAppConfig,
    WhatsAppHandler,
)
from personal_knowledge_base.queue.models import Job

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRUSTED = "+16039885837"
OTHER = "+10000000000"
URL_A = "https://example.com/article"
URL_B = "https://openai.com/blog/gpt4"
URL_C = "https://github.com/owner/repo?tab=readme"


def _make_job(url: str, days_ago: int = 0) -> Job:
    """Create a minimal Job fixture."""
    created = datetime.now(UTC) - timedelta(days=days_ago)
    return Job(
        id="job-" + url[-8:].replace("/", "-"),
        url=url,
        priority=2,
        status="pending",
        created_at=created,
        updated_at=created,
    )


def _handler(trusted: list[str] | None = None, window: int = 30) -> WhatsAppHandler:
    cfg = WhatsAppConfig(
        trusted_senders=trusted if trusted is not None else [],
        duplicate_window_days=window,
    )
    return WhatsAppHandler(cfg)


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------


class TestExtractUrls:
    def test_single_url(self) -> None:
        h = _handler()
        assert h.extract_urls(f"Check this out: {URL_A}") == [URL_A]

    def test_multiple_urls(self) -> None:
        h = _handler()
        text = f"First {URL_A} and second {URL_B}"
        assert h.extract_urls(text) == [URL_A, URL_B]

    def test_url_at_start(self) -> None:
        h = _handler()
        assert URL_A in h.extract_urls(f"{URL_A} is the link")

    def test_url_at_end(self) -> None:
        h = _handler()
        assert URL_A in h.extract_urls(f"See {URL_A}")

    def test_url_with_query_and_fragment(self) -> None:
        h = _handler()
        url = "https://example.com/path?q=1&b=2#section"
        assert h.extract_urls(url) == [url]

    def test_no_urls(self) -> None:
        h = _handler()
        assert h.extract_urls("Hello world, no links here!") == []

    def test_deduplicates_same_url(self) -> None:
        h = _handler()
        text = f"{URL_A} and again {URL_A}"
        assert h.extract_urls(text) == [URL_A]

    def test_url_with_path(self) -> None:
        h = _handler()
        assert h.extract_urls(URL_C) == [URL_C]


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------


class TestIsDuplicate:
    @patch("personal_knowledge_base.interface.whatsapp.list_jobs")
    def test_no_jobs_not_duplicate(self, mock_list: MagicMock) -> None:
        mock_list.return_value = []
        assert _handler().is_duplicate(URL_A) is False

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs")
    def test_recent_job_is_duplicate(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [_make_job(URL_A, days_ago=5)]
        assert _handler(window=30).is_duplicate(URL_A) is True

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs")
    def test_old_job_not_duplicate(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [_make_job(URL_A, days_ago=31)]
        assert _handler(window=30).is_duplicate(URL_A) is False

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs")
    def test_different_url_not_duplicate(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [_make_job(URL_B, days_ago=1)]
        assert _handler().is_duplicate(URL_A) is False

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs")
    def test_job_at_window_boundary_is_duplicate(self, mock_list: MagicMock) -> None:
        # Exactly on the boundary (0 days ago) should be a duplicate
        mock_list.return_value = [_make_job(URL_A, days_ago=0)]
        assert _handler(window=30).is_duplicate(URL_A) is True


# ---------------------------------------------------------------------------
# queue_url
# ---------------------------------------------------------------------------


class TestQueueUrl:
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_returns_job_id(self, mock_add: MagicMock) -> None:
        mock_add.return_value = _make_job(URL_A)
        job_id = _handler().queue_url(URL_A)
        assert isinstance(job_id, str)
        mock_add.assert_called_once_with(url=URL_A, priority=2)

    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_custom_priority(self, mock_add: MagicMock) -> None:
        mock_add.return_value = _make_job(URL_A)
        _handler().queue_url(URL_A, priority=1)
        mock_add.assert_called_once_with(url=URL_A, priority=1)

    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_uses_config_default_priority(self, mock_add: MagicMock) -> None:
        cfg = WhatsAppConfig(default_priority=1)
        handler = WhatsAppHandler(cfg)
        mock_add.return_value = _make_job(URL_A)
        handler.queue_url(URL_A)
        mock_add.assert_called_once_with(url=URL_A, priority=1)


# ---------------------------------------------------------------------------
# handle_message — full integration (all DB mocked)
# ---------------------------------------------------------------------------


class TestHandleMessage:
    @patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[])
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_single_url_queued(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_add.return_value = _make_job(URL_A)
        result = _handler().handle_message(TRUSTED, f"Read {URL_A}")
        assert result.urls_found == 1
        assert result.queued == 1
        assert result.duplicates == 0
        assert result.rejected == 0
        assert len(result.job_ids) == 1

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[])
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_multiple_urls_queued(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_add.side_effect = [_make_job(URL_A), _make_job(URL_B)]
        result = _handler().handle_message(TRUSTED, f"{URL_A} and {URL_B}")
        assert result.urls_found == 2
        assert result.queued == 2
        assert len(result.job_ids) == 2

    @patch(
        "personal_knowledge_base.interface.whatsapp.list_jobs",
        return_value=[],
    )
    def test_no_urls_in_message(self, mock_list: MagicMock) -> None:
        result = _handler().handle_message(TRUSTED, "Hey, no links here!")
        assert result.urls_found == 0
        assert result.queued == 0
        assert result.message == "No URLs found"

    @patch(
        "personal_knowledge_base.interface.whatsapp.list_jobs",
    )
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_duplicate_skipped(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_list.return_value = [_make_job(URL_A, days_ago=1)]
        result = _handler().handle_message(TRUSTED, f"Seen it {URL_A}")
        assert result.duplicates == 1
        assert result.queued == 0
        mock_add.assert_not_called()

    @patch(
        "personal_knowledge_base.interface.whatsapp.list_jobs",
    )
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_one_new_one_duplicate(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_list.return_value = [_make_job(URL_A, days_ago=1)]
        mock_add.return_value = _make_job(URL_B)
        result = _handler().handle_message(TRUSTED, f"{URL_A} {URL_B}")
        assert result.queued == 1
        assert result.duplicates == 1

    def test_untrusted_sender_rejected(self) -> None:
        result = _handler(trusted=[TRUSTED]).handle_message(OTHER, f"Check {URL_A}")
        assert result.rejected == 1
        assert result.queued == 0
        assert result.message == "Untrusted sender"

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[])
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_trusted_sender_allowed(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_add.return_value = _make_job(URL_A)
        result = _handler(trusted=[TRUSTED]).handle_message(TRUSTED, f"Link {URL_A}")
        assert result.queued == 1
        assert result.rejected == 0

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[])
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_open_mode_any_sender_allowed(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        """Empty trusted_senders list = open mode, any sender accepted."""
        mock_add.return_value = _make_job(URL_A)
        result = _handler(trusted=[]).handle_message(OTHER, f"Link {URL_A}")
        assert result.queued == 1
        assert result.rejected == 0

    def test_untrusted_rejected_count_equals_urls_found(self) -> None:
        result = _handler(trusted=[TRUSTED]).handle_message(OTHER, f"{URL_A} {URL_B}")
        assert result.rejected == 2
        assert result.urls_found == 2

    # ------------------------------------------------------------------
    # Summary message strings
    # ------------------------------------------------------------------

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[])
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_summary_single_queued(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_add.return_value = _make_job(URL_A)
        result = _handler().handle_message(TRUSTED, URL_A)
        assert result.message == "Queued 1 URL"

    @patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[])
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_summary_multiple_queued(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_add.side_effect = [_make_job(URL_A), _make_job(URL_B)]
        result = _handler().handle_message(TRUSTED, f"{URL_A} {URL_B}")
        assert result.message == "Queued 2 URLs"

    @patch(
        "personal_knowledge_base.interface.whatsapp.list_jobs",
    )
    @patch("personal_knowledge_base.interface.whatsapp.add_job")
    def test_summary_queued_and_duplicate(self, mock_add: MagicMock, mock_list: MagicMock) -> None:
        mock_list.return_value = [_make_job(URL_A, days_ago=1)]
        mock_add.return_value = _make_job(URL_B)
        result = _handler().handle_message(TRUSTED, f"{URL_A} {URL_B}")
        assert "1" in result.message
        assert "duplicate" in result.message

    @patch(
        "personal_knowledge_base.interface.whatsapp.list_jobs",
    )
    def test_summary_all_duplicates(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [_make_job(URL_A, days_ago=1)]
        result = _handler().handle_message(TRUSTED, URL_A)
        assert "duplicate" in result.message


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_config_used_when_none(self) -> None:
        h = WhatsAppHandler()
        assert h.config.default_priority == 2
        assert h.config.duplicate_window_days == 30
        assert h.config.trusted_senders == []
