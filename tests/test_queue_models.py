"""Tests for queue models."""

from datetime import UTC, datetime

import pytest

from personal_knowledge_base.queue.models import Job


class TestJob:
    """Test cases for the Job dataclass."""

    def test_job_creation_defaults(self) -> None:
        """Test creating a job with default values."""
        job = Job(id="test-id", url="https://example.com")

        assert job.id == "test-id"
        assert job.url == "https://example.com"
        assert job.priority == 2
        assert job.status == "pending"
        assert job.content_type is None
        assert job.kb_name is None
        assert job.retry_count == 0
        assert job.error_message is None
        assert isinstance(job.created_at, datetime)
        assert isinstance(job.updated_at, datetime)

    def test_job_creation_custom_values(self) -> None:
        """Test creating a job with custom values."""
        created = datetime.now(UTC)
        updated = datetime.now(UTC)

        job = Job(
            id="custom-id",
            url="https://youtube.com/watch",
            priority=1,
            status="processing",
            content_type="youtube",
            kb_name="my_kb",
            created_at=created,
            updated_at=updated,
            retry_count=2,
            error_message="Previous error",
        )

        assert job.id == "custom-id"
        assert job.url == "https://youtube.com/watch"
        assert job.priority == 1
        assert job.status == "processing"
        assert job.content_type == "youtube"
        assert job.kb_name == "my_kb"
        assert job.created_at == created
        assert job.updated_at == updated
        assert job.retry_count == 2
        assert job.error_message == "Previous error"

    def test_job_invalid_priority(self) -> None:
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="Invalid priority"):
            Job(id="test", url="https://example.com", priority=3)

        with pytest.raises(ValueError, match="Invalid priority"):
            Job(id="test", url="https://example.com", priority=0)

    def test_job_invalid_status(self) -> None:
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            Job(id="test", url="https://example.com", status="unknown")

    def test_job_valid_statuses(self) -> None:
        """Test that valid statuses are accepted."""
        for status in ("pending", "processing", "done", "failed"):
            job = Job(id=f"test-{status}", url="https://example.com", status=status)
            assert job.status == status

    def test_job_invalid_content_type(self) -> None:
        """Test that invalid content_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid content_type"):
            Job(id="test", url="https://example.com", content_type="invalid")

    def test_job_valid_content_types(self) -> None:
        """Test that valid content types are accepted."""
        valid_types = (None, "youtube", "pdf", "article", "image", "code")
        for content_type in valid_types:
            job = Job(
                id=f"test-{content_type or 'none'}",
                url="https://example.com",
                content_type=content_type,
            )
            assert job.content_type == content_type

    def test_job_priority_values(self) -> None:
        """Test that priority 1 and 2 are accepted."""
        job1 = Job(id="test-1", url="https://example.com", priority=1)
        assert job1.priority == 1

        job2 = Job(id="test-2", url="https://example.com", priority=2)
        assert job2.priority == 2
