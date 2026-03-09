"""Tests for queue CRUD operations."""

from datetime import datetime
from pathlib import Path

import pytest
from personal_knowledge_base.queue.db import close_connection, init_db
from personal_knowledge_base.queue.models import Job
from personal_knowledge_base.queue.operations import (
    add_job,
    delete_job,
    get_job_by_id,
    get_next_job,
    list_jobs,
    retry_job,
    update_job_status,
)


class TestQueueOperations:
    """Test cases for queue CRUD operations."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database path and initialize it."""
        db_path = tmp_path / "test_queue.db"
        init_db(db_path)
        return db_path

    @pytest.fixture(autouse=True)
    def cleanup_connections(self) -> None:
        """Clean up connections after each test."""
        yield
        close_connection()

    def test_add_job(self, temp_db: Path) -> None:
        """Test adding a job to the queue."""
        job = add_job(
            url="https://example.com/article",
            priority=1,
            content_type="article",
            kb_name="test_kb",
        )

        assert isinstance(job, Job)
        assert job.url == "https://example.com/article"
        assert job.priority == 1
        assert job.status == "pending"
        assert job.content_type == "article"
        assert job.kb_name == "test_kb"
        assert job.retry_count == 0
        assert job.error_message is None

    def test_add_job_defaults(self, temp_db: Path) -> None:
        """Test adding a job with default values."""
        job = add_job(url="https://example.com")

        assert job.priority == 2
        assert job.status == "pending"
        assert job.content_type is None
        assert job.kb_name is None

    def test_get_next_job_empty_queue(self, temp_db: Path) -> None:
        """Test getting next job from empty queue."""
        job = get_next_job()
        assert job is None

    def test_get_next_job_single(self, temp_db: Path) -> None:
        """Test getting next job with single job."""
        added = add_job(url="https://example.com")

        next_job = get_next_job()
        assert next_job is not None
        assert next_job.id == added.id
        assert next_job.url == added.url

    def test_get_next_job_priority_ordering(self, temp_db: Path) -> None:
        """Test that higher priority jobs are returned first."""
        # Add normal priority job first
        add_job(url="https://example.com/normal", priority=2)

        # Add immediate priority job second
        immediate_job = add_job(url="https://example.com/immediate", priority=1)

        # Should get immediate priority first
        next_job = get_next_job()
        assert next_job is not None
        assert next_job.id == immediate_job.id

    def test_get_next_job_fifo_within_priority(self, temp_db: Path) -> None:
        """Test FIFO ordering within same priority."""
        job1 = add_job(url="https://example.com/1", priority=2)
        add_job(url="https://example.com/2", priority=2)

        next_job = get_next_job()
        assert next_job is not None
        assert next_job.id == job1.id

    def test_get_next_job_with_priority_filter(self, temp_db: Path) -> None:
        """Test getting next job with priority filter."""
        immediate_job = add_job(url="https://example.com/immediate", priority=1)
        add_job(url="https://example.com/normal", priority=2)

        # Filter for priority 2 only
        next_job = get_next_job(priority=2)
        assert next_job is not None
        assert next_job.priority == 2

        # Filter for priority 1
        next_job = get_next_job(priority=1)
        assert next_job is not None
        assert next_job.id == immediate_job.id

    def test_get_next_job_skips_non_pending(self, temp_db: Path) -> None:
        """Test that non-pending jobs are skipped."""
        job = add_job(url="https://example.com")
        update_job_status(job.id, "processing")

        next_job = get_next_job()
        assert next_job is None

    def test_update_job_status(self, temp_db: Path) -> None:
        """Test updating job status."""
        job = add_job(url="https://example.com")

        update_job_status(job.id, "processing")

        updated = get_job_by_id(job.id)
        assert updated is not None
        assert updated.status == "processing"
        assert updated.updated_at > job.updated_at

    def test_update_job_status_with_error(self, temp_db: Path) -> None:
        """Test updating job status with error message."""
        job = add_job(url="https://example.com")

        update_job_status(job.id, "failed", error_message="Network error")

        updated = get_job_by_id(job.id)
        assert updated is not None
        assert updated.status == "failed"
        assert updated.error_message == "Network error"

    def test_update_job_status_invalid(self, temp_db: Path) -> None:
        """Test that invalid status raises ValueError."""
        job = add_job(url="https://example.com")

        with pytest.raises(ValueError, match="Invalid status"):
            update_job_status(job.id, "invalid_status")

    def test_get_job_by_id(self, temp_db: Path) -> None:
        """Test getting job by ID."""
        job = add_job(url="https://example.com")

        found = get_job_by_id(job.id)
        assert found is not None
        assert found.id == job.id
        assert found.url == job.url

    def test_get_job_by_id_not_found(self, temp_db: Path) -> None:
        """Test getting non-existent job."""
        found = get_job_by_id("non-existent-id")
        assert found is None

    def test_list_jobs(self, temp_db: Path) -> None:
        """Test listing all jobs."""
        job1 = add_job(url="https://example.com/1")
        job2 = add_job(url="https://example.com/2")

        jobs = list_jobs()
        assert len(jobs) == 2
        # Most recent first
        assert jobs[0].id == job2.id
        assert jobs[1].id == job1.id

    def test_list_jobs_with_status_filter(self, temp_db: Path) -> None:
        """Test listing jobs with status filter."""
        pending_job = add_job(url="https://example.com/pending")
        done_job = add_job(url="https://example.com/done")
        update_job_status(done_job.id, "done")

        pending_jobs = list_jobs(status="pending")
        assert len(pending_jobs) == 1
        assert pending_jobs[0].id == pending_job.id

        done_jobs = list_jobs(status="done")
        assert len(done_jobs) == 1
        assert done_jobs[0].id == done_job.id

    def test_list_jobs_limit(self, temp_db: Path) -> None:
        """Test listing jobs with limit."""
        for i in range(5):
            add_job(url=f"https://example.com/{i}")

        jobs = list_jobs(limit=3)
        assert len(jobs) == 3

    def test_retry_job(self, temp_db: Path) -> None:
        """Test retrying a failed job."""
        job = add_job(url="https://example.com")
        update_job_status(job.id, "failed", error_message="Error")

        result = retry_job(job.id)
        assert result is True

        retried = get_job_by_id(job.id)
        assert retried is not None
        assert retried.status == "pending"
        assert retried.retry_count == 1
        assert retried.error_message is None

    def test_retry_job_not_found(self, temp_db: Path) -> None:
        """Test retrying non-existent job."""
        result = retry_job("non-existent-id")
        assert result is False

    def test_retry_job_multiple_times(self, temp_db: Path) -> None:
        """Test retrying a job multiple times."""
        job = add_job(url="https://example.com")

        for i in range(3):
            update_job_status(job.id, "failed", error_message=f"Error {i}")
            retry_job(job.id)

        retried = get_job_by_id(job.id)
        assert retried is not None
        assert retried.retry_count == 3
        assert retried.status == "pending"

    def test_delete_job(self, temp_db: Path) -> None:
        """Test deleting a job."""
        job = add_job(url="https://example.com")

        result = delete_job(job.id)
        assert result is True

        found = get_job_by_id(job.id)
        assert found is None

    def test_delete_job_not_found(self, temp_db: Path) -> None:
        """Test deleting non-existent job."""
        result = delete_job("non-existent-id")
        assert result is False

    def test_job_persistence(self, temp_db: Path) -> None:
        """Test that jobs are persisted correctly."""
        job = add_job(
            url="https://youtube.com/watch",
            priority=1,
            content_type="youtube",
            kb_name="videos",
        )

        # Retrieve and verify all fields
        found = get_job_by_id(job.id)
        assert found is not None
        assert found.id == job.id
        assert found.url == "https://youtube.com/watch"
        assert found.priority == 1
        assert found.status == "pending"
        assert found.content_type == "youtube"
        assert found.kb_name == "videos"
        assert found.retry_count == 0
        assert isinstance(found.created_at, datetime)
        assert isinstance(found.updated_at, datetime)
