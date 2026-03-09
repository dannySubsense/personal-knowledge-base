"""CRUD operations for the job queue."""

import sqlite3
import uuid
from datetime import UTC, datetime

from personal_knowledge_base.queue.db import get_connection
from personal_knowledge_base.queue.models import Job


def add_job(
    url: str,
    priority: int = 2,
    content_type: str | None = None,
    kb_name: str | None = None,
) -> Job:
    """Add a new job to the queue.

    Args:
        url: URL or path to the content to be ingested.
        priority: Priority level (1=immediate, 2=normal). Defaults to 2.
        content_type: Type of content (youtube, pdf, article, image, code).
        kb_name: Target knowledge base name.

    Returns:
        The created Job object.

    """
    job = Job(
        id=str(uuid.uuid4()),
        url=url,
        priority=priority,
        status="pending",
        content_type=content_type,
        kb_name=kb_name,
    )

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO jobs (id, url, priority, status, content_type, kb_name,
                         created_at, updated_at, retry_count, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job.id,
            job.url,
            job.priority,
            job.status,
            job.content_type,
            job.kb_name,
            job.created_at.isoformat(),
            job.updated_at.isoformat(),
            job.retry_count,
            job.error_message,
        ),
    )
    conn.commit()

    return job


def get_next_job(priority: int | None = None) -> Job | None:
    """Get the next job to process.

    Retrieves the highest priority pending job, using FIFO ordering
    within the same priority level.

    Args:
        priority: Optional priority filter. If None, considers all priorities.

    Returns:
        The next Job to process, or None if no pending jobs.

    """
    conn = get_connection()

    if priority is not None:
        cursor = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'pending' AND priority = ?
            ORDER BY priority ASC, created_at ASC
            LIMIT 1
            """,
            (priority,),
        )
    else:
        cursor = conn.execute("""
            SELECT * FROM jobs
            WHERE status = 'pending'
            ORDER BY priority ASC, created_at ASC
            LIMIT 1
            """)

    row = cursor.fetchone()
    if row is None:
        return None

    return _row_to_job(row)


def update_job_status(job_id: str, status: str, error_message: str | None = None) -> None:
    """Update the status of a job.

    Args:
        job_id: The ID of the job to update.
        status: The new status (pending, processing, done, failed).
        error_message: Optional error message if status is 'failed'.

    Raises:
        ValueError: If the status is invalid.

    """
    valid_statuses = ("pending", "processing", "done", "failed")
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}.")

    conn = get_connection()
    now = datetime.now(UTC).isoformat()

    if error_message is not None:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error_message, now, job_id),
        )
    else:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now, job_id),
        )

    conn.commit()


def get_job_by_id(job_id: str) -> Job | None:
    """Get a job by its ID.

    Args:
        job_id: The ID of the job to retrieve.

    Returns:
        The Job if found, or None.

    """
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()

    if row is None:
        return None

    return _row_to_job(row)


def list_jobs(status: str | None = None, limit: int = 100) -> list[Job]:
    """List jobs, optionally filtered by status.

    Args:
        status: Optional status filter.
        limit: Maximum number of jobs to return. Defaults to 100.

    Returns:
        List of Job objects.

    """
    conn = get_connection()

    if status is not None:
        cursor = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (status, limit),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cursor.fetchall()
    return [_row_to_job(row) for row in rows]


def retry_job(job_id: str) -> bool:
    """Retry a failed job.

    Increments the retry count and sets status back to pending.

    Args:
        job_id: The ID of the job to retry.

    Returns:
        True if the job was found and updated, False otherwise.

    """
    conn = get_connection()
    now = datetime.now(UTC).isoformat()

    cursor = conn.execute(
        """
        UPDATE jobs
        SET status = 'pending', retry_count = retry_count + 1,
            error_message = NULL, updated_at = ?
        WHERE id = ?
        """,
        (now, job_id),
    )
    conn.commit()

    result: bool = cursor.rowcount > 0
    return result


def delete_job(job_id: str) -> bool:
    """Delete a job from the queue.

    Args:
        job_id: The ID of the job to delete.

    Returns:
        True if the job was found and deleted, False otherwise.

    """
    conn = get_connection()
    cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()

    result: bool = cursor.rowcount > 0
    return result


def _row_to_job(row: sqlite3.Row) -> Job:
    """Convert a database row to a Job object.

    Args:
        row: SQLite row from the jobs table.

    Returns:
        Job object populated from the row.

    """
    return Job(
        id=row["id"],
        url=row["url"],
        priority=row["priority"],
        status=row["status"],
        content_type=row["content_type"],
        kb_name=row["kb_name"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        retry_count=row["retry_count"],
        error_message=row["error_message"],
    )
