"""Tests for queue database operations."""

import sqlite3
import threading
from pathlib import Path

import pytest

from personal_knowledge_base.queue.db import (
    close_connection,
    get_connection,
    init_db,
)


class TestDatabase:
    """Test cases for database operations."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database path."""
        return tmp_path / "test_queue.db"

    @pytest.fixture(autouse=True)
    def cleanup_connections(self) -> None:
        """Clean up connections after each test."""
        yield
        close_connection()

    def test_get_connection_creates_db(self, temp_db: Path) -> None:
        """Test that get_connection creates the database file."""
        assert not temp_db.exists()
        conn = get_connection(temp_db)
        assert conn is not None
        assert temp_db.exists()

    def test_get_connection_returns_same_connection(self, temp_db: Path) -> None:
        """Test that get_connection returns the same connection for same thread."""
        conn1 = get_connection(temp_db)
        conn2 = get_connection(temp_db)
        assert conn1 is conn2

    def test_get_connection_row_factory(self, temp_db: Path) -> None:
        """Test that connection has row factory set."""
        conn = get_connection(temp_db)
        assert conn.row_factory is sqlite3.Row

    def test_init_db_creates_tables(self, temp_db: Path) -> None:
        """Test that init_db creates the jobs table."""
        init_db(temp_db)
        conn = get_connection(temp_db)

        # Check that jobs table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        assert cursor.fetchone() is not None

    def test_init_db_creates_indexes(self, temp_db: Path) -> None:
        """Test that init_db creates indexes."""
        init_db(temp_db)
        conn = get_connection(temp_db)

        # Check that indexes exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_jobs%'"
        )
        indexes = {row["name"] for row in cursor.fetchall()}

        expected_indexes = {
            "idx_jobs_status",
            "idx_jobs_priority",
            "idx_jobs_status_priority",
            "idx_jobs_created_at",
        }
        assert expected_indexes.issubset(indexes)

    def test_init_db_idempotent(self, temp_db: Path) -> None:
        """Test that init_db can be called multiple times without error."""
        init_db(temp_db)
        init_db(temp_db)  # Should not raise

        # Verify table still exists
        conn = get_connection(temp_db)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        assert cursor.fetchone() is not None

    def test_close_connection(self, temp_db: Path) -> None:
        """Test that close_connection closes the connection."""
        conn1 = get_connection(temp_db)
        close_connection()
        conn2 = get_connection(temp_db)

        # After closing, a new connection should be created
        assert conn1 is not conn2

    def test_connection_thread_safety(self, temp_db: Path) -> None:
        """Test that connections are thread-local."""
        connections = []

        def get_conn() -> None:
            conn = get_connection(temp_db)
            connections.append(conn)

        thread1 = threading.Thread(target=get_conn)
        thread2 = threading.Thread(target=get_conn)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Different threads should have different connections
        assert connections[0] is not connections[1]
