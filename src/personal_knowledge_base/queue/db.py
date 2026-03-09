"""SQLite database operations for the queue system."""

import sqlite3
import threading
from pathlib import Path

# Thread-local storage for database connections
_local = threading.local()

DEFAULT_DB_PATH = Path.home() / ".personal_knowledge_base" / "queue.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a database connection with row factory.

    Args:
        db_path: Path to the SQLite database file. If None, uses default path.

    Returns:
        SQLite connection with row factory set to sqlite3.Row.

    """
    path = db_path or DEFAULT_DB_PATH

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get thread-local connection or create new one
    if not hasattr(_local, "connection") or _local.connection is None:
        _local.connection = sqlite3.connect(str(path))
        _local.connection.row_factory = sqlite3.Row

    conn: sqlite3.Connection = _local.connection
    return conn


def close_connection() -> None:
    """Close the thread-local database connection."""
    if hasattr(_local, "connection") and _local.connection is not None:
        _local.connection.close()
        _local.connection = None


def init_db(db_path: Path | None = None) -> None:
    """Initialize the database with required tables.

    Args:
        db_path: Path to the SQLite database file. If None, uses default path.

    """
    conn = get_connection(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 2,
            status TEXT NOT NULL DEFAULT 'pending',
            content_type TEXT,
            kb_name TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        )
    """)

    # Create indexes for efficient querying
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status_priority ON jobs(status, priority)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)
    """)

    conn.commit()
