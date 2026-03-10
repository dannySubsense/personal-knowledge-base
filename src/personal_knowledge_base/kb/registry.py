"""SQLite-backed registry of topic-based knowledge bases."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeBase:
    """A topic-scoped knowledge base managed by KBRegistry.

    Attributes:
        id: Slug identifier, e.g. ``"quant-trading"``.
        name: Human-readable display name, e.g. ``"Quant Trading"``.
        description: Plain-text description used for embedding similarity routing.
        parent_id: Optional parent KB id for sub-KBs.
        created_at: UTC timestamp of creation.
        max_documents: Maximum number of documents this KB may hold.
        document_count: Current number of ingested documents.
        auto_subcategory: If ``True``, automatically create sub-categories on overflow.
        centroid_vector: Running mean of document embeddings; updated after each ingest.

    """

    id: str
    name: str
    description: str
    parent_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    max_documents: int = 50
    document_count: int = 0
    auto_subcategory: bool = False
    centroid_vector: list[float] | None = None


# ---------------------------------------------------------------------------
# Default KB seeds
# ---------------------------------------------------------------------------

_DEFAULT_KBS: list[dict[str, Any]] = [
    {
        "id": "quant-trading",
        "name": "Quant Trading",
        "description": (
            "Quantitative finance, trading strategies, factor models, risk management, "
            "portfolio construction, backtesting, alpha generation"
        ),
    },
    {
        "id": "ml-ai",
        "name": "ML & AI",
        "description": (
            "Machine learning, deep learning, large language models, transformers, "
            "neural networks, AI research, computer vision, NLP"
        ),
    },
    {
        "id": "general",
        "name": "General",
        "description": (
            "General web content, articles, blog posts, miscellaneous topics not "
            "matching other knowledge bases"
        ),
    },
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    parent_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    max_documents INTEGER DEFAULT 50,
    document_count INTEGER DEFAULT 0,
    auto_subcategory BOOLEAN DEFAULT FALSE,
    centroid_vector TEXT,
    FOREIGN KEY (parent_id) REFERENCES knowledge_bases(id)
);
"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class KBRegistry:
    """SQLite-backed registry of :class:`KnowledgeBase` instances.

    Parameters:
        db_path: Path to the SQLite database file.  ``~`` is expanded
            automatically.  The parent directory is created if absent.

    """

    def __init__(self, db_path: str = "~/pkb-data/pkb_metadata.db") -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create the ``knowledge_bases`` table if it does not yet exist."""
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    @staticmethod
    def _row_to_kb(row: sqlite3.Row) -> KnowledgeBase:
        """Convert a database row to a :class:`KnowledgeBase` instance."""
        centroid: list[float] | None = None
        if row["centroid_vector"] is not None:
            centroid = json.loads(row["centroid_vector"])

        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return KnowledgeBase(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            parent_id=row["parent_id"],
            created_at=created_at,
            max_documents=row["max_documents"],
            document_count=row["document_count"],
            auto_subcategory=bool(row["auto_subcategory"]),
            centroid_vector=centroid,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_kb(
        self,
        id: str,
        name: str,
        description: str,
        parent_id: str | None = None,
        max_documents: int = 50,
        auto_subcategory: bool = False,
    ) -> KnowledgeBase:
        """Create and persist a new knowledge base.

        Args:
            id: Unique slug identifier (e.g. ``"quant-trading"``).
            name: Human-readable display name.
            description: Plain-text description for routing similarity.
            parent_id: Optional parent KB id for hierarchical organisation.
            max_documents: Capacity cap.
            auto_subcategory: Automatically sub-categorise on overflow.

        Returns:
            The newly created :class:`KnowledgeBase`.

        Raises:
            ValueError: If a KB with *id* already exists.

        """
        if self.get_kb(id) is not None:
            raise ValueError(f"Knowledge base with id '{id}' already exists.")

        kb = KnowledgeBase(
            id=id,
            name=name,
            description=description,
            parent_id=parent_id,
            max_documents=max_documents,
            auto_subcategory=auto_subcategory,
        )

        self._conn.execute(
            """
            INSERT INTO knowledge_bases
                (id, name, description, parent_id, created_at,
                 max_documents, document_count, auto_subcategory, centroid_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kb.id,
                kb.name,
                kb.description,
                kb.parent_id,
                kb.created_at.isoformat(),
                kb.max_documents,
                kb.document_count,
                int(kb.auto_subcategory),
                None,
            ),
        )
        self._conn.commit()
        return kb

    def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        """Retrieve a knowledge base by id.

        Args:
            kb_id: The slug id to look up.

        Returns:
            The matching :class:`KnowledgeBase`, or ``None`` if not found.

        """
        row = self._conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
        return self._row_to_kb(row) if row else None

    def list_kbs(self, include_sub: bool = True) -> list[KnowledgeBase]:
        """Return all registered knowledge bases.

        Args:
            include_sub: When ``False``, only top-level KBs (``parent_id IS NULL``)
                are returned.

        Returns:
            List of :class:`KnowledgeBase` instances, ordered by ``created_at``.

        """
        if include_sub:
            rows = self._conn.execute(
                "SELECT * FROM knowledge_bases ORDER BY created_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM knowledge_bases WHERE parent_id IS NULL ORDER BY created_at"
            ).fetchall()
        return [self._row_to_kb(r) for r in rows]

    def update_centroid(self, kb_id: str, centroid: list[float]) -> None:
        """Persist a new centroid vector for the given knowledge base.

        Args:
            kb_id: The slug id of the target KB.
            centroid: Embedding centroid as a list of floats.

        Raises:
            ValueError: If no KB with *kb_id* exists.

        """
        if self.get_kb(kb_id) is None:
            raise ValueError(f"Knowledge base '{kb_id}' not found.")

        self._conn.execute(
            "UPDATE knowledge_bases SET centroid_vector = ? WHERE id = ?",
            (json.dumps(centroid), kb_id),
        )
        self._conn.commit()

    def increment_doc_count(self, kb_id: str, delta: int = 1) -> None:
        """Increment the document count for a knowledge base.

        Args:
            kb_id: The slug id of the target KB.
            delta: Amount to add to ``document_count`` (default ``1``).

        Raises:
            ValueError: If no KB with *kb_id* exists.

        """
        if self.get_kb(kb_id) is None:
            raise ValueError(f"Knowledge base '{kb_id}' not found.")

        self._conn.execute(
            "UPDATE knowledge_bases SET document_count = document_count + ? WHERE id = ?",
            (delta, kb_id),
        )
        self._conn.commit()

    def delete_kb(self, kb_id: str) -> None:
        """Remove a knowledge base from the registry.

        Args:
            kb_id: The slug id of the KB to delete.

        Raises:
            ValueError: If no KB with *kb_id* exists.

        """
        if self.get_kb(kb_id) is None:
            raise ValueError(f"Knowledge base '{kb_id}' not found.")

        self._conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        self._conn.commit()

    def seed_defaults(self) -> None:
        """Seed the registry with default knowledge bases if it is empty.

        Idempotent — calling this method multiple times has no additional effect
        once the default KBs exist.

        Default KBs created:
        - ``quant-trading`` — Quantitative finance and trading
        - ``ml-ai`` — Machine learning and AI research
        - ``general`` — Miscellaneous web content

        """
        existing_ids = {kb.id for kb in self.list_kbs()}
        for spec in _DEFAULT_KBS:
            if spec["id"] not in existing_ids:
                self.create_kb(
                    id=spec["id"],
                    name=spec["name"],
                    description=spec["description"],
                )
