"""Query interface for semantic search across the personal knowledge base."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from personal_knowledge_base.processing.embedder import EmbedderConfig, OllamaEmbedder
from personal_knowledge_base.storage.vector_store import VectorStore, VectorStoreConfig


@dataclass
class QueryConfig:
    """Configuration for the KB query interface.

    Attributes:
        qdrant_url: URL of the Qdrant server.
        ollama_url: Base URL for the Ollama API.
        embedding_model: Name of the Ollama model to use for embeddings.
        default_limit: Default number of results to return.
        default_score_threshold: Minimum similarity score for results.
        default_collection: Default collection to search.
    """

    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    default_limit: int = 10
    default_score_threshold: float = 0.3
    default_collection: str = "general"


@dataclass
class QueryResult:
    """A single result from a KB query.

    Attributes:
        chunk_text: The text content of the matching chunk.
        score: Similarity score (higher is better).
        source: Source file or identifier for the chunk.
        doc_id: Document identifier.
        collection: The collection this result came from.
        metadata: Additional metadata about the chunk.
    """

    chunk_text: str
    score: float
    source: str
    doc_id: str
    collection: str
    metadata: dict[str, Any] = field(default_factory=dict)


class KBQueryInterface:
    """Semantic search interface for the personal knowledge base.

    Provides methods to query single or multiple collections and format
    results as readable text suitable for use in conversations.
    """

    DEFAULT_COLLECTIONS: list[str] = ["videos", "papers", "code", "general"]

    def __init__(self, config: QueryConfig | None = None) -> None:
        """Initialize the query interface.

        Args:
            config: Query configuration. Uses defaults if not provided.
        """
        self.config = config or QueryConfig()

    def _build_vector_store(self, collection: str) -> VectorStore:
        """Build a VectorStore instance for the given collection.

        Args:
            collection: Collection name.

        Returns:
            Configured VectorStore instance.
        """
        vs_config = VectorStoreConfig(
            qdrant_url=self.config.qdrant_url,
            collection_name=collection,
        )
        return VectorStore(config=vs_config)

    def _build_embedder(self) -> OllamaEmbedder:
        """Build an OllamaEmbedder instance from config.

        Returns:
            Configured OllamaEmbedder instance.
        """
        embedder_config = EmbedderConfig(
            model=self.config.embedding_model,
            ollama_url=self.config.ollama_url,
        )
        return OllamaEmbedder(config=embedder_config)

    async def _embed_question(self, question: str) -> list[float]:
        """Embed the query question using Ollama.

        Args:
            question: The question or query text.

        Returns:
            Embedding vector for the question.
        """
        embedder = self._build_embedder()
        async with embedder:
            return await embedder.embed_text(question)

    def _search_collection(
        self,
        query_vector: list[float],
        collection: str,
        limit: int,
        score_threshold: float,
    ) -> list[QueryResult]:
        """Search a single collection and return QueryResult objects.

        Args:
            query_vector: The embedding vector for the query.
            collection: Collection name to search.
            limit: Maximum number of results.
            score_threshold: Minimum score threshold.

        Returns:
            List of QueryResult objects sorted by score descending.
        """
        store = self._build_vector_store(collection)
        try:
            store.connect()
            search_results = store.search(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
        finally:
            store.disconnect()

        results: list[QueryResult] = []
        for sr in search_results:
            results.append(
                QueryResult(
                    chunk_text=sr.chunk_text,
                    score=sr.score,
                    source=sr.metadata.get("source", ""),
                    doc_id=sr.metadata.get("doc_id", ""),
                    collection=collection,
                    metadata={
                        k: v for k, v in sr.metadata.items() if k not in ("source", "doc_id")
                    },
                )
            )
        return results

    def query(
        self,
        question: str,
        collection: str | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[QueryResult]:
        """Semantic search across a single KB collection.

        1. Embeds the question using OllamaEmbedder.
        2. Searches VectorStore for similar chunks.
        3. Returns ranked QueryResult list.

        Args:
            question: The question or query text.
            collection: Collection to search. Uses default_collection if not provided.
            limit: Maximum number of results. Uses default_limit if not provided.
            score_threshold: Minimum score. Uses default_score_threshold if not provided.

        Returns:
            List of QueryResult objects sorted by score descending.
        """
        collection = collection or self.config.default_collection
        limit = limit if limit is not None else self.config.default_limit
        score_threshold = (
            score_threshold if score_threshold is not None else self.config.default_score_threshold
        )

        query_vector = asyncio.run(self._embed_question(question))
        return self._search_collection(query_vector, collection, limit, score_threshold)

    def query_all_collections(
        self,
        question: str,
        collections: list[str] | None = None,
        limit_per_collection: int = 5,
    ) -> list[QueryResult]:
        """Search across multiple collections, merge and re-rank by score.

        Args:
            question: The question or query text.
            collections: Collections to search. Defaults to
                ["videos", "papers", "code", "general"].
            limit_per_collection: Maximum results per collection.

        Returns:
            Merged list of QueryResult objects sorted by score descending.
        """
        if collections is None:
            collections = list(self.DEFAULT_COLLECTIONS)

        query_vector = asyncio.run(self._embed_question(question))

        all_results: list[QueryResult] = []
        for collection in collections:
            try:
                results = self._search_collection(
                    query_vector,
                    collection,
                    limit_per_collection,
                    self.config.default_score_threshold,
                )
                all_results.extend(results)
            except Exception:
                # Skip collections that fail (e.g. don't exist yet)
                continue

        # Re-rank merged results by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results

    def format_results(self, results: list[QueryResult], max_chars: int = 2000) -> str:
        """Format query results as readable text for conversation context.

        Args:
            results: List of QueryResult objects to format.
            max_chars: Maximum character length for the output.

        Returns:
            Formatted string with source, score, and chunk_text snippet,
            truncated to max_chars.
        """
        if not results:
            return "No results found."

        parts: list[str] = []
        for i, result in enumerate(results, 1):
            entry = (
                f"[{i}] Source: {result.source or 'unknown'} | "
                f"Collection: {result.collection} | "
                f"Score: {result.score:.3f}\n"
                f"{result.chunk_text}"
            )
            parts.append(entry)

        full_text = "\n\n".join(parts)

        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]
            # Try to truncate at a word boundary
            last_space = full_text.rfind(" ")
            if last_space > max_chars - 100:
                full_text = full_text[:last_space]
            full_text += "..."

        return full_text
