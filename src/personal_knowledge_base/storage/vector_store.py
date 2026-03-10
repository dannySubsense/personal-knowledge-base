"""Qdrant vector store integration for storing and searching embeddings."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store.

    Attributes:
        qdrant_url: URL of the Qdrant server.
        collection_name: Name of the collection to use.
        vector_size: Dimensionality of the embedding vectors.
        distance: Distance metric for similarity search.
        batch_size: Number of vectors to upsert in a single batch.
        max_retries: Maximum number of retries for failed operations.
    """

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "knowledge_base"
    vector_size: int = 768
    distance: models.Distance = models.Distance.COSINE
    batch_size: int = 100
    max_retries: int = 3


@dataclass
class SearchResult:
    """Result from a vector similarity search.

    Attributes:
        score: Similarity score (higher is better).
        chunk_text: The text content of the matching chunk.
        metadata: Additional metadata about the chunk.
    """

    score: float
    chunk_text: str
    metadata: dict[str, Any]


class VectorStore:
    """Qdrant-based vector store for embeddings.

    This class provides:
    - Collection management (create if not exists)
    - Batch upsert with retry logic
    - Similarity search with filters
    - Source-based deletion
    - Collection statistics
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        """Initialize the vector store.

        Args:
            config: Vector store configuration. Uses defaults if not provided.
        """
        self.config = config or VectorStoreConfig()
        self.client: QdrantClient | None = None

    def connect(self) -> None:
        """Connect to the Qdrant server.

        Raises:
            ConnectionError: If unable to connect to Qdrant.
        """
        if self.client is not None:
            return

        try:
            self.client = QdrantClient(url=self.config.qdrant_url)
            # Test connection
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.config.qdrant_url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the Qdrant server."""
        if self.client is not None:
            self.client.close()
            self.client = None
            logger.info("Disconnected from Qdrant")

    def _get_client(self) -> QdrantClient:
        """Get the Qdrant client, connecting if necessary.

        Returns:
            The Qdrant client instance.

        Raises:
            RuntimeError: If not connected to Qdrant.
        """
        if self.client is None:
            self.connect()
        if self.client is None:
            raise RuntimeError("Not connected to Qdrant")
        return self.client

    def ensure_collection(self) -> bool:
        """Ensure the collection exists, creating it if necessary.

        Creates the collection with:
        - Proper vector parameters (size, distance metric)
        - Payload indices on source, doc_id, and chunk_index for efficient filtering

        Returns:
            True if collection was created, False if it already existed.

        Raises:
            RuntimeError: If collection creation fails after max retries.
        """
        client = self._get_client()

        try:
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.collection_name in collection_names:
                logger.debug(f"Collection '{self.config.collection_name}' already exists")
                return False

            # Create collection with vector parameters
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.vector_size,
                    distance=self.config.distance,
                ),
            )

            # Create payload indices for efficient filtering
            client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="chunk_index",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

            logger.info(f"Created collection '{self.config.collection_name}'")
            return True

        except UnexpectedResponse as e:
            # Collection might have been created by another process
            if e.status_code == 409:  # Conflict
                logger.debug(f"Collection '{self.config.collection_name}' already exists")
                return False
            raise RuntimeError(f"Failed to create collection: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to ensure collection: {e}") from e

    def upsert_embeddings(
        self,
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
        show_progress: bool = False,
    ) -> list[str]:
        """Upsert embeddings into the vector store.

        Args:
            embeddings: List of embedding vectors.
            payloads: List of payload dictionaries (must include 'chunk_text').
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
            show_progress: Whether to log progress during upsert.

        Returns:
            List of IDs for the upserted vectors.

        Raises:
            ValueError: If embeddings and payloads have different lengths.
            RuntimeError: If upsert fails after max retries.
        """
        if len(embeddings) != len(payloads):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and payloads ({len(payloads)}) "
                "must have the same length"
            )

        if not embeddings:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        elif len(ids) != len(embeddings):
            raise ValueError(
                f"IDs ({len(ids)}) and embeddings ({len(embeddings)}) " "must have the same length"
            )

        client = self._get_client()

        # Prepare points
        points = [
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(ids, embeddings, payloads, strict=True)
        ]

        # Upsert in batches with retry logic
        total_upserted = 0
        for batch_start in range(0, len(points), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(points))
            batch = points[batch_start:batch_end]

            for attempt in range(self.config.max_retries):
                try:
                    client.upsert(
                        collection_name=self.config.collection_name,
                        points=batch,
                    )
                    total_upserted += len(batch)

                    if show_progress:
                        logger.info(
                            f"Upserted batch {batch_start // self.config.batch_size + 1}/"
                            f"{(len(points) - 1) // self.config.batch_size + 1} "
                            f"({len(batch)} points)"
                        )

                    break  # Success, move to next batch

                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise RuntimeError(
                            f"Failed to upsert batch after {self.config.max_retries} attempts: {e}"
                        ) from e
                    logger.warning(f"Upsert attempt {attempt + 1} failed, retrying: {e}")

        logger.info(f"Successfully upserted {total_upserted} vectors")
        return ids

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        source_filter: str | None = None,
        doc_id_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: The query embedding vector.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            source_filter: Optional filter for source file (supports wildcards).
            doc_id_filter: Optional filter for specific document ID.

        Returns:
            List of search results sorted by similarity score.

        Raises:
            RuntimeError: If search fails.
        """
        client = self._get_client()

        # Build filter
        filter_conditions: list[models.Condition] = []

        if source_filter:
            if "*" in source_filter or "?" in source_filter:
                # Use wildcard match for patterns
                filter_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchText(text=source_filter.replace("*", "")),
                    )
                )
            else:
                filter_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter),
                    )
                )

        if doc_id_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id_filter),
                )
            )

        search_filter = None
        if filter_conditions:
            search_filter = models.Filter(must=filter_conditions)

        try:
            results = client.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
            )

            return [
                SearchResult(
                    score=result.score,
                    chunk_text=result.payload.get("chunk_text", "") if result.payload else "",
                    metadata={k: v for k, v in (result.payload or {}).items() if k != "chunk_text"},
                )
                for result in results.points
            ]

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

    def delete_by_source(self, source: str) -> int:
        """Delete all vectors from a specific source.

        Args:
            source: The source file path or identifier.

        Returns:
            Number of vectors deleted.

        Raises:
            RuntimeError: If deletion fails.
        """
        client = self._get_client()

        try:
            # First, count how many points will be deleted
            count_result = client.count(
                collection_name=self.config.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                ),
            )
            count = count_result.count

            if count == 0:
                logger.debug(f"No vectors found for source: {source}")
                return 0

            # Delete the points
            client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source),
                            )
                        ]
                    )
                ),
            )

            logger.info(f"Deleted {count} vectors for source: {source}")
            return count

        except Exception as e:
            raise RuntimeError(f"Failed to delete by source: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary containing collection statistics:
            - vector_count: Total number of vectors
            - indexed_vectors_count: Number of indexed vectors
            - segments_count: Number of segments
            - disk_data_size: Size of data on disk
            - ram_data_size: Size of data in RAM
            - config: Collection configuration

        Raises:
            RuntimeError: If stats retrieval fails.
        """
        client = self._get_client()

        try:
            collection_info = client.get_collection(self.config.collection_name)
            stats = {
                "vector_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": self.config.vector_size,
                    "distance": self.config.distance.value,
                    "collection_name": self.config.collection_name,
                },
            }

            stats["disk_data_size"] = None
            stats["ram_data_size"] = None

            return stats

        except Exception as e:
            raise RuntimeError(f"Failed to get stats: {e}") from e

    def __enter__(self) -> VectorStore:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()
