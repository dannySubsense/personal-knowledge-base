"""Tests for vector store module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from personal_knowledge_base.storage.vector_store import (
    SearchResult,
    VectorStore,
    VectorStoreConfig,
)


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VectorStoreConfig()
        assert config.qdrant_url == "http://localhost:6333"
        assert config.collection_name == "knowledge_base"
        assert config.vector_size == 768
        assert config.distance == models.Distance.COSINE
        assert config.batch_size == 100
        assert config.max_retries == 3

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VectorStoreConfig(
            qdrant_url="http://qdrant:6333",
            collection_name="test_collection",
            vector_size=512,
            distance=models.Distance.EUCLID,
            batch_size=50,
            max_retries=5,
        )
        assert config.qdrant_url == "http://qdrant:6333"
        assert config.collection_name == "test_collection"
        assert config.vector_size == 512
        assert config.distance == models.Distance.EUCLID
        assert config.batch_size == 50
        assert config.max_retries == 5


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating a SearchResult."""
        result = SearchResult(
            score=0.95,
            chunk_text="Test chunk text",
            metadata={"source": "test.txt", "doc_id": "doc1"},
        )
        assert result.score == 0.95
        assert result.chunk_text == "Test chunk text"
        assert result.metadata == {"source": "test.txt", "doc_id": "doc1"}


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with default config."""
        store = VectorStore()
        assert store.config.qdrant_url == "http://localhost:6333"
        assert store.client is None

    def test_custom_initialization(self) -> None:
        """Test initialization with custom config."""
        config = VectorStoreConfig(qdrant_url="http://custom:6333")
        store = VectorStore(config)
        assert store.config.qdrant_url == "http://custom:6333"
        assert store.client is None


class TestVectorStoreConnection:
    """Tests for VectorStore connection management."""

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_connect_success(self, mock_client_class: MagicMock) -> None:
        """Test successful connection."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()

        mock_client_class.assert_called_once_with(url="http://localhost:6333")
        mock_client.get_collections.assert_called_once()
        assert store.client is mock_client

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_connect_failure(self, mock_client_class: MagicMock) -> None:
        """Test connection failure."""
        mock_client_class.side_effect = Exception("Connection refused")

        store = VectorStore()
        with pytest.raises(ConnectionError, match="Failed to connect to Qdrant"):
            store.connect()

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_connect_already_connected(self, mock_client_class: MagicMock) -> None:
        """Test connecting when already connected."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()
        mock_client_class.reset_mock()

        # Second connect should not create new client
        store.connect()
        mock_client_class.assert_not_called()

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_disconnect(self, mock_client_class: MagicMock) -> None:
        """Test disconnection."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()
        store.disconnect()

        mock_client.close.assert_called_once()
        assert store.client is None

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_disconnect_not_connected(self, mock_client_class: MagicMock) -> None:
        """Test disconnect when not connected."""
        store = VectorStore()
        store.disconnect()  # Should not raise
        assert store.client is None

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_context_manager(self, mock_client_class: MagicMock) -> None:
        """Test context manager."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with VectorStore() as store:
            assert store.client is mock_client

        mock_client.close.assert_called_once()


class TestEnsureCollection:
    """Tests for ensure_collection method."""

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_create_new_collection(self, mock_client_class: MagicMock) -> None:
        """Test creating a new collection."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Simulate collection not existing
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        store = VectorStore()
        store.connect()

        result = store.ensure_collection()

        assert result is True
        mock_client.create_collection.assert_called_once()
        # Check that payload indices are created
        assert mock_client.create_payload_index.call_count == 3

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_collection_already_exists(self, mock_client_class: MagicMock) -> None:
        """Test when collection already exists."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Simulate collection existing
        mock_collection = MagicMock()
        mock_collection.name = "knowledge_base"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        store = VectorStore()
        store.connect()

        result = store.ensure_collection()

        assert result is False
        mock_client.create_collection.assert_not_called()

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_collection_conflict_error(self, mock_client_class: MagicMock) -> None:
        """Test handling of 409 conflict error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Simulate collection not existing initially
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # But creation fails with conflict
        error = UnexpectedResponse(
            status_code=409,
            reason_phrase="Conflict",
            content=b"Collection already exists",
            headers={},
        )
        mock_client.create_collection.side_effect = error

        store = VectorStore()
        store.connect()

        result = store.ensure_collection()

        assert result is False

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_ensure_collection_non_409_error(self, mock_client_class: MagicMock) -> None:
        """Test that non-409 UnexpectedResponse raises RuntimeError."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Simulate collection not existing initially
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # But creation fails with a 500 error (not 409)
        error = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"Server error",
            headers={},
        )
        mock_client.create_collection.side_effect = error

        store = VectorStore()
        store.connect()

        with pytest.raises(RuntimeError, match="Failed to create collection"):
            store.ensure_collection()


class TestUpsertEmbeddings:
    """Tests for upsert_embeddings method."""

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_single_batch(self, mock_client_class: MagicMock) -> None:
        """Test upserting a single batch."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        payloads = [
            {"chunk_text": "chunk 1", "source": "test.txt"},
            {"chunk_text": "chunk 2", "source": "test.txt"},
        ]

        ids = store.upsert_embeddings(embeddings, payloads)

        assert len(ids) == 2
        assert all(isinstance(id_, str) for id_ in ids)
        mock_client.upsert.assert_called_once()

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_with_custom_ids(self, mock_client_class: MagicMock) -> None:
        """Test upserting with custom IDs."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()

        embeddings = [[0.1, 0.2, 0.3]]
        payloads = [{"chunk_text": "chunk 1", "source": "test.txt"}]
        custom_ids = ["custom-id-1"]

        ids = store.upsert_embeddings(embeddings, payloads, ids=custom_ids)

        assert ids == custom_ids

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_empty(self, mock_client_class: MagicMock) -> None:
        """Test upserting empty lists."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()

        ids = store.upsert_embeddings([], [])

        assert ids == []
        mock_client.upsert.assert_not_called()

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_mismatched_lengths(self, mock_client_class: MagicMock) -> None:
        """Test upserting with mismatched lengths."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()

        with pytest.raises(ValueError, match="must have the same length"):
            store.upsert_embeddings([[0.1, 0.2]], [{}, {}, {}])

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_mismatched_ids_length(self, mock_client_class: MagicMock) -> None:
        """Test upserting with mismatched IDs length."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()

        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings
        payloads = [{"chunk_text": "chunk 1"}, {"chunk_text": "chunk 2"}]
        ids = ["id-1", "id-2", "id-3"]  # 3 IDs

        with pytest.raises(ValueError, match="must have the same length"):
            store.upsert_embeddings(embeddings, payloads, ids=ids)

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_with_show_progress(self, mock_client_class: MagicMock) -> None:
        """Test upserting with show_progress=True completes successfully."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = VectorStore()
        store.connect()

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        payloads = [
            {"chunk_text": "chunk 1", "source": "test.txt"},
            {"chunk_text": "chunk 2", "source": "test.txt"},
        ]

        # Should complete without exception
        ids = store.upsert_embeddings(embeddings, payloads, show_progress=True)

        assert len(ids) == 2
        assert mock_client.upsert.call_count >= 1

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_multiple_batches(self, mock_client_class: MagicMock) -> None:
        """Test upserting multiple batches."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(batch_size=2)
        store = VectorStore(config)
        store.connect()

        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        payloads = [{"chunk_text": f"chunk {i}"} for i in range(3)]

        ids = store.upsert_embeddings(embeddings, payloads)

        assert len(ids) == 3
        assert mock_client.upsert.call_count == 2  # 2 batches

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_retry_success(self, mock_client_class: MagicMock) -> None:
        """Test successful retry after failure."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # First call fails, second succeeds
        mock_client.upsert.side_effect = [Exception("Timeout"), None]

        store = VectorStore()
        store.connect()

        embeddings = [[0.1, 0.2]]
        payloads = [{"chunk_text": "chunk 1"}]

        ids = store.upsert_embeddings(embeddings, payloads)

        assert len(ids) == 1
        assert mock_client.upsert.call_count == 2

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_upsert_retry_exhausted(self, mock_client_class: MagicMock) -> None:
        """Test failure after max retries."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # All calls fail
        mock_client.upsert.side_effect = Exception("Timeout")

        store = VectorStore()
        store.connect()

        embeddings = [[0.1, 0.2]]
        payloads = [{"chunk_text": "chunk 1"}]

        with pytest.raises(RuntimeError, match="Failed to upsert batch"):
            store.upsert_embeddings(embeddings, payloads)


class TestSearch:
    """Tests for search method."""

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_search_basic(self, mock_client_class: MagicMock) -> None:
        """Test basic search."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock query_points results
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.payload = {"chunk_text": "result text", "source": "test.txt"}
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_result]
        mock_client.query_points.return_value = mock_query_response

        store = VectorStore()
        store.connect()

        results = store.search([0.1, 0.2, 0.3], limit=5)

        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].chunk_text == "result text"
        assert results[0].metadata == {"source": "test.txt"}

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_search_with_source_filter(self, mock_client_class: MagicMock) -> None:
        """Test search with source filter."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_query_response = MagicMock()
        mock_query_response.points = []
        mock_client.query_points.return_value = mock_query_response

        store = VectorStore()
        store.connect()

        store.search([0.1, 0.2], source_filter="test.txt")

        call_args = mock_client.query_points.call_args
        assert call_args.kwargs["query_filter"] is not None

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_search_with_wildcard_source_filter(self, mock_client_class: MagicMock) -> None:
        """Test search with wildcard source filter uses MatchText."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_query_response = MagicMock()
        mock_query_response.points = []
        mock_client.query_points.return_value = mock_query_response

        store = VectorStore()
        store.connect()

        store.search([0.1, 0.2], source_filter="docs/*")

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        # Verify that MatchText is used (not MatchValue) for wildcard patterns
        assert isinstance(query_filter.must[0].match, models.MatchText)

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_search_with_doc_id_filter(self, mock_client_class: MagicMock) -> None:
        """Test search with doc_id filter."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_query_response = MagicMock()
        mock_query_response.points = []
        mock_client.query_points.return_value = mock_query_response

        store = VectorStore()
        store.connect()

        store.search([0.1, 0.2], doc_id_filter="doc1")

        call_args = mock_client.query_points.call_args
        assert call_args.kwargs["query_filter"] is not None

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_search_with_score_threshold(self, mock_client_class: MagicMock) -> None:
        """Test search with score threshold."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_query_response = MagicMock()
        mock_query_response.points = []
        mock_client.query_points.return_value = mock_query_response

        store = VectorStore()
        store.connect()

        store.search([0.1, 0.2], score_threshold=0.8)

        call_args = mock_client.query_points.call_args
        assert call_args.kwargs["score_threshold"] == 0.8

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_search_empty_results(self, mock_client_class: MagicMock) -> None:
        """Test search with no results."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_query_response = MagicMock()
        mock_query_response.points = []
        mock_client.query_points.return_value = mock_query_response

        store = VectorStore()
        store.connect()

        results = store.search([0.1, 0.2])

        assert results == []

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_search_error(self, mock_client_class: MagicMock) -> None:
        """Test search error handling."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.query_points.side_effect = Exception("Search failed")

        store = VectorStore()
        store.connect()

        with pytest.raises(RuntimeError, match="Search failed"):
            store.search([0.1, 0.2])


class TestDeleteBySource:
    """Tests for delete_by_source method."""

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_delete_existing_source(self, mock_client_class: MagicMock) -> None:
        """Test deleting vectors from existing source."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock count result
        mock_count = MagicMock()
        mock_count.count = 5
        mock_client.count.return_value = mock_count

        store = VectorStore()
        store.connect()

        count = store.delete_by_source("test.txt")

        assert count == 5
        mock_client.delete.assert_called_once()

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_delete_nonexistent_source(self, mock_client_class: MagicMock) -> None:
        """Test deleting vectors from non-existent source."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock count result
        mock_count = MagicMock()
        mock_count.count = 0
        mock_client.count.return_value = mock_count

        store = VectorStore()
        store.connect()

        count = store.delete_by_source("nonexistent.txt")

        assert count == 0
        mock_client.delete.assert_not_called()

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_delete_error(self, mock_client_class: MagicMock) -> None:
        """Test delete error handling."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.count.side_effect = Exception("Delete failed")

        store = VectorStore()
        store.connect()

        with pytest.raises(RuntimeError, match="Failed to delete by source"):
            store.delete_by_source("test.txt")


class TestGetStats:
    """Tests for get_stats method."""

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_get_stats_success(self, mock_client_class: MagicMock) -> None:
        """Test successful stats retrieval."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock collection info
        mock_info = MagicMock()
        mock_info.points_count = 1000
        mock_info.indexed_vectors_count = 950
        mock_info.segments_count = 2
        mock_client.get_collection.return_value = mock_info

        store = VectorStore()
        store.connect()

        stats = store.get_stats()

        assert stats["vector_count"] == 1000
        assert stats["indexed_vectors_count"] == 950
        assert stats["segments_count"] == 2
        assert stats["config"]["vector_size"] == 768
        assert stats["config"]["distance"] == "Cosine"

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_get_stats_error(self, mock_client_class: MagicMock) -> None:
        """Test stats retrieval error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.get_collection.side_effect = Exception("Stats error")

        store = VectorStore()
        store.connect()

        with pytest.raises(RuntimeError, match="Failed to get stats"):
            store.get_stats()


class TestIntegrationScenarios:
    """Integration-style tests for common workflows."""

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_full_workflow(self, mock_client_class: MagicMock) -> None:
        """Test a complete workflow: connect, ensure collection, upsert, search."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Setup mocks
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        mock_search_result = MagicMock()
        mock_search_result.score = 0.9
        mock_search_result.payload = {"chunk_text": "found chunk", "source": "doc.txt"}
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_search_result]
        mock_client.query_points.return_value = mock_query_response

        # Execute workflow
        with VectorStore() as store:
            store.ensure_collection()

            embeddings = [[0.1, 0.2, 0.3]]
            payloads = [{"chunk_text": "test chunk", "source": "doc.txt"}]
            store.upsert_embeddings(embeddings, payloads)

            results = store.search([0.1, 0.2, 0.3])

        assert len(results) == 1
        assert results[0].chunk_text == "found chunk"

    @patch("personal_knowledge_base.storage.vector_store.QdrantClient")
    def test_update_existing_document(self, mock_client_class: MagicMock) -> None:
        """Test updating a document by deleting old vectors and upserting new ones."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Setup mocks
        mock_count = MagicMock()
        mock_count.count = 3
        mock_client.count.return_value = mock_count

        store = VectorStore()
        store.connect()

        # Delete old vectors
        deleted = store.delete_by_source("doc.txt")
        assert deleted == 3

        # Upsert new vectors
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [
            {"chunk_text": "new chunk 1", "source": "doc.txt"},
            {"chunk_text": "new chunk 2", "source": "doc.txt"},
        ]
        ids = store.upsert_embeddings(embeddings, payloads)

        assert len(ids) == 2
