"""Tests for the processing embedder module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from personal_knowledge_base.processing.embedder import (
    EmbedderConfig,
    EmbeddingResult,
    OllamaEmbedder,
)


class TestEmbedderConfig:
    """Test EmbedderConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EmbedderConfig()
        assert config.model == "nomic-embed-text"
        assert config.ollama_url == "http://localhost:11434"
        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.timeout == 60.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = EmbedderConfig(
            model="custom-model",
            ollama_url="http://custom:11434",
            batch_size=64,
            max_retries=5,
            retry_base_delay=2.0,
            timeout=120.0,
        )
        assert config.model == "custom-model"
        assert config.ollama_url == "http://custom:11434"
        assert config.batch_size == 64
        assert config.max_retries == 5
        assert config.retry_base_delay == 2.0
        assert config.timeout == 120.0


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_embedding_result_creation(self) -> None:
        """Test creating an embedding result."""
        vector = [0.1] * 768
        result = EmbeddingResult(
            chunk_text="Test text",
            embedding_vector=vector,
        )
        assert result.chunk_text == "Test text"
        assert result.embedding_vector == vector
        assert result.metadata == {}

    def test_embedding_result_with_metadata(self) -> None:
        """Test creating an embedding result with metadata."""
        vector = [0.1] * 768
        metadata = {"source": "test", "index": 0}
        result = EmbeddingResult(
            chunk_text="Test text",
            embedding_vector=vector,
            metadata=metadata,
        )
        assert result.metadata == metadata

    def test_embedding_result_metadata_defaults_to_empty_dict(self) -> None:
        """Test that metadata defaults to empty dict when None."""
        vector = [0.1] * 768
        result = EmbeddingResult(
            chunk_text="Test text",
            embedding_vector=vector,
            metadata=None,
        )
        assert result.metadata == {}


class TestOllamaEmbedderInitialization:
    """Test OllamaEmbedder initialization."""

    def test_default_config(self) -> None:
        """Test embedder with default config."""
        embedder = OllamaEmbedder()
        assert embedder.config.model == "nomic-embed-text"
        assert embedder.config.ollama_url == "http://localhost:11434"
        assert embedder._session is None

    def test_custom_config(self) -> None:
        """Test embedder with custom config."""
        config = EmbedderConfig(model="custom-model", batch_size=64)
        embedder = OllamaEmbedder(config)
        assert embedder.config.model == "custom-model"
        assert embedder.config.batch_size == 64


class TestOllamaEmbedderRetryDelay:
    """Test retry delay calculation."""

    def test_retry_delay_exponential_backoff(self) -> None:
        """Test exponential backoff calculation."""
        config = EmbedderConfig(retry_base_delay=1.0)
        embedder = OllamaEmbedder(config)

        assert embedder._calculate_retry_delay(0) == 1.0
        assert embedder._calculate_retry_delay(1) == 2.0
        assert embedder._calculate_retry_delay(2) == 4.0
        assert embedder._calculate_retry_delay(3) == 8.0

    def test_retry_delay_custom_base(self) -> None:
        """Test retry delay with custom base delay."""
        config = EmbedderConfig(retry_base_delay=0.5)
        embedder = OllamaEmbedder(config)

        assert embedder._calculate_retry_delay(0) == 0.5
        assert embedder._calculate_retry_delay(1) == 1.0
        assert embedder._calculate_retry_delay(2) == 2.0


class TestOllamaEmbedderValidation:
    """Test input validation."""

    def test_validate_chunk_valid_string(self) -> None:
        """Test validation passes for valid string."""
        embedder = OllamaEmbedder()
        embedder._validate_chunk("Valid text")  # Should not raise

    def test_validate_chunk_invalid_type(self) -> None:
        """Test validation fails for non-string types."""
        embedder = OllamaEmbedder()

        with pytest.raises(ValueError, match="Expected string chunk, got int"):
            embedder._validate_chunk(123)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Expected string chunk, got list"):
            embedder._validate_chunk(["test"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Expected string chunk, got NoneType"):
            embedder._validate_chunk(None)  # type: ignore[arg-type]


class TestOllamaEmbedderBatching:
    """Test batch creation logic."""

    def test_create_batches_empty(self) -> None:
        """Test batching empty list."""
        embedder = OllamaEmbedder()
        batches = embedder._create_batches([])
        assert batches == []

    def test_create_batches_single_batch(self) -> None:
        """Test batching when all chunks fit in one batch."""
        config = EmbedderConfig(batch_size=5)
        embedder = OllamaEmbedder(config)

        chunks = [(f"chunk {i}", {"idx": i}) for i in range(3)]
        batches = embedder._create_batches(chunks)

        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_create_batches_multiple_batches(self) -> None:
        """Test batching across multiple batches."""
        config = EmbedderConfig(batch_size=2)
        embedder = OllamaEmbedder(config)

        chunks = [(f"chunk {i}", {"idx": i}) for i in range(5)]
        batches = embedder._create_batches(chunks)

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_create_batches_exact_fit(self) -> None:
        """Test batching when chunks fit exactly."""
        config = EmbedderConfig(batch_size=3)
        embedder = OllamaEmbedder(config)

        chunks = [(f"chunk {i}", None) for i in range(6)]
        batches = embedder._create_batches(chunks)

        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3


def _create_mock_session(mock_response: MagicMock) -> MagicMock:
    """Create a properly configured mock session."""
    mock_session = MagicMock()
    mock_session.post = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_session.closed = False
    return mock_session


@pytest.mark.asyncio
class TestOllamaEmbedderEmbedBatch:
    """Test the _embed_batch method."""

    async def test_embed_batch_success(self) -> None:
        """Test successful batch embedding."""
        embedder = OllamaEmbedder()

        # Mock response data
        mock_embeddings = [[0.1] * 768, [0.2] * 768]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"embeddings": mock_embeddings})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        result = await embedder._embed_batch(["text1", "text2"])

        assert result == mock_embeddings
        mock_session.post.assert_called_once()

    async def test_embed_batch_missing_embeddings(self) -> None:
        """Test handling of response missing embeddings."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"error": "some error"})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        with pytest.raises(RuntimeError, match="Ollama response missing 'embeddings'"):
            await embedder._embed_batch(["text1"])

    async def test_embed_batch_wrong_count(self) -> None:
        """Test handling of wrong number of embeddings."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"embeddings": [[0.1] * 768]})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        with pytest.raises(RuntimeError, match="Expected 2 embeddings, got 1"):
            await embedder._embed_batch(["text1", "text2"])

    async def test_embed_batch_retry_on_client_error(self) -> None:
        """Test retry on client error."""
        config = EmbedderConfig(max_retries=2, retry_base_delay=0.001)
        embedder = OllamaEmbedder(config)

        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientError("Connection error")
        )

        mock_response_success = MagicMock()
        mock_response_success.raise_for_status = MagicMock()
        mock_response_success.json = AsyncMock(return_value={"embeddings": [[0.1] * 768]})

        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(
            side_effect=[mock_response_fail, mock_response_success]
        )
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session.closed = False

        embedder._session = mock_session

        result = await embedder._embed_batch(["text1"])

        assert result == [[0.1] * 768]
        assert mock_session.post.call_count == 2

    async def test_embed_batch_retry_on_timeout(self) -> None:
        """Test retry on timeout."""
        config = EmbedderConfig(max_retries=1, retry_base_delay=0.001)
        embedder = OllamaEmbedder(config)

        mock_response_success = MagicMock()
        mock_response_success.raise_for_status = MagicMock()
        mock_response_success.json = AsyncMock(return_value={"embeddings": [[0.1] * 768]})

        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(
            side_effect=[TimeoutError(), mock_response_success]
        )
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session.closed = False

        embedder._session = mock_session

        result = await embedder._embed_batch(["text1"])

        assert result == [[0.1] * 768]
        assert mock_session.post.call_count == 2

    async def test_embed_batch_exhausted_retries(self) -> None:
        """Test failure after exhausting all retries."""
        config = EmbedderConfig(max_retries=1, retry_base_delay=0.001)
        embedder = OllamaEmbedder(config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientError("Connection error")
        )

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        with pytest.raises(RuntimeError, match="Failed to embed batch after 2 attempts"):
            await embedder._embed_batch(["text1"])


@pytest.mark.asyncio
class TestOllamaEmbedderEmbedChunks:
    """Test the embed_chunks method."""

    async def test_embed_chunks_empty(self) -> None:
        """Test embedding empty list."""
        embedder = OllamaEmbedder()
        result = await embedder.embed_chunks([])
        assert result == []

    async def test_embed_chunks_single_chunk(self) -> None:
        """Test embedding a single chunk."""
        embedder = OllamaEmbedder()

        mock_embedding = [0.1] * 768
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"embeddings": [mock_embedding]})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        result = await embedder.embed_chunks([("test text", {"source": "test"})])

        assert len(result) == 1
        assert result[0].chunk_text == "test text"
        assert result[0].embedding_vector == mock_embedding
        assert result[0].metadata == {"source": "test"}

    async def test_embed_chunks_multiple_batches(self) -> None:
        """Test embedding across multiple batches."""
        config = EmbedderConfig(batch_size=2)
        embedder = OllamaEmbedder(config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            side_effect=[
                {"embeddings": [[0.1] * 768, [0.2] * 768]},
                {"embeddings": [[0.3] * 768]},
            ]
        )

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        chunks = [
            ("chunk1", {"idx": 0}),
            ("chunk2", {"idx": 1}),
            ("chunk3", {"idx": 2}),
        ]
        result = await embedder.embed_chunks(chunks)

        assert len(result) == 3
        assert result[0].chunk_text == "chunk1"
        assert result[1].chunk_text == "chunk2"
        assert result[2].chunk_text == "chunk3"
        assert mock_session.post.call_count == 2

    async def test_embed_chunks_invalid_chunk_type(self) -> None:
        """Test validation of invalid chunk types."""
        embedder = OllamaEmbedder()

        with pytest.raises(ValueError, match="Expected string chunk, got int"):
            await embedder.embed_chunks([(123, None)])  # type: ignore[list-item]

    async def test_embed_chunks_show_progress(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress output when show_progress=True."""
        config = EmbedderConfig(batch_size=2)
        embedder = OllamaEmbedder(config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            side_effect=[
                {"embeddings": [[0.1] * 768, [0.2] * 768]},
                {"embeddings": [[0.3] * 768]},
            ]
        )

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        chunks = [
            ("chunk1", {"idx": 0}),
            ("chunk2", {"idx": 1}),
            ("chunk3", {"idx": 2}),
        ]
        result = await embedder.embed_chunks(chunks, show_progress=True)

        assert len(result) == 3
        captured = capsys.readouterr()
        assert "Processing batch 1/2" in captured.out
        assert "Processing batch 2/2" in captured.out

    async def test_embed_chunks_wrong_embedding_dimension(self) -> None:
        """Test handling of wrong embedding dimension."""
        embedder = OllamaEmbedder()

        # Return wrong dimension
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"embeddings": [[0.1] * 512]})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        with pytest.raises(RuntimeError, match="Expected 768-dimensional embedding"):
            await embedder.embed_chunks([("test", None)])

    async def test_embed_chunks_preserves_metadata(self) -> None:
        """Test that metadata is preserved in results."""
        embedder = OllamaEmbedder()

        mock_embedding = [0.1] * 768
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"embeddings": [mock_embedding]})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        metadata = {"source": "test", "index": 42, "nested": {"key": "value"}}
        result = await embedder.embed_chunks([("test text", metadata)])

        assert result[0].metadata == metadata

    async def test_embed_chunks_none_metadata(self) -> None:
        """Test handling of None metadata."""
        embedder = OllamaEmbedder()

        mock_embedding = [0.1] * 768
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"embeddings": [mock_embedding]})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        result = await embedder.embed_chunks([("test text", None)])

        assert result[0].metadata == {}


@pytest.mark.asyncio
class TestOllamaEmbedderEmbedText:
    """Test the embed_text convenience method."""

    async def test_embed_text_success(self) -> None:
        """Test embedding a single text."""
        embedder = OllamaEmbedder()

        mock_embedding = [0.1] * 768
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"embeddings": [mock_embedding]})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        result = await embedder.embed_text("test text")

        assert result == mock_embedding

    async def test_embed_text_invalid_type(self) -> None:
        """Test embedding invalid type."""
        embedder = OllamaEmbedder()

        with pytest.raises(ValueError, match="Expected string chunk, got int"):
            await embedder.embed_text(123)  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestOllamaEmbedderHealthCheck:
    """Test the health_check method."""

    async def test_health_check_healthy(self) -> None:
        """Test health check when model is available."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "models": [
                    {"name": "nomic-embed-text:latest"},
                    {"name": "llama2:latest"},
                ]
            }
        )

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        result = await embedder.health_check()

        assert result["status"] == "healthy"
        assert result["model_available"] is True
        assert "nomic-embed-text" in result["available_models"][0]

    async def test_health_check_model_missing(self) -> None:
        """Test health check when Ollama is up but model is missing."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"models": [{"name": "llama2:latest"}]})

        mock_session = _create_mock_session(mock_response)
        embedder._session = mock_session

        result = await embedder.health_check()

        assert result["status"] == "model_missing"
        assert result["model_available"] is False

    async def test_health_check_unreachable(self) -> None:
        """Test health check when Ollama is unreachable."""
        embedder = OllamaEmbedder()

        mock_session = MagicMock()
        mock_session.get = MagicMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session.closed = False

        embedder._session = mock_session

        result = await embedder.health_check()

        assert result["status"] == "unreachable"
        assert "error" in result


@pytest.mark.asyncio
class TestOllamaEmbedderSessionManagement:
    """Test session management."""

    async def test_get_session_creates_new_session(self) -> None:
        """Test that get_session creates a new session if none exists."""
        embedder = OllamaEmbedder()

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_instance = MagicMock()
            mock_session_instance.closed = False
            mock_session_class.return_value = mock_session_instance

            session = await embedder._get_session()

            assert session is mock_session_instance
            assert embedder._session is session

    async def test_get_session_reuses_existing(self) -> None:
        """Test that get_session reuses existing session."""
        embedder = OllamaEmbedder()

        mock_session = MagicMock()
        mock_session.closed = False
        embedder._session = mock_session

        session = await embedder._get_session()

        assert session is mock_session

    async def test_close_session(self) -> None:
        """Test closing the session."""
        embedder = OllamaEmbedder()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        embedder._session = mock_session

        await embedder.close()

        assert embedder._session is None
        mock_session.close.assert_called_once()

    async def test_context_manager(self) -> None:
        """Test async context manager."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_instance = MagicMock()
            mock_session_instance.closed = False
            mock_session_instance.close = AsyncMock()
            mock_session_class.return_value = mock_session_instance

            async with OllamaEmbedder() as embedder:
                assert isinstance(embedder, OllamaEmbedder)


class TestOllamaEmbedderConstants:
    """Test embedder constants."""

    def test_embedding_dimension_constant(self) -> None:
        """Test that EMBEDDING_DIMENSION is 768."""
        assert OllamaEmbedder.EMBEDDING_DIMENSION == 768
