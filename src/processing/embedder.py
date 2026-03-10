"""Embedding engine for generating vector embeddings using Ollama."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass
class EmbedderConfig:
    """Configuration for the embedding engine.

    Attributes:
        model: Name of the Ollama model to use for embeddings.
        ollama_url: Base URL for the Ollama API.
        batch_size: Number of chunks to embed in a single batch.
        max_retries: Maximum number of retry attempts for failed requests.
        retry_base_delay: Base delay in seconds for exponential backoff.
        timeout: Request timeout in seconds.
    """

    model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    batch_size: int = 32
    max_retries: int = 3
    retry_base_delay: float = 1.0
    timeout: float = 60.0


@dataclass
class EmbeddingResult:
    """Result of embedding a single chunk.

    Attributes:
        chunk_text: The original text that was embedded.
        embedding_vector: The embedding vector (768-dimensional for nomic-embed-text).
        metadata: Additional metadata about the chunk.
    """

    chunk_text: str
    embedding_vector: list[float]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Ensure metadata is never None."""
        if self.metadata is None:
            self.metadata = {}


class OllamaEmbedder:
    """Embedding engine that generates vector embeddings using local Ollama instance.

    This class provides:
    - Batch processing of chunks for efficient embedding generation
    - Exponential backoff for handling transient failures
    - Configurable model and connection parameters
    - 768-dimensional embeddings from nomic-embed-text model
    """

    EMBEDDING_DIMENSION = 768  # nomic-embed-text produces 768-dim embeddings

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        """Initialize the embedder with configuration.

        Args:
            config: Embedder configuration. Uses defaults if not provided.
        """
        self.config = config or EmbedderConfig()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session.

        Returns:
            An active aiohttp ClientSession.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session.

        Should be called when done using the embedder to clean up resources.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> OllamaEmbedder:
        """Async context manager entry.

        Returns:
            The embedder instance.
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        await self.close()

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay.

        Args:
            attempt: The current retry attempt (0-indexed).

        Returns:
            Delay in seconds before the next retry.
        """
        return float(self.config.retry_base_delay * (2**attempt))

    async def _embed_batch(self, texts: list[str], attempt: int = 0) -> list[list[float]]:
        """Embed a batch of texts using Ollama API.

        Args:
            texts: List of texts to embed.
            attempt: Current retry attempt (for internal use).

        Returns:
            List of embedding vectors corresponding to input texts.

        Raises:
            aiohttp.ClientError: If all retry attempts fail.
            RuntimeError: If Ollama returns an error response.
        """
        session = await self._get_session()
        url = f"{self.config.ollama_url}/api/embed"

        payload = {
            "model": self.config.model,
            "input": texts,
        }

        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if "embeddings" not in data:
                    raise RuntimeError(f"Ollama response missing 'embeddings': {data}")

                embeddings: list[list[float]] = data["embeddings"]
                if len(embeddings) != len(texts):
                    raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")

                return embeddings

        except (TimeoutError, aiohttp.ClientError) as e:
            if attempt < self.config.max_retries:
                delay = self._calculate_retry_delay(attempt)
                await asyncio.sleep(delay)
                return await self._embed_batch(texts, attempt + 1)
            raise RuntimeError(
                f"Failed to embed batch after {self.config.max_retries + 1} attempts: {e}"
            ) from e

    def _validate_chunk(self, chunk_text: str) -> None:
        """Validate a chunk before embedding.

        Args:
            chunk_text: The text to validate.

        Raises:
            ValueError: If the chunk is invalid.
        """
        if not isinstance(chunk_text, str):
            raise ValueError(f"Expected string chunk, got {type(chunk_text).__name__}")

    def _create_batches(
        self, chunks: list[tuple[str, dict[str, Any] | None]]
    ) -> list[list[tuple[str, dict[str, Any] | None]]]:
        """Split chunks into batches for processing.

        Args:
            chunks: List of (chunk_text, metadata) tuples.

        Returns:
            List of batches, where each batch is a list of (chunk_text, metadata) tuples.
        """
        batches: list[list[tuple[str, dict[str, Any] | None]]] = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i : i + self.config.batch_size]
            batches.append(batch)
        return batches

    async def embed_chunks(
        self,
        chunks: list[tuple[str, dict[str, Any] | None]],
        show_progress: bool = False,
    ) -> list[EmbeddingResult]:
        """Embed a list of chunks with their metadata.

        This is the main entry point for embedding chunks. It handles:
        - Batch processing for efficiency
        - Validation of input chunks
        - Retry logic with exponential backoff
        - Progress tracking (optional)

        Args:
            chunks: List of (chunk_text, metadata) tuples to embed.
            show_progress: Whether to print progress information.

        Returns:
            List of EmbeddingResult objects containing the embeddings.

        Raises:
            ValueError: If any chunk is invalid.
            RuntimeError: If embedding fails after all retries.
        """
        if not chunks:
            return []

        # Validate all chunks first
        for chunk_text, _metadata in chunks:
            self._validate_chunk(chunk_text)

        # Create batches
        batches = self._create_batches(chunks)
        results: list[EmbeddingResult] = []

        for batch_idx, batch in enumerate(batches):
            if show_progress:
                print(
                    f"Processing batch {batch_idx + 1}/{len(batches)} " f"({len(batch)} chunks)..."
                )

            # Extract texts from batch
            texts = [chunk_text for chunk_text, _metadata in batch]

            # Get embeddings for this batch
            embeddings = await self._embed_batch(texts)

            # Create EmbeddingResult objects
            for (chunk_text, metadata), embedding_vector in zip(batch, embeddings, strict=True):
                # Validate embedding dimension
                if len(embedding_vector) != self.EMBEDDING_DIMENSION:
                    raise RuntimeError(
                        f"Expected {self.EMBEDDING_DIMENSION}-dimensional embedding, "
                        f"got {len(embedding_vector)} dimensions"
                    )

                result = EmbeddingResult(
                    chunk_text=chunk_text,
                    embedding_vector=embedding_vector,
                    metadata=metadata.copy() if metadata else {},
                )
                results.append(result)

        return results

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Convenience method for embedding a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            ValueError: If text is not a string.
            RuntimeError: If embedding fails after all retries.
        """
        self._validate_chunk(text)
        results = await self.embed_chunks([(text, None)])
        return results[0].embedding_vector

    async def health_check(self) -> dict[str, Any]:
        """Check if Ollama is accessible and the model is available.

        Returns:
            Dictionary with health status information.
        """
        session = await self._get_session()
        url = f"{self.config.ollama_url}/api/tags"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                response.raise_for_status()
                data = await response.json()

                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                model_available = any(self.config.model in name for name in model_names)

                return {
                    "status": "healthy" if model_available else "model_missing",
                    "ollama_url": self.config.ollama_url,
                    "model": self.config.model,
                    "model_available": model_available,
                    "available_models": model_names,
                }

        except (TimeoutError, aiohttp.ClientError) as e:
            return {
                "status": "unreachable",
                "ollama_url": self.config.ollama_url,
                "model": self.config.model,
                "error": str(e),
            }
