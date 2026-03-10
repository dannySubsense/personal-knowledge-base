"""Batch processor for ingesting queued jobs into the knowledge base."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from personal_knowledge_base.processing.chunker import Chunker
from personal_knowledge_base.processing.classifier import ClassifierConfig, ContentClassifier
from personal_knowledge_base.processing.embedder import EmbedderConfig, OllamaEmbedder
from personal_knowledge_base.queue.models import Job
from personal_knowledge_base.queue.operations import get_next_job, update_job_status

if TYPE_CHECKING:
    from personal_knowledge_base.fetchers.base import Fetcher

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for the batch processor.

    Attributes:
        max_jobs_per_run: Maximum number of jobs to process in a single run.
        retry_limit: Maximum number of retry attempts per job before skipping.
        ollama_url: Base URL for the Ollama API.
        qdrant_url: URL of the Qdrant server.
        embedding_model: Ollama model to use for embeddings.
    """

    max_jobs_per_run: int = 50
    retry_limit: int = 3
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "nomic-embed-text"


@dataclass
class BatchResult:
    """Summary of a batch processing run.

    Attributes:
        processed: Total number of jobs attempted.
        succeeded: Number of jobs completed successfully.
        failed: Number of jobs that encountered errors.
        skipped: Number of jobs skipped because they exceeded the retry limit.
        duration_seconds: Wall-clock time for the run.
    """

    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0


class BatchProcessor:
    """Process queued ingestion jobs and store results in Qdrant.

    Pulls jobs from the queue in priority order (1 before 2), fetches
    content, chunks it, embeds it, classifies it to a KB collection, and
    stores the resulting vectors in Qdrant.

    Designed to be run on a schedule (e.g. cron every 15 minutes) without
    holding long-lived locks or spawning background threads.
    """

    def __init__(
        self,
        config: BatchConfig | None = None,
        db_path: str = "~/pkb-data/queue.db",
    ) -> None:
        """Initialise the batch processor.

        Args:
            config: Batch processing configuration.  Defaults used if None.
            db_path: Path to the SQLite queue database (passed for context;
                the queue module resolves its own path via environment or
                defaults).
        """
        self.config = config or BatchConfig()
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BatchResult:
        """Process up to ``max_jobs_per_run`` jobs from the queue.

        Jobs are pulled in priority order (priority=1 first, then 2).
        Returns a :class:`BatchResult` summarising the run.
        """
        start = time.monotonic()
        result = BatchResult()

        jobs_processed = 0
        while jobs_processed < self.config.max_jobs_per_run:
            job = get_next_job()
            if job is None:
                break  # Queue is empty

            result.processed += 1
            jobs_processed += 1

            if not self._should_retry(job):
                logger.warning(
                    "Job %s exceeded retry limit (%d), skipping",
                    job.id,
                    self.config.retry_limit,
                )
                update_job_status(job.id, "failed", error_message="Exceeded retry limit")
                result.skipped += 1
                continue

            success = self._process_job(job)
            if success:
                result.succeeded += 1
            else:
                result.failed += 1

        result.duration_seconds = time.monotonic() - start
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_job(self, job: Job) -> bool:
        """Process a single job through the ingestion pipeline.

        Args:
            job: The job to process.

        Returns:
            ``True`` on success, ``False`` on any failure.
        """
        try:
            update_job_status(job.id, "processing")

            # 1. Fetch content
            fetcher = self._get_fetcher(job.url)
            fetch_result = fetcher.fetch(job.url)
            if not fetch_result.success:
                raise RuntimeError(f"Fetch failed: {fetch_result.error_message}")

            # 2. Chunk content
            chunker = Chunker()
            chunks = chunker.chunk(fetch_result.content, source=job.url)
            if not chunks:
                raise RuntimeError("Chunker returned no chunks for content")

            # 3. Embed chunks
            chunk_tuples = [(c.text, c.metadata) for c in chunks]
            embedding_results = asyncio.run(self._embed_chunks(chunk_tuples))

            # 4. Classify to KB collection (respect explicit kb_name if set)
            if job.kb_name:
                collection_name = job.kb_name
                logger.info("Using explicit KB assignment: %s", collection_name)
            else:
                classifier = ContentClassifier(
                    config=ClassifierConfig(
                        ollama_url=self.config.ollama_url,
                        embedding_model=self.config.embedding_model,
                    )
                )
                collection_name = classifier.classify(
                    url=job.url,
                    title=fetch_result.title,
                    description="",
                )
                logger.info("Classifier routed to KB: %s", collection_name)

            # 5. Store in Qdrant
            from personal_knowledge_base.storage.vector_store import (  # noqa: PLC0415
                VectorStore,
                VectorStoreConfig,
            )

            vs_config = VectorStoreConfig(
                qdrant_url=self.config.qdrant_url,
                collection_name=collection_name,
            )
            vector_store = VectorStore(config=vs_config)
            with vector_store:
                vector_store.ensure_collection()
                embeddings = [er.embedding_vector for er in embedding_results]
                payloads = [
                    {
                        "chunk_text": er.chunk_text,
                        "source": job.url,
                        "title": fetch_result.title,
                        "content_type": fetch_result.content_type,
                        **(er.metadata or {}),
                    }
                    for er in embedding_results
                ]
                vector_store.upsert_embeddings(embeddings, payloads)

            # 6. Mark done
            update_job_status(job.id, "done")
            logger.info("Job %s completed successfully (%d chunks)", job.id, len(chunks))
            return True

        except Exception as exc:
            logger.error("Job %s failed: %s", job.id, exc, exc_info=True)
            update_job_status(job.id, "failed", error_message=str(exc))
            return False

    def _should_retry(self, job: Job) -> bool:
        """Return ``True`` if the job has not yet exceeded the retry limit.

        Args:
            job: The job to check.

        Returns:
            ``True`` when ``job.retry_count < retry_limit``.
        """
        return job.retry_count < self.config.retry_limit

    def _get_fetcher(self, url: str) -> Fetcher:
        """Return the appropriate fetcher for the given URL.

        Routing rules (first match wins):
        - YouTube → :class:`YouTubeFetcher`
        - ``*.pdf`` extension → :class:`PDFFetcher`
        - GitHub → :class:`CodeRepoFetcher`
        - Everything else → :class:`WebFetcher`

        Args:
            url: The URL to route.

        Returns:
            An instantiated :class:`Fetcher` subclass.
        """
        if "youtube.com" in url or "youtu.be" in url:
            from personal_knowledge_base.fetchers.youtube import YouTubeFetcher

            return YouTubeFetcher()
        if url.endswith(".pdf"):
            from personal_knowledge_base.fetchers.pdf import PDFFetcher

            return PDFFetcher()
        if "github.com" in url:
            from personal_knowledge_base.fetchers.code_repo import CodeRepoFetcher

            return CodeRepoFetcher()
        from personal_knowledge_base.fetchers.web import WebFetcher

        return WebFetcher()

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    async def _embed_chunks(
        self,
        chunk_tuples: list[tuple[str, dict | None]],
    ) -> list:
        """Async helper to embed chunks using :class:`OllamaEmbedder`.

        Args:
            chunk_tuples: List of ``(text, metadata)`` pairs.

        Returns:
            List of :class:`~personal_knowledge_base.processing.embedder.EmbeddingResult`.
        """
        embedder_config = EmbedderConfig(
            model=self.config.embedding_model,
            ollama_url=self.config.ollama_url,
        )
        embedder = OllamaEmbedder(config=embedder_config)
        async with embedder:
            return await embedder.embed_chunks(chunk_tuples)
