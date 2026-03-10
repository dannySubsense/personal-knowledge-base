"""Tests for the batch processor module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from personal_knowledge_base.batch.processor import BatchConfig, BatchProcessor, BatchResult
from personal_knowledge_base.fetchers.base import FetchResult
from personal_knowledge_base.processing.chunker import Chunk
from personal_knowledge_base.processing.embedder import EmbeddingResult
from personal_knowledge_base.queue.models import Job

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_job(
    job_id: str = "job-001",
    url: str = "https://example.com",
    priority: int = 2,
    status: str = "pending",
    retry_count: int = 0,
) -> Job:
    """Return a Job with sensible defaults."""
    return Job(
        id=job_id,
        url=url,
        priority=priority,
        status=status,
        retry_count=retry_count,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_fetch_result(url: str = "https://example.com", success: bool = True) -> FetchResult:
    """Return a FetchResult with sensible defaults."""
    if success:
        return FetchResult(
            url=url,
            title="Test Article",
            content="This is the article content. " * 20,
            content_type="article",
            success=True,
        )
    return FetchResult(
        url=url,
        success=False,
        error_message="Network error",
    )


def _make_chunks(n: int = 2, url: str = "https://example.com") -> list[Chunk]:
    """Return a list of Chunk objects."""
    return [
        Chunk(
            text=f"Chunk {i} content",
            index=i,
            total=n,
            source=url,
            metadata={"token_estimate": 20},
        )
        for i in range(n)
    ]


def _make_embedding_results(chunks: list[Chunk]) -> list[EmbeddingResult]:
    """Return EmbeddingResult objects matching the given chunks."""
    return [
        EmbeddingResult(
            chunk_text=c.text,
            embedding_vector=[0.1] * 768,
            metadata=c.metadata,
        )
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------


class TestBatchConfig:
    """Test BatchConfig dataclass defaults and custom values."""

    def test_defaults(self) -> None:
        config = BatchConfig()
        assert config.max_jobs_per_run == 50
        assert config.retry_limit == 3
        assert config.ollama_url == "http://localhost:11434"
        assert config.qdrant_url == "http://localhost:6333"
        assert config.embedding_model == "nomic-embed-text"

    def test_custom_values(self) -> None:
        config = BatchConfig(
            max_jobs_per_run=10,
            retry_limit=1,
            ollama_url="http://ollama:11434",
            qdrant_url="http://qdrant:6333",
            embedding_model="all-minilm",
        )
        assert config.max_jobs_per_run == 10
        assert config.retry_limit == 1
        assert config.ollama_url == "http://ollama:11434"
        assert config.qdrant_url == "http://qdrant:6333"
        assert config.embedding_model == "all-minilm"


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_defaults(self) -> None:
        result = BatchResult()
        assert result.processed == 0
        assert result.succeeded == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.duration_seconds == 0.0

    def test_counts_are_independent(self) -> None:
        result = BatchResult(processed=5, succeeded=3, failed=1, skipped=1, duration_seconds=2.5)
        assert result.processed == 5
        assert result.succeeded == 3
        assert result.failed == 1
        assert result.skipped == 1
        assert result.duration_seconds == 2.5


# ---------------------------------------------------------------------------
# BatchProcessor._should_retry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    """Test _should_retry logic."""

    def test_fresh_job_should_retry(self) -> None:
        processor = BatchProcessor(config=BatchConfig(retry_limit=3))
        job = _make_job(retry_count=0)
        assert processor._should_retry(job) is True

    def test_job_at_limit_should_not_retry(self) -> None:
        processor = BatchProcessor(config=BatchConfig(retry_limit=3))
        job = _make_job(retry_count=3)
        assert processor._should_retry(job) is False

    def test_job_beyond_limit_should_not_retry(self) -> None:
        processor = BatchProcessor(config=BatchConfig(retry_limit=3))
        job = _make_job(retry_count=5)
        assert processor._should_retry(job) is False

    def test_job_one_below_limit_should_retry(self) -> None:
        processor = BatchProcessor(config=BatchConfig(retry_limit=3))
        job = _make_job(retry_count=2)
        assert processor._should_retry(job) is True


# ---------------------------------------------------------------------------
# BatchProcessor._get_fetcher
# ---------------------------------------------------------------------------


class TestGetFetcher:
    """Test fetcher routing logic."""

    def test_youtube_url(self) -> None:
        from personal_knowledge_base.fetchers.youtube import YouTubeFetcher

        processor = BatchProcessor()
        fetcher = processor._get_fetcher("https://www.youtube.com/watch?v=abc123")
        assert isinstance(fetcher, YouTubeFetcher)

    def test_youtu_be_url(self) -> None:
        from personal_knowledge_base.fetchers.youtube import YouTubeFetcher

        processor = BatchProcessor()
        fetcher = processor._get_fetcher("https://youtu.be/abc123")
        assert isinstance(fetcher, YouTubeFetcher)

    def test_pdf_url(self) -> None:
        from personal_knowledge_base.fetchers.pdf import PDFFetcher

        processor = BatchProcessor()
        fetcher = processor._get_fetcher("https://example.com/paper.pdf")
        assert isinstance(fetcher, PDFFetcher)

    def test_github_url(self) -> None:
        from personal_knowledge_base.fetchers.code_repo import CodeRepoFetcher

        processor = BatchProcessor()
        fetcher = processor._get_fetcher("https://github.com/user/repo")
        assert isinstance(fetcher, CodeRepoFetcher)

    def test_generic_web_url(self) -> None:
        from personal_knowledge_base.fetchers.web import WebFetcher

        processor = BatchProcessor()
        fetcher = processor._get_fetcher("https://example.com/article")
        assert isinstance(fetcher, WebFetcher)

    def test_https_arxiv_falls_back_to_web(self) -> None:
        """Non-PDF, non-youtube, non-github → WebFetcher."""
        from personal_knowledge_base.fetchers.web import WebFetcher

        processor = BatchProcessor()
        fetcher = processor._get_fetcher("https://arxiv.org/abs/2301.00001")
        assert isinstance(fetcher, WebFetcher)


# ---------------------------------------------------------------------------
# BatchProcessor._process_job
# ---------------------------------------------------------------------------


class TestProcessJob:
    """Test _process_job success and failure paths."""

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.storage.vector_store.VectorStore")
    @patch("personal_knowledge_base.batch.processor.ContentClassifier")
    @patch("personal_knowledge_base.batch.processor.Chunker")
    def test_successful_job(
        self,
        mock_chunker_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        mock_vs_cls: MagicMock,
        mock_update_status: MagicMock,
    ) -> None:
        processor = BatchProcessor()
        job = _make_job()
        chunks = _make_chunks()
        embedding_results = _make_embedding_results(chunks)

        mock_chunker_cls.return_value.chunk.return_value = chunks
        mock_classifier_cls.return_value.classify.return_value = "general"

        mock_vs = MagicMock()
        mock_vs.__enter__ = MagicMock(return_value=mock_vs)
        mock_vs.__exit__ = MagicMock(return_value=False)
        mock_vs_cls.return_value = mock_vs

        with (
            patch.object(
                processor,
                "_get_fetcher",
                return_value=MagicMock(fetch=MagicMock(return_value=_make_fetch_result())),
            ),
            patch.object(processor, "_embed_chunks", new_callable=AsyncMock) as mock_embed,
        ):
            mock_embed.return_value = embedding_results
            result = processor._process_job(job)

        assert result is True
        # status set to "processing" then "done"
        calls = [c.args for c in mock_update_status.call_args_list]
        assert (job.id, "processing") in calls
        assert (job.id, "done") in calls

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    def test_failed_fetch_marks_job_failed(self, mock_update_status: MagicMock) -> None:
        processor = BatchProcessor()
        job = _make_job()

        with patch.object(
            processor,
            "_get_fetcher",
            return_value=MagicMock(fetch=MagicMock(return_value=_make_fetch_result(success=False))),
        ):
            result = processor._process_job(job)

        assert result is False
        statuses = [c.args[1] for c in mock_update_status.call_args_list]
        assert "failed" in statuses

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.Chunker")
    def test_empty_chunks_marks_job_failed(
        self,
        mock_chunker_cls: MagicMock,
        mock_update_status: MagicMock,
    ) -> None:
        processor = BatchProcessor()
        job = _make_job()
        mock_chunker_cls.return_value.chunk.return_value = []

        with patch.object(
            processor,
            "_get_fetcher",
            return_value=MagicMock(fetch=MagicMock(return_value=_make_fetch_result())),
        ):
            result = processor._process_job(job)

        assert result is False
        statuses = [c.args[1] for c in mock_update_status.call_args_list]
        assert "failed" in statuses

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.storage.vector_store.VectorStore")
    @patch("personal_knowledge_base.batch.processor.ContentClassifier")
    @patch("personal_knowledge_base.batch.processor.Chunker")
    def test_exception_in_upsert_marks_job_failed(
        self,
        mock_chunker_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        mock_vs_cls: MagicMock,
        mock_update_status: MagicMock,
    ) -> None:
        processor = BatchProcessor()
        job = _make_job()
        chunks = _make_chunks()
        embedding_results = _make_embedding_results(chunks)

        mock_chunker_cls.return_value.chunk.return_value = chunks
        mock_classifier_cls.return_value.classify.return_value = "general"

        mock_vs = MagicMock()
        mock_vs.__enter__ = MagicMock(return_value=mock_vs)
        mock_vs.__exit__ = MagicMock(return_value=False)
        mock_vs.upsert_embeddings.side_effect = RuntimeError("Qdrant unavailable")
        mock_vs_cls.return_value = mock_vs

        with (
            patch.object(
                processor,
                "_get_fetcher",
                return_value=MagicMock(fetch=MagicMock(return_value=_make_fetch_result())),
            ),
            patch.object(processor, "_embed_chunks", new_callable=AsyncMock) as mock_embed,
        ):
            mock_embed.return_value = embedding_results
            result = processor._process_job(job)

        assert result is False
        statuses = [c.args[1] for c in mock_update_status.call_args_list]
        assert "failed" in statuses


# ---------------------------------------------------------------------------
# BatchProcessor.run
# ---------------------------------------------------------------------------


class TestBatchProcessorRun:
    """Integration-level tests for the run() method."""

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_empty_queue_returns_zero_counts(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        mock_get_next.return_value = None
        processor = BatchProcessor()
        result = processor.run()

        assert result.processed == 0
        assert result.succeeded == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.duration_seconds >= 0.0

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_max_jobs_per_run_respected(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        """run() should stop after max_jobs_per_run even if more jobs exist."""
        jobs = [_make_job(job_id=f"job-{i}") for i in range(10)]
        mock_get_next.side_effect = jobs + [None]

        processor = BatchProcessor(config=BatchConfig(max_jobs_per_run=3))
        with patch.object(processor, "_process_job", return_value=True) as mock_proc:
            result = processor.run()

        assert result.processed == 3
        assert mock_proc.call_count == 3

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_priority_order_processed(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        """Jobs are processed in the order returned by get_next_job (priority 1 first)."""
        p1_job = _make_job(job_id="p1-job", priority=1)
        p2_job = _make_job(job_id="p2-job", priority=2)
        # get_next_job returns p1 first (queue already sorted by priority)
        mock_get_next.side_effect = [p1_job, p2_job, None]

        processor = BatchProcessor()
        processed_ids: list[str] = []

        def track_job(job: Job) -> bool:
            processed_ids.append(job.id)
            return True

        with patch.object(processor, "_process_job", side_effect=track_job):
            processor.run()

        assert processed_ids == ["p1-job", "p2-job"]

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_succeeded_count_correct(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        jobs = [_make_job(job_id=f"job-{i}") for i in range(4)]
        mock_get_next.side_effect = jobs + [None]

        processor = BatchProcessor()
        with patch.object(processor, "_process_job", return_value=True):
            result = processor.run()

        assert result.succeeded == 4
        assert result.failed == 0

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_failed_count_correct(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        jobs = [_make_job(job_id=f"job-{i}") for i in range(3)]
        mock_get_next.side_effect = jobs + [None]

        processor = BatchProcessor()
        with patch.object(processor, "_process_job", return_value=False):
            result = processor.run()

        assert result.failed == 3
        assert result.succeeded == 0

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_exceeded_retry_limit_skipped(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        """Jobs that exceeded retry_limit are skipped, not processed."""
        exhausted = _make_job(job_id="exhausted", retry_count=3)
        mock_get_next.side_effect = [exhausted, None]

        processor = BatchProcessor(config=BatchConfig(retry_limit=3))
        with patch.object(processor, "_process_job") as mock_proc:
            result = processor.run()

        assert result.skipped == 1
        assert result.processed == 1  # counted as processed attempt
        mock_proc.assert_not_called()
        # status must be set to "failed"
        mock_update.assert_called_once_with(
            exhausted.id, "failed", error_message="Exceeded retry limit"
        )

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_mixed_success_failure_skipped(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        ok_job = _make_job(job_id="ok", retry_count=0)
        fail_job = _make_job(job_id="fail", retry_count=0)
        skip_job = _make_job(job_id="skip", retry_count=5)
        mock_get_next.side_effect = [ok_job, fail_job, skip_job, None]

        processor = BatchProcessor(config=BatchConfig(retry_limit=3))

        def side_effect(job: Job) -> bool:
            return job.id == "ok"

        with patch.object(processor, "_process_job", side_effect=side_effect):
            result = processor.run()

        assert result.processed == 3
        assert result.succeeded == 1
        assert result.failed == 1
        assert result.skipped == 1

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    def test_duration_is_recorded(
        self,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        mock_get_next.return_value = None
        processor = BatchProcessor()
        result = processor.run()
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# Full pipeline smoke test (all external calls mocked)
# ---------------------------------------------------------------------------


class TestFullPipelineSmoke:
    """End-to-end smoke test with all external services mocked."""

    @patch("personal_knowledge_base.batch.processor.update_job_status")
    @patch("personal_knowledge_base.batch.processor.get_next_job")
    @patch("personal_knowledge_base.storage.vector_store.VectorStore")
    @patch("personal_knowledge_base.batch.processor.ContentClassifier")
    @patch("personal_knowledge_base.batch.processor.Chunker")
    def test_full_pipeline_web(
        self,
        mock_chunker_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        mock_vs_cls: MagicMock,
        mock_get_next: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        """Smoke test: a web URL job flows through the full pipeline."""
        job = _make_job(url="https://example.com/article")
        mock_get_next.side_effect = [job, None]

        chunks = _make_chunks(url=job.url)
        embedding_results = _make_embedding_results(chunks)

        mock_chunker_cls.return_value.chunk.return_value = chunks
        mock_classifier_cls.return_value.classify.return_value = "general"

        mock_vs = MagicMock()
        mock_vs.__enter__ = MagicMock(return_value=mock_vs)
        mock_vs.__exit__ = MagicMock(return_value=False)
        mock_vs_cls.return_value = mock_vs

        processor = BatchProcessor()
        with (
            patch.object(
                processor,
                "_get_fetcher",
                return_value=MagicMock(fetch=MagicMock(return_value=_make_fetch_result())),
            ),
            patch.object(processor, "_embed_chunks", new_callable=AsyncMock) as mock_embed,
        ):
            mock_embed.return_value = embedding_results
            result = processor.run()

        assert result.succeeded == 1
        assert result.failed == 0
        assert result.skipped == 0
        mock_vs.upsert_embeddings.assert_called_once()


# ---------------------------------------------------------------------------
# KB name respect tests (R4-D)
# ---------------------------------------------------------------------------


def test_process_job_uses_explicit_kb_name():
    """Test that explicit kb_name is respected and classifier is NOT called."""
    processor = BatchProcessor()

    # Create job with explicit kb_name
    job = _make_job(job_id="job-explicit", url="https://example.com/video")
    job.kb_name = "agentic-coding"

    embedding_results = [
        EmbeddingResult(chunk_text="test", embedding_vector=[0.1] * 768, metadata={}),
    ]

    with (
        patch.object(
            processor,
            "_get_fetcher",
            return_value=MagicMock(fetch=MagicMock(return_value=_make_fetch_result())),
        ),
        patch.object(processor, "_embed_chunks", new_callable=AsyncMock) as mock_embed,
        patch("personal_knowledge_base.batch.processor.ContentClassifier") as mock_classifier_cls,
        patch("personal_knowledge_base.storage.vector_store.VectorStore") as mock_vs_cls,
    ):
        mock_embed.return_value = embedding_results
        mock_vs = MagicMock()
        mock_vs.__enter__ = MagicMock(return_value=mock_vs)
        mock_vs.__exit__ = MagicMock(return_value=False)
        mock_vs_cls.return_value = mock_vs

        success = processor._process_job(job)

        assert success is True
        # Classifier should NOT be instantiated when kb_name is set
        mock_classifier_cls.assert_not_called()
        # VectorStore config should have the explicit kb_name as collection
        mock_vs_cls.assert_called_once()
        call_kwargs = mock_vs_cls.call_args.kwargs
        assert call_kwargs["config"].collection_name == "agentic-coding"


def test_process_job_calls_classifier_when_kb_name_none():
    """Test that classifier is called when kb_name is None."""
    processor = BatchProcessor()

    # Create job without kb_name (None)
    job = _make_job(job_id="job-classify", url="https://example.com/article")
    job.kb_name = None

    embedding_results = [
        EmbeddingResult(chunk_text="test", embedding_vector=[0.1] * 768, metadata={}),
    ]

    with (
        patch.object(
            processor,
            "_get_fetcher",
            return_value=MagicMock(fetch=MagicMock(return_value=_make_fetch_result())),
        ),
        patch.object(processor, "_embed_chunks", new_callable=AsyncMock) as mock_embed,
        patch("personal_knowledge_base.batch.processor.ContentClassifier") as mock_classifier_cls,
        patch("personal_knowledge_base.storage.vector_store.VectorStore") as mock_vs_cls,
    ):
        mock_embed.return_value = embedding_results
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = "ml-ai"
        mock_classifier_cls.return_value = mock_classifier
        mock_vs = MagicMock()
        mock_vs.__enter__ = MagicMock(return_value=mock_vs)
        mock_vs.__exit__ = MagicMock(return_value=False)
        mock_vs_cls.return_value = mock_vs

        success = processor._process_job(job)

        assert success is True
        # Classifier SHOULD be instantiated and called
        mock_classifier_cls.assert_called_once()
        mock_classifier.classify.assert_called_once()
        # VectorStore config should have the classifier result as collection
        mock_vs_cls.assert_called_once()
        call_kwargs = mock_vs_cls.call_args.kwargs
        assert call_kwargs["config"].collection_name == "ml-ai"
