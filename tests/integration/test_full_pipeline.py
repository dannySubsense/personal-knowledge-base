"""Integration tests for the full PKB pipeline.

All external services (Qdrant, Ollama, SQLite, HTTP) are mocked so these
tests can run without any infrastructure.  The focus is on the real code
paths connecting multiple modules together.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from personal_knowledge_base.batch.processor import BatchConfig, BatchProcessor
from personal_knowledge_base.fetchers.base import FetchResult
from personal_knowledge_base.interface.query import KBQueryInterface, QueryConfig, QueryResult
from personal_knowledge_base.interface.suggestions import SuggestionsConfig, SuggestionsEngine
from personal_knowledge_base.interface.whatsapp import WhatsAppConfig, WhatsAppHandler
from personal_knowledge_base.processing.chunker import Chunk
from personal_knowledge_base.processing.classifier import ClassifierConfig, ContentClassifier
from personal_knowledge_base.processing.embedder import EmbeddingResult
from personal_knowledge_base.processing.staleness import StalenessConfig, StalenessDetector
from personal_knowledge_base.queue.models import Job
from personal_knowledge_base.storage.vector_store import SearchResult

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 768
_FAKE_EMBEDDING: list[float] = [0.1] * _EMBEDDING_DIM
_FAKE_EMBEDDING_ALT: list[float] = [0.2] * _EMBEDDING_DIM


def _make_job(
    job_id: str = "job-001",
    url: str = "https://example.com/article",
    priority: int = 2,
    status: str = "pending",
    retry_count: int = 0,
) -> Job:
    return Job(
        id=job_id,
        url=url,
        priority=priority,
        status=status,
        retry_count=retry_count,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_fetch_result(url: str, success: bool = True, title: str = "Test Title") -> FetchResult:
    if success:
        return FetchResult(
            url=url,
            title=title,
            content="Deep learning fundamentals: layers, activations, backprop. " * 15,
            content_type="article",
            success=True,
        )
    return FetchResult(url=url, success=False, error_message="HTTP error 404")


def _make_chunks(url: str = "https://example.com", n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            text=f"Chunk {i}: factor model content about equities and returns.",
            index=i,
            total=n,
            source=url,
            metadata={"token_estimate": 30},
        )
        for i in range(n)
    ]


def _make_embedding_results(chunks: list[Chunk]) -> list[EmbeddingResult]:
    return [
        EmbeddingResult(
            chunk_text=c.text,
            embedding_vector=_FAKE_EMBEDDING[:],
            metadata=c.metadata,
        )
        for c in chunks
    ]


def _make_search_result(
    text: str, score: float = 0.85, source: str = "https://example.com"
) -> SearchResult:
    return SearchResult(
        score=score,
        chunk_text=text,
        metadata={"source": source, "doc_id": "doc-001", "title": "Test"},
    )


# ---------------------------------------------------------------------------
# Scenario 1: Full ingestion pipeline
# WhatsApp → BatchProcessor → Chunker → Embedder → Classifier → VectorStore
# ---------------------------------------------------------------------------


class TestFullIngestionPipeline:
    """Scenario 1: End-to-end ingestion from URL discovery to vector storage."""

    def test_whatsapp_url_queued_then_batch_processed(self) -> None:
        """URL extracted from WhatsApp message is eventually stored in Qdrant."""
        article_url = "https://arxiv.org/abs/2301.00001"

        job = _make_job(url=article_url)
        chunks = _make_chunks(url=article_url, n=3)
        embedding_results = _make_embedding_results(chunks)

        with (
            # Queue layer — in-memory DB
            patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[]),
            patch(
                "personal_knowledge_base.interface.whatsapp.add_job", return_value=job
            ) as mock_add_job,
            # Batch processor queue
            patch("personal_knowledge_base.batch.processor.get_next_job", side_effect=[job, None]),
            patch(
                "personal_knowledge_base.batch.processor.update_job_status"
            ) as mock_update_status,
            # Fetcher
            patch.object(
                __import__(
                    "personal_knowledge_base.fetchers.web",
                    fromlist=["WebFetcher"],
                ).WebFetcher,
                "fetch",
                return_value=_make_fetch_result(article_url, title="Factor Models in Finance"),
            ),
            # Chunker — real chunker, but monkey-patch for determinism
            patch("personal_knowledge_base.batch.processor.Chunker") as mock_chunker_cls,
            # Embedder
            patch("personal_knowledge_base.batch.processor.OllamaEmbedder") as mock_embedder_cls,
            # Classifier — URL rule routes arxiv → papers
            patch(
                "personal_knowledge_base.batch.processor.ContentClassifier"
            ) as mock_classifier_cls,
            # VectorStore
            patch("personal_knowledge_base.storage.vector_store.QdrantClient") as mock_qdrant_cls,
        ):
            # Wire up chunker mock
            mock_chunker_inst = MagicMock()
            mock_chunker_inst.chunk.return_value = chunks
            mock_chunker_cls.return_value = mock_chunker_inst

            # Wire up embedder mock (async context manager)
            mock_embedder_inst = AsyncMock()
            mock_embedder_inst.embed_chunks = AsyncMock(return_value=embedding_results)
            mock_embedder_inst.__aenter__ = AsyncMock(return_value=mock_embedder_inst)
            mock_embedder_inst.__aexit__ = AsyncMock(return_value=False)
            mock_embedder_cls.return_value = mock_embedder_inst

            # Wire up classifier mock
            mock_classifier_inst = MagicMock()
            mock_classifier_inst.classify.return_value = "ml-ai"
            mock_classifier_cls.return_value = mock_classifier_inst

            # Wire up Qdrant mock
            mock_qdrant_inst = MagicMock()
            mock_qdrant_inst.get_collections.return_value = MagicMock(collections=[])
            mock_qdrant_inst.collection_exists.return_value = False
            mock_qdrant_cls.return_value = mock_qdrant_inst

            # Step 1: WhatsApp handler queues the URL
            handler = WhatsAppHandler(WhatsAppConfig(trusted_senders=["+16039885837"]))
            result = handler.handle_message("+16039885837", f"Check this paper: {article_url}")

            assert result.urls_found == 1
            assert result.queued == 1
            assert result.duplicates == 0
            mock_add_job.assert_called_once_with(url=article_url, priority=2)

            # Step 2: Batch processor runs
            processor = BatchProcessor(config=BatchConfig(max_jobs_per_run=5))
            batch_result = processor.run()

            assert batch_result.processed == 1
            assert batch_result.succeeded == 1
            assert batch_result.failed == 0

            # Verify the pipeline called each component
            mock_chunker_inst.chunk.assert_called_once()
            mock_embedder_inst.embed_chunks.assert_awaited_once()
            mock_classifier_inst.classify.assert_called_once()
            mock_qdrant_inst.upsert.assert_called()

            # Job marked done
            status_calls = [call.args[1] for call in mock_update_status.call_args_list]
            assert "processing" in status_calls
            assert "done" in status_calls


# ---------------------------------------------------------------------------
# Scenario 2: Query pipeline
# KBQueryInterface → OllamaEmbedder → VectorStore → QueryResult
# ---------------------------------------------------------------------------


class TestQueryPipeline:
    """Scenario 2: Full query flow from question text to ranked results."""

    def test_query_returns_results_from_collection(self) -> None:
        """query() embeds the question and returns matching chunks."""
        question = "factor models"

        search_hits = [
            _make_search_result("Fama-French three-factor model explanation.", score=0.91),
            _make_search_result("CAPM and systematic risk factors.", score=0.83),
        ]

        with (
            patch("personal_knowledge_base.interface.query.OllamaEmbedder") as mock_embedder_cls,
            patch("personal_knowledge_base.interface.query.VectorStore") as mock_vectorstore_cls,
        ):
            # Embedder returns a fixed vector
            mock_emb = AsyncMock()
            mock_emb.embed_text = AsyncMock(return_value=_FAKE_EMBEDDING)
            mock_emb.__aenter__ = AsyncMock(return_value=mock_emb)
            mock_emb.__aexit__ = AsyncMock(return_value=False)
            mock_embedder_cls.return_value = mock_emb

            # VectorStore returns our search hits
            mock_store = MagicMock()
            mock_store.search.return_value = search_hits
            mock_store.connect = MagicMock()
            mock_store.disconnect = MagicMock()
            mock_vectorstore_cls.return_value = mock_store

            iface = KBQueryInterface(QueryConfig(default_collection="quant-trading"))
            results = iface.query(question)

        assert len(results) == 2
        assert results[0].score == 0.91
        assert results[1].score == 0.83
        assert results[0].collection == "quant-trading"
        # Embedder was called with the question
        mock_emb.embed_text.assert_awaited_once_with(question)
        # VectorStore was searched with the returned embedding
        mock_store.search.assert_called_once_with(
            query_vector=_FAKE_EMBEDDING,
            limit=10,
            score_threshold=0.3,
        )

    def test_query_format_results(self) -> None:
        """format_results() produces human-readable text with source and score."""
        qr = QueryResult(
            chunk_text="Factor exposure to momentum.",
            score=0.78,
            source="https://arxiv.org/abs/2201.00001",
            doc_id="doc-123",
            collection="quant-trading",
        )
        iface = KBQueryInterface()
        text = iface.format_results([qr])
        assert "0.778" in text or "0.780" in text or "0.78" in text
        assert "quant-trading" in text
        assert "arxiv.org" in text


# ---------------------------------------------------------------------------
# Scenario 3: Cross-collection query
# query_all_collections → merge + re-rank by score
# ---------------------------------------------------------------------------


class TestCrossCollectionQuery:
    """Scenario 3: query_all_collections merges and re-ranks across collections."""

    def test_results_merged_and_sorted_by_score(self) -> None:
        """Results from all collections are merged and sorted descending."""
        quant_hits = [
            _make_search_result(
                "Factor model backtesting strategy.", score=0.72, source="https://quantopian.com/1"
            )
        ]
        ml_hits = [
            _make_search_result(
                "Attention is all you need.", score=0.95, source="https://arxiv.org/abs/1706"
            )
        ]
        general_hits: list[SearchResult] = []

        collection_hits = {
            "quant-trading": quant_hits,
            "ml-ai": ml_hits,
            "general": general_hits,
        }

        with (
            patch("personal_knowledge_base.interface.query.OllamaEmbedder") as mock_embedder_cls,
            patch("personal_knowledge_base.interface.query.VectorStore") as mock_vectorstore_cls,
            patch(
                "personal_knowledge_base.interface.query.KBQueryInterface._load_collection_ids",
                return_value=["quant-trading", "ml-ai", "general"],
            ),
        ):
            mock_emb = AsyncMock()
            mock_emb.embed_text = AsyncMock(return_value=_FAKE_EMBEDDING)
            mock_emb.__aenter__ = AsyncMock(return_value=mock_emb)
            mock_emb.__aexit__ = AsyncMock(return_value=False)
            mock_embedder_cls.return_value = mock_emb

            def store_factory(config=None):
                store = MagicMock()
                store.connect = MagicMock()
                store.disconnect = MagicMock()
                coll = config.collection_name if config else "general"
                store.search.return_value = collection_hits.get(coll, [])
                return store

            mock_vectorstore_cls.side_effect = store_factory

            iface = KBQueryInterface()
            results = iface.query_all_collections("neural networks")

        assert len(results) == 2  # general returned empty
        # Should be sorted descending by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].score == 0.95
        assert results[1].score == 0.72


# ---------------------------------------------------------------------------
# Scenario 4: Content classification routing
# ---------------------------------------------------------------------------


class TestContentClassificationRouting:
    """Scenario 4: URLs are routed to the correct topic KB."""

    def test_quant_url_classified_as_quant_trading(self) -> None:
        """URL with quant/trading/finance signals → quant-trading (URL hint tier)."""
        classifier = ContentClassifier(config=ClassifierConfig())
        assert (
            classifier.classify("https://quantopian.com/research/factor-models") == "quant-trading"
        )

    def test_finance_url_classified_as_quant_trading(self) -> None:
        """URL containing 'finance' → quant-trading via URL hint."""
        classifier = ContentClassifier(config=ClassifierConfig())
        assert (
            classifier.classify("https://alphaarchitect.com/finance/factor-investing")
            == "quant-trading"
        )

    def test_youtube_trading_video_classified_as_quant_trading(self) -> None:
        """YouTube video about trading → quant-trading via content similarity."""
        with patch.object(ContentClassifier, "_embed") as mock_embed:
            # Simulate embedder returning vectors: quant-trading description scores highest
            quant_vector = [1.0, 0.0, 0.0]
            mock_embed.return_value = quant_vector

            classifier = ContentClassifier(config=ClassifierConfig(similarity_threshold=0.3))
            # Patch _desc_embeddings to control similarity outcome
            classifier._desc_embeddings = {
                "quant-trading": [1.0, 0.0, 0.0],  # cosine = 1.0
                "ml-ai": [0.0, 1.0, 0.0],  # cosine = 0.0
                "general": [0.0, 0.0, 1.0],  # cosine = 0.0
            }
            result = classifier.classify(
                url="https://www.youtube.com/watch?v=abc123",
                title="Introduction to quantitative trading strategies",
                description="Learn factor models and backtesting",
            )
        assert result == "quant-trading"

    def test_arxiv_ml_paper_classified_as_ml_ai(self) -> None:
        """arXiv paper about ML → ml-ai via content similarity."""
        with patch.object(ContentClassifier, "_embed") as mock_embed:
            ml_vector = [0.0, 1.0, 0.0]
            mock_embed.return_value = ml_vector

            classifier = ContentClassifier(config=ClassifierConfig(similarity_threshold=0.3))
            classifier._desc_embeddings = {
                "quant-trading": [1.0, 0.0, 0.0],  # cosine = 0.0
                "ml-ai": [0.0, 1.0, 0.0],  # cosine = 1.0
                "general": [0.0, 0.0, 1.0],  # cosine = 0.0
            }
            result = classifier.classify(
                url="https://arxiv.org/abs/2301.00001",
                title="Attention Is All You Need: Transformer Models for NLP",
                description="Deep learning transformer architecture for sequence modelling",
            )
        assert result == "ml-ai"

    def test_github_ml_repo_classified_as_ml_ai(self) -> None:
        """GitHub ML repo → ml-ai via content similarity."""
        with patch.object(ContentClassifier, "_embed") as mock_embed:
            ml_vector = [0.0, 1.0, 0.0]
            mock_embed.return_value = ml_vector

            classifier = ContentClassifier(config=ClassifierConfig(similarity_threshold=0.3))
            classifier._desc_embeddings = {
                "quant-trading": [1.0, 0.0, 0.0],
                "ml-ai": [0.0, 1.0, 0.0],
                "general": [0.0, 0.0, 1.0],
            }
            result = classifier.classify(
                url="https://github.com/huggingface/transformers",
                title="Transformers: State-of-the-art Machine Learning for PyTorch and TF",
                description="Thousands of pretrained models for NLP, vision and audio",
            )
        assert result == "ml-ai"

    def test_unknown_url_classified_by_content_similarity(self) -> None:
        """Unknown URL falls through to Tier-2 embedding similarity or general fallback."""
        with patch.object(ContentClassifier, "_embed") as mock_embed:
            # Zero embeddings → cosine similarity = 0.0 < threshold → falls back to general
            mock_embed.return_value = [0.0] * _EMBEDDING_DIM

            classifier = ContentClassifier(config=ClassifierConfig(similarity_threshold=0.3))
            result = classifier.classify(
                url="https://someunknownsite.com/page",
                title="Random blog post",
                description="",
            )
        assert result == "general"


# ---------------------------------------------------------------------------
# Scenario 5: Staleness detection pipeline
# ---------------------------------------------------------------------------


class TestStalenessPipeline:
    """Scenario 5: Staleness scoring computes correct sub-scores and flags stale content."""

    def test_old_ai_content_flagged_stale(self) -> None:
        """3-year-old AI content from a blog exceeds the staleness threshold."""
        old_date = datetime.now(UTC) - timedelta(days=3 * 365)
        detector = StalenessDetector(config=StalenessConfig(stale_threshold=0.7))
        result = detector.score(
            content_date=old_date,
            topic_keywords=["ai", "llm", "neural"],
            source_url="https://medium.com/blog/ai-trends",
        )

        assert result.age_score == 1.0  # > 2 years
        assert result.topic_score == 0.9  # AI/LLM = highly volatile
        assert result.source_score == 0.6  # medium.com = blog
        assert result.is_stale is True
        assert result.score >= 0.7

    def test_recent_math_content_is_fresh(self) -> None:
        """Recent timeless content should NOT be stale."""
        recent_date = datetime.now(UTC) - timedelta(days=10)
        detector = StalenessDetector()
        result = detector.score(
            content_date=recent_date,
            topic_keywords=["mathematics", "statistics"],
            source_url="https://arxiv.org/abs/2401.12345",
        )

        assert result.age_score == 0.0  # < 30 days
        assert result.topic_score == 0.1  # timeless
        assert result.source_score == 0.2  # arxiv = stable
        assert result.is_stale is False

    def test_weighted_combination_is_correct(self) -> None:
        """The combined score is the correct weighted sum of sub-scores."""
        old_date = datetime.now(UTC) - timedelta(days=400)
        config = StalenessConfig(
            age_weight=0.4,
            topic_weight=0.4,
            source_weight=0.2,
            stale_threshold=0.99,  # Never flag stale; just check math
        )
        detector = StalenessDetector(config=config)
        result = detector.score(
            content_date=old_date,
            topic_keywords=["transformer"],
            source_url="https://arxiv.org/abs/1706.03762",
        )

        expected = 0.4 * result.age_score + 0.4 * result.topic_score + 0.2 * result.source_score
        assert abs(result.score - expected) < 1e-9


# ---------------------------------------------------------------------------
# Scenario 6: Suggestion pipeline
# SuggestionsEngine.suggest → related content + gap identification
# ---------------------------------------------------------------------------


class TestSuggestionPipeline:
    """Scenario 6: Suggestions engine returns related content and gaps."""

    def test_suggest_returns_related_and_gaps(self) -> None:
        """suggest() finds related content and identifies knowledge gaps."""
        query = "transformer architectures"

        related_hit = _make_search_result(
            "Attention mechanism in transformer models.",
            score=0.88,
            source="https://arxiv.org/abs/1706.03762",
        )

        with (
            patch(
                "personal_knowledge_base.interface.suggestions.OllamaEmbedder"
            ) as mock_embedder_cls,
            patch(
                "personal_knowledge_base.interface.suggestions.VectorStore"
            ) as mock_vectorstore_cls,
        ):
            # Two calls to OllamaEmbedder — first for main query, then per subtopic
            mock_emb = AsyncMock()
            mock_emb.embed_text = AsyncMock(return_value=_FAKE_EMBEDDING)
            mock_emb.__aenter__ = AsyncMock(return_value=mock_emb)
            mock_emb.__aexit__ = AsyncMock(return_value=False)
            mock_embedder_cls.return_value = mock_emb

            call_count = {"n": 0}

            def store_factory(config=None):
                store = MagicMock()
                store.connect = MagicMock()
                store.disconnect = MagicMock()
                call_count["n"] += 1
                # First few calls return results (related search); later calls return empty (gaps)
                if call_count["n"] <= 2:
                    store.search.return_value = [related_hit]
                else:
                    store.search.return_value = []
                return store

            mock_vectorstore_cls.side_effect = store_factory

            engine = SuggestionsEngine(config=SuggestionsConfig(min_score=0.3, gap_threshold=0.5))
            result = engine.suggest(query)

        assert result.query == query
        assert len(result.related) >= 1
        assert result.related[0].score >= 0.3
        assert result.summary != ""
        # Summary contains the query
        assert query in result.summary

    def test_suggest_formats_output(self) -> None:
        """format_suggestions() produces sections for related and gaps."""
        engine = SuggestionsEngine()

        from personal_knowledge_base.interface.suggestions import Suggestion, SuggestionResult

        fake_result = SuggestionResult(
            query="neural networks",
            related=[
                Suggestion(
                    text="Backpropagation explained.",
                    source="related",
                    score=0.82,
                    collection="ml-ai",
                )
            ],
            gaps=[
                Suggestion(text="vanishing gradient", source="gap", score=0.25, collection="ml-ai")
            ],
        )
        formatted = engine.format_suggestions(fake_result)
        assert "Related content" in formatted
        assert "knowledge gaps" in formatted
        assert "Backpropagation" in formatted
        assert "vanishing gradient" in formatted


# ---------------------------------------------------------------------------
# Scenario 7: Error recovery in BatchProcessor
# ---------------------------------------------------------------------------


class TestBatchProcessorErrorRecovery:
    """Scenario 7: A failing job is marked failed while other jobs continue."""

    def test_failed_job_does_not_block_successful_jobs(self) -> None:
        """One failing job increments failed count; other jobs still succeed."""
        failing_job = _make_job("job-fail", url="https://badsite.example.com/404")
        good_job = _make_job("job-ok", url="https://arxiv.org/abs/2301.00001")
        good_chunks = _make_chunks(url=good_job.url, n=2)
        good_embeddings = _make_embedding_results(good_chunks)

        jobs_iter = iter([failing_job, good_job, None])

        with (
            patch("personal_knowledge_base.batch.processor.get_next_job", side_effect=jobs_iter),
            patch("personal_knowledge_base.batch.processor.update_job_status") as mock_update,
            patch("personal_knowledge_base.batch.processor.Chunker") as mock_chunker_cls,
            patch("personal_knowledge_base.batch.processor.OllamaEmbedder") as mock_embedder_cls,
            patch(
                "personal_knowledge_base.batch.processor.ContentClassifier"
            ) as mock_classifier_cls,
            patch("personal_knowledge_base.storage.vector_store.QdrantClient") as mock_qdrant_cls,
        ):
            # Fetcher for bad URL raises; fetcher for good URL succeeds
            def fetcher_fetch_side_effect(url: str) -> FetchResult:
                if "badsite" in url:
                    raise RuntimeError("Connection refused")
                return _make_fetch_result(url)

            with patch(
                "personal_knowledge_base.batch.processor.BatchProcessor._get_fetcher"
            ) as mock_get_fetcher:
                bad_fetcher = MagicMock()
                bad_fetcher.fetch.side_effect = RuntimeError("Connection refused")
                good_fetcher = MagicMock()
                good_fetcher.fetch.return_value = _make_fetch_result(good_job.url)

                def _fetcher_factory(url):
                    return bad_fetcher if "badsite" in url else good_fetcher

                mock_get_fetcher.side_effect = _fetcher_factory

                # Chunker
                mock_chunker = MagicMock()
                mock_chunker.chunk.return_value = good_chunks
                mock_chunker_cls.return_value = mock_chunker

                # Embedder
                mock_emb = AsyncMock()
                mock_emb.embed_chunks = AsyncMock(return_value=good_embeddings)
                mock_emb.__aenter__ = AsyncMock(return_value=mock_emb)
                mock_emb.__aexit__ = AsyncMock(return_value=False)
                mock_embedder_cls.return_value = mock_emb

                # Classifier
                mock_cls = MagicMock()
                mock_cls.classify.return_value = "ml-ai"
                mock_classifier_cls.return_value = mock_cls

                # Qdrant
                mock_qdrant = MagicMock()
                mock_qdrant.get_collections.return_value = MagicMock(collections=[])
                mock_qdrant.collection_exists.return_value = False
                mock_qdrant_cls.return_value = mock_qdrant

                processor = BatchProcessor(config=BatchConfig(max_jobs_per_run=10))
                result = processor.run()

        assert result.processed == 2
        assert result.succeeded == 1
        assert result.failed == 1
        assert result.skipped == 0

        # The failing job must have been marked failed
        all_status_calls = [(c.args[0], c.args[1]) for c in mock_update.call_args_list]
        failed_calls = [s for jid, s in all_status_calls if jid == failing_job.id and s == "failed"]
        assert len(failed_calls) == 1

    def test_retry_limit_exceeded_marks_skipped(self) -> None:
        """A job that has exhausted retries is skipped, not processed."""
        exhausted_job = _make_job("job-exhaust", retry_count=3)  # limit is 3
        jobs_iter = iter([exhausted_job, None])

        with (
            patch("personal_knowledge_base.batch.processor.get_next_job", side_effect=jobs_iter),
            patch("personal_knowledge_base.batch.processor.update_job_status") as mock_update,
        ):
            processor = BatchProcessor(config=BatchConfig(retry_limit=3))
            result = processor.run()

        assert result.processed == 1
        assert result.skipped == 1
        assert result.succeeded == 0
        # Status set to failed with "Exceeded retry limit"
        mock_update.assert_called_once_with(
            exhausted_job.id, "failed", error_message="Exceeded retry limit"
        )


# ---------------------------------------------------------------------------
# Scenario 8: WhatsApp duplicate detection
# ---------------------------------------------------------------------------


class TestWhatsAppDuplicateDetection:
    """Scenario 8: Same URL sent twice is detected as duplicate on second attempt."""

    def test_first_send_queues_second_is_duplicate(self) -> None:
        """First send queues the URL; identical second send is detected as duplicate."""
        url = "https://arxiv.org/abs/2310.99999"
        job = _make_job(url=url)

        # First call: no existing jobs → not a duplicate
        # Second call: the job from first send is present → duplicate
        list_jobs_side_effects = [
            [],  # first handle_message: is_duplicate returns False
            [job],  # second handle_message: is_duplicate returns True
        ]

        add_job_call_count = {"n": 0}

        def fake_add_job(url, priority=2):
            add_job_call_count["n"] += 1
            return job

        with (
            patch(
                "personal_knowledge_base.interface.whatsapp.list_jobs",
                side_effect=list_jobs_side_effects,
            ),
            patch("personal_knowledge_base.interface.whatsapp.add_job", side_effect=fake_add_job),
        ):
            handler = WhatsAppHandler(WhatsAppConfig(trusted_senders=["+16039885837"]))

            # First send
            r1 = handler.handle_message("+16039885837", f"Check this: {url}")
            assert r1.queued == 1
            assert r1.duplicates == 0

            # Second send — same URL
            r2 = handler.handle_message("+16039885837", f"Check this: {url}")
            assert r2.queued == 0
            assert r2.duplicates == 1

        # add_job should only have been called once (not for the duplicate)
        assert add_job_call_count["n"] == 1

    def test_duplicate_within_window_not_requeued(self) -> None:
        """URL queued 5 days ago is still within the 30-day window → duplicate."""
        url = "https://github.com/openai/gpt-2"
        recent_job = _make_job(url=url)
        # Simulate the job was created 5 days ago (within default 30-day window)
        recent_job.created_at = datetime.now(UTC) - timedelta(days=5)

        with (
            patch(
                "personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[recent_job]
            ),
            patch("personal_knowledge_base.interface.whatsapp.add_job") as mock_add_job,
        ):
            handler = WhatsAppHandler(WhatsAppConfig())
            result = handler.handle_message("+1234567890", f"See this repo: {url}")

        assert result.duplicates == 1
        assert result.queued == 0
        mock_add_job.assert_not_called()

    def test_url_outside_duplicate_window_is_requeued(self) -> None:
        """URL queued 40 days ago is outside the 30-day window → re-queued."""
        url = "https://github.com/pytorch/pytorch"
        old_job = _make_job(url=url)
        old_job.created_at = datetime.now(UTC) - timedelta(days=40)
        new_job = _make_job("job-new", url=url)

        with (
            patch("personal_knowledge_base.interface.whatsapp.list_jobs", return_value=[old_job]),
            patch("personal_knowledge_base.interface.whatsapp.add_job", return_value=new_job),
        ):
            handler = WhatsAppHandler(WhatsAppConfig())
            result = handler.handle_message("+1234567890", f"Check again: {url}")

        assert result.queued == 1
        assert result.duplicates == 0
