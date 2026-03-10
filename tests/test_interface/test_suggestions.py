"""Tests for the suggestions interface module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from personal_knowledge_base.interface.suggestions import (
    DEFAULT_COLLECTIONS,
    Suggestion,
    SuggestionResult,
    SuggestionsConfig,
    SuggestionsEngine,
)
from personal_knowledge_base.storage.vector_store import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_VECTOR = [0.1] * 768


def _make_search_result(text: str, score: float) -> SearchResult:
    return SearchResult(score=score, chunk_text=text, metadata={})


def _make_mock_embedder(vector: list[float] = DUMMY_VECTOR) -> MagicMock:
    """Return a mock OllamaEmbedder whose embed_text is an async function."""

    async def _async_embed(text: str) -> list[float]:  # noqa: ARG001
        return vector

    mock_emb = MagicMock()
    mock_emb.embed_text = MagicMock(side_effect=_async_embed)
    return mock_emb


# ---------------------------------------------------------------------------
# SuggestionsConfig
# ---------------------------------------------------------------------------


class TestSuggestionsConfig:
    def test_defaults(self) -> None:
        cfg = SuggestionsConfig()
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.ollama_url == "http://localhost:11434"
        assert cfg.embedding_model == "nomic-embed-text"
        assert cfg.max_suggestions == 5
        assert cfg.min_score == 0.3
        assert cfg.gap_threshold == 0.5

    def test_custom_values(self) -> None:
        cfg = SuggestionsConfig(max_suggestions=3, min_score=0.4)
        assert cfg.max_suggestions == 3
        assert cfg.min_score == 0.4


# ---------------------------------------------------------------------------
# Suggestion / SuggestionResult
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_suggestion_fields(self) -> None:
        s = Suggestion(text="hello", source="related", score=0.8, collection="papers")
        assert s.text == "hello"
        assert s.source == "related"
        assert s.score == 0.8
        assert s.collection == "papers"

    def test_suggestion_result_defaults(self) -> None:
        r = SuggestionResult(query="test")
        assert r.related == []
        assert r.gaps == []
        assert r.summary == ""

    def test_suggestion_result_with_data(self) -> None:
        s = Suggestion(text="foo", source="gap", score=0.2, collection="code")
        r = SuggestionResult(query="test", gaps=[s])
        assert len(r.gaps) == 1


# ---------------------------------------------------------------------------
# _extract_subtopics
# ---------------------------------------------------------------------------


class TestExtractSubtopics:
    def setup_method(self) -> None:
        self.engine = SuggestionsEngine()

    def test_split_on_and(self) -> None:
        topics = self.engine._extract_subtopics("machine learning and neural networks")
        assert len(topics) >= 1
        full = " ".join(topics)
        assert "machine" in full or "learning" in full
        assert "neural" in full or "networks" in full

    def test_split_on_or(self) -> None:
        topics = self.engine._extract_subtopics("docker or kubernetes deployment")
        assert len(topics) >= 1

    def test_split_on_with(self) -> None:
        topics = self.engine._extract_subtopics("python with fastapi")
        assert len(topics) >= 1

    def test_split_on_using(self) -> None:
        topics = self.engine._extract_subtopics("data pipeline using spark")
        assert len(topics) >= 1

    def test_split_on_about(self) -> None:
        topics = self.engine._extract_subtopics("tutorial about transformers")
        assert len(topics) >= 1

    def test_split_on_for(self) -> None:
        topics = self.engine._extract_subtopics("tools for distributed systems")
        assert len(topics) >= 1

    def test_max_three_subtopics(self) -> None:
        topics = self.engine._extract_subtopics("alpha and beta and gamma and delta and epsilon")
        assert len(topics) <= 3

    def test_short_words_excluded(self) -> None:
        # "AI" and "ML" are ≤3 chars → filtered out; result may be empty
        topics = self.engine._extract_subtopics("AI and ML")
        for t in topics:
            for word in t.split():
                assert len(word) > 3

    def test_stop_words_excluded(self) -> None:
        topics = self.engine._extract_subtopics("using the about for and")
        assert len(topics) == 0 or all(
            not any(w in ["the", "for", "and"] for w in t.split()) for t in topics
        )

    def test_single_topic_no_connectors(self) -> None:
        topics = self.engine._extract_subtopics("reinforcement learning")
        assert len(topics) == 1
        assert "reinforcement" in topics[0]

    def test_deduplication(self) -> None:
        topics = self.engine._extract_subtopics("transformer models and transformer models")
        assert len(topics) == 1


# ---------------------------------------------------------------------------
# _find_related
# ---------------------------------------------------------------------------


class TestFindRelated:
    def setup_method(self) -> None:
        self.engine = SuggestionsEngine(SuggestionsConfig(max_suggestions=3))

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    def test_returns_suggestions_sorted_by_score(self, mock_vs: MagicMock) -> None:
        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = [
            _make_search_result("doc A", 0.9),
            _make_search_result("doc B", 0.6),
        ]

        results = self.engine._find_related(DUMMY_VECTOR, ["papers"])

        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[0].source == "related"
        assert results[0].collection == "papers"

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    def test_respects_max_suggestions(self, mock_vs: MagicMock) -> None:
        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = [
            _make_search_result(f"doc {i}", 0.9 - i * 0.1) for i in range(10)
        ]

        results = self.engine._find_related(DUMMY_VECTOR, ["papers"])
        assert len(results) <= self.engine.config.max_suggestions

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    def test_aggregates_across_collections(self, mock_vs: MagicMock) -> None:
        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = [_make_search_result("doc", 0.8)]

        results = self.engine._find_related(DUMMY_VECTOR, ["papers", "videos"])
        # Two collections × 1 result each = 2, capped at max_suggestions=3
        assert len(results) == 2

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    def test_empty_collections_returns_empty(self, mock_vs: MagicMock) -> None:
        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = []

        results = self.engine._find_related(DUMMY_VECTOR, ["papers"])
        assert results == []

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    def test_connect_exception_skipped(self, mock_vs: MagicMock) -> None:
        instance = MagicMock()
        mock_vs.return_value = instance
        instance.connect.side_effect = ConnectionError("qdrant down")

        results = self.engine._find_related(DUMMY_VECTOR, ["papers"])
        assert results == []

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    def test_search_exception_skipped(self, mock_vs: MagicMock) -> None:
        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.side_effect = RuntimeError("search failed")

        results = self.engine._find_related(DUMMY_VECTOR, ["papers"])
        assert results == []


# ---------------------------------------------------------------------------
# _find_gaps
# ---------------------------------------------------------------------------


class TestFindGaps:
    def setup_method(self) -> None:
        self.engine = SuggestionsEngine(SuggestionsConfig(gap_threshold=0.5, min_score=0.3))

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_gap_when_low_score(self, mock_emb_cls: MagicMock, mock_vs: MagicMock) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = [_make_search_result("weak match", 0.2)]

        gaps = self.engine._find_gaps(
            "machine learning and neural networks", DUMMY_VECTOR, ["papers"]
        )
        assert len(gaps) >= 1
        assert all(g.source == "gap" for g in gaps)
        assert all(g.score < self.engine.config.gap_threshold for g in gaps)

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_no_gap_when_high_score(self, mock_emb_cls: MagicMock, mock_vs: MagicMock) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = [_make_search_result("strong match", 0.9)]

        gaps = self.engine._find_gaps(
            "machine learning and neural networks", DUMMY_VECTOR, ["papers"]
        )
        assert gaps == []

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_empty_kb_all_gaps(self, mock_emb_cls: MagicMock, mock_vs: MagicMock) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = []

        gaps = self.engine._find_gaps(
            "transformers and attention mechanisms", DUMMY_VECTOR, ["papers"]
        )
        assert len(gaps) >= 1

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_exception_during_gap_search_skipped(
        self, mock_emb_cls: MagicMock, mock_vs: MagicMock
    ) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.connect.side_effect = ConnectionError("down")

        gaps = self.engine._find_gaps(
            "machine learning and deep learning", DUMMY_VECTOR, ["papers"]
        )
        assert all(g.source == "gap" for g in gaps)


# ---------------------------------------------------------------------------
# format_suggestions
# ---------------------------------------------------------------------------


class TestFormatSuggestions:
    def setup_method(self) -> None:
        self.engine = SuggestionsEngine()

    def test_with_related_and_gaps(self) -> None:
        result = SuggestionResult(
            query="transformers",
            related=[Suggestion("paper A", "related", 0.85, "papers")],
            gaps=[Suggestion("attention heads", "gap", 0.2, "code")],
        )
        text = self.engine.format_suggestions(result)
        assert "transformers" in text
        assert "paper A" in text
        assert "attention heads" in text
        assert "0.85" in text
        assert "0.20" in text

    def test_no_related(self) -> None:
        result = SuggestionResult(query="obscure topic", related=[], gaps=[])
        text = self.engine.format_suggestions(result)
        assert "No related content found" in text

    def test_no_gaps(self) -> None:
        result = SuggestionResult(
            query="well-covered topic",
            related=[Suggestion("doc", "related", 0.9, "general")],
            gaps=[],
        )
        text = self.engine.format_suggestions(result)
        assert "No significant knowledge gaps" in text

    def test_output_is_string(self) -> None:
        result = SuggestionResult(query="test")
        text = self.engine.format_suggestions(result)
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# suggest() — integration-level (all external calls mocked)
# ---------------------------------------------------------------------------


class TestSuggest:
    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_suggest_good_coverage(self, mock_emb_cls: MagicMock, mock_vs: MagicMock) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = [
            _make_search_result("relevant doc", 0.9),
        ]

        engine = SuggestionsEngine(SuggestionsConfig(gap_threshold=0.5))
        result = engine.suggest("machine learning", collections=["papers"])

        assert isinstance(result, SuggestionResult)
        assert result.query == "machine learning"
        assert len(result.related) >= 1
        assert result.summary != ""

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_suggest_poor_coverage_creates_gaps(
        self, mock_emb_cls: MagicMock, mock_vs: MagicMock
    ) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance

        # Simulate Qdrant honouring score_threshold: only return result if
        # score >= score_threshold (or threshold is None/unset).
        def _search_side_effect(**kwargs: object) -> list[SearchResult]:
            threshold = kwargs.get("score_threshold") or 0.0
            hit = _make_search_result("weak", 0.1)
            return [hit] if hit.score >= float(threshold) else []

        instance.search.side_effect = _search_side_effect

        engine = SuggestionsEngine(SuggestionsConfig(min_score=0.3, gap_threshold=0.5))
        result = engine.suggest("quantum computing and cryptography", collections=["papers"])

        # min_score=0.3 passed as score_threshold → mock filters the 0.1-score doc
        assert result.related == []
        # gap_threshold=0.5 > 0.1 → gaps found
        assert len(result.gaps) >= 1

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_suggest_empty_kb(self, mock_emb_cls: MagicMock, mock_vs: MagicMock) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = []

        engine = SuggestionsEngine()
        result = engine.suggest(
            "reinforcement learning and policy gradients",
            collections=["papers"],
        )

        assert result.related == []
        assert len(result.gaps) >= 1

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_suggest_uses_default_collections(
        self, mock_emb_cls: MagicMock, mock_vs: MagicMock
    ) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = []

        engine = SuggestionsEngine()
        engine.suggest("test query")

        assert mock_vs.call_count >= len(DEFAULT_COLLECTIONS)

    @patch("personal_knowledge_base.interface.suggestions.VectorStore")
    @patch("personal_knowledge_base.interface.suggestions.OllamaEmbedder")
    def test_suggest_max_suggestions_limit(
        self, mock_emb_cls: MagicMock, mock_vs: MagicMock
    ) -> None:
        mock_emb_cls.return_value = _make_mock_embedder()

        instance = MagicMock()
        mock_vs.return_value = instance
        instance.search.return_value = [
            _make_search_result(f"doc {i}", 0.9 - i * 0.01) for i in range(20)
        ]

        engine = SuggestionsEngine(SuggestionsConfig(max_suggestions=3))
        result = engine.suggest("python asyncio", collections=["code", "papers"])

        assert len(result.related) <= 3
