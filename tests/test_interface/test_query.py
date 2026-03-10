"""Tests for the KB query interface."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from personal_knowledge_base.interface.query import (
    KBQueryInterface,
    QueryConfig,
    QueryResult,
)
from personal_knowledge_base.storage.vector_store import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_VECTOR = [0.1] * 768


def make_search_result(
    chunk_text: str = "Test chunk",
    score: float = 0.8,
    source: str = "docs/test.md",
    doc_id: str = "doc-001",
) -> SearchResult:
    return SearchResult(
        score=score,
        chunk_text=chunk_text,
        metadata={"source": source, "doc_id": doc_id, "extra": "value"},
    )


# ---------------------------------------------------------------------------
# QueryConfig
# ---------------------------------------------------------------------------


def test_query_config_defaults() -> None:
    cfg = QueryConfig()
    assert cfg.qdrant_url == "http://localhost:6333"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.default_limit == 10
    assert cfg.default_score_threshold == 0.3
    assert cfg.default_collection == "general"


def test_query_config_custom() -> None:
    cfg = QueryConfig(
        qdrant_url="http://my-qdrant:6333",
        default_limit=5,
        default_collection="ml-ai",
    )
    assert cfg.qdrant_url == "http://my-qdrant:6333"
    assert cfg.default_limit == 5
    assert cfg.default_collection == "ml-ai"


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


def test_query_result_fields() -> None:
    result = QueryResult(
        chunk_text="Some text",
        score=0.9,
        source="path/to/file.pdf",
        doc_id="abc-123",
        collection="ml-ai",
        metadata={"page": 3},
    )
    assert result.chunk_text == "Some text"
    assert result.score == 0.9
    assert result.source == "path/to/file.pdf"
    assert result.doc_id == "abc-123"
    assert result.collection == "ml-ai"
    assert result.metadata == {"page": 3}


def test_query_result_default_metadata() -> None:
    result = QueryResult(
        chunk_text="text",
        score=0.5,
        source="src",
        doc_id="id",
        collection="col",
    )
    assert result.metadata == {}


# ---------------------------------------------------------------------------
# query() — single collection
# ---------------------------------------------------------------------------


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_returns_ranked_results(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    # Arrange
    mock_asyncio_run.return_value = SAMPLE_VECTOR

    mock_store = MagicMock()
    mock_vs_cls.return_value = mock_store
    mock_store.search.return_value = [
        make_search_result(chunk_text="chunk A", score=0.9),
        make_search_result(chunk_text="chunk B", score=0.7),
    ]

    iface = KBQueryInterface()

    # Act
    results = iface.query("what is X?", collection="general")

    # Assert
    assert len(results) == 2
    assert results[0].chunk_text == "chunk A"
    assert results[0].score == 0.9
    assert results[0].collection == "general"
    assert results[1].chunk_text == "chunk B"
    assert results[1].score == 0.7

    mock_store.connect.assert_called_once()
    mock_store.disconnect.assert_called_once()
    mock_store.search.assert_called_once_with(
        query_vector=SAMPLE_VECTOR,
        limit=10,
        score_threshold=0.3,
    )


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_empty_results(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR

    mock_store = MagicMock()
    mock_vs_cls.return_value = mock_store
    mock_store.search.return_value = []

    iface = KBQueryInterface()
    results = iface.query("nothing here")
    assert results == []


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_uses_default_collection(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR
    mock_store = MagicMock()
    mock_vs_cls.return_value = mock_store
    mock_store.search.return_value = []

    config = QueryConfig(default_collection="quant-trading")
    iface = KBQueryInterface(config=config)
    iface.query("test")

    # VectorStore should be constructed with collection_name="quant-trading"
    call_kwargs = mock_vs_cls.call_args
    vs_config = (
        call_kwargs[1]["config"] if "config" in (call_kwargs[1] or {}) else call_kwargs[0][0]
    )
    assert vs_config.collection_name == "quant-trading"


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_custom_limit_and_threshold(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR
    mock_store = MagicMock()
    mock_vs_cls.return_value = mock_store
    mock_store.search.return_value = []

    iface = KBQueryInterface()
    iface.query("test", limit=3, score_threshold=0.7)

    mock_store.search.assert_called_once_with(
        query_vector=SAMPLE_VECTOR,
        limit=3,
        score_threshold=0.7,
    )


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_metadata_extraction(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR
    mock_store = MagicMock()
    mock_vs_cls.return_value = mock_store
    mock_store.search.return_value = [make_search_result(source="path/file.md", doc_id="d-99")]

    iface = KBQueryInterface()
    results = iface.query("test")

    assert results[0].source == "path/file.md"
    assert results[0].doc_id == "d-99"
    # 'extra' key should be in metadata, not source/doc_id
    assert "extra" in results[0].metadata
    assert "source" not in results[0].metadata
    assert "doc_id" not in results[0].metadata


# ---------------------------------------------------------------------------
# score_threshold filtering behaviour (via store mock)
# ---------------------------------------------------------------------------


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_score_threshold_passed_to_store(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    """Score threshold is forwarded to VectorStore.search so the store filters."""
    mock_asyncio_run.return_value = SAMPLE_VECTOR
    mock_store = MagicMock()
    mock_vs_cls.return_value = mock_store
    # Simulate store returning only the high-score result
    mock_store.search.return_value = [make_search_result(score=0.9)]

    iface = KBQueryInterface()
    results = iface.query("test", score_threshold=0.8)

    mock_store.search.assert_called_once_with(
        query_vector=SAMPLE_VECTOR,
        limit=10,
        score_threshold=0.8,
    )
    assert len(results) == 1
    assert results[0].score == 0.9


# ---------------------------------------------------------------------------
# query_all_collections()
# ---------------------------------------------------------------------------


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_all_collections_merges_and_reranks(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR

    # Return different results per collection
    def store_side_effect(config: Any) -> MagicMock:
        mock_store = MagicMock()
        col = config.collection_name
        if col == "quant-trading":
            mock_store.search.return_value = [
                make_search_result(chunk_text="quant chunk", score=0.6)
            ]
        elif col == "ml-ai":
            mock_store.search.return_value = [make_search_result(chunk_text="ml chunk", score=0.85)]
        else:
            mock_store.search.return_value = []
        return mock_store

    mock_vs_cls.side_effect = lambda config: store_side_effect(config)

    iface = KBQueryInterface()
    results = iface.query_all_collections("test", collections=["quant-trading", "ml-ai"])

    assert len(results) == 2
    # Re-ranked: ml-ai (0.85) before quant-trading (0.6)
    assert results[0].chunk_text == "ml chunk"
    assert results[0].score == 0.85
    assert results[1].chunk_text == "quant chunk"
    assert results[1].score == 0.6


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_all_collections_default_collections(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR

    searched_collections: list[str] = []

    def store_side_effect(config: Any) -> MagicMock:
        searched_collections.append(config.collection_name)
        mock_store = MagicMock()
        mock_store.search.return_value = []
        return mock_store

    mock_vs_cls.side_effect = lambda config: store_side_effect(config)

    with patch(
        "personal_knowledge_base.interface.query.KBQueryInterface._load_collection_ids",
        return_value=["quant-trading", "ml-ai", "general"],
    ):
        iface = KBQueryInterface()
        iface.query_all_collections("test")

    assert set(searched_collections) == {"quant-trading", "ml-ai", "general"}


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_all_collections_skips_failing_collection(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR

    def store_side_effect(config: Any) -> MagicMock:
        mock_store = MagicMock()
        if config.collection_name == "quant-trading":
            mock_store.connect.side_effect = ConnectionError("Qdrant unreachable")
        else:
            mock_store.search.return_value = [make_search_result(score=0.75)]
        return mock_store

    mock_vs_cls.side_effect = lambda config: store_side_effect(config)

    iface = KBQueryInterface()
    results = iface.query_all_collections("test", collections=["general", "quant-trading"])

    # Only results from "general" — "quant-trading" was skipped
    assert all(r.collection == "general" for r in results)


@patch("personal_knowledge_base.interface.query.asyncio.run")
@patch("personal_knowledge_base.interface.query.VectorStore")
@patch("personal_knowledge_base.interface.query.OllamaEmbedder")
def test_query_all_collections_empty(
    mock_embedder_cls: MagicMock,
    mock_vs_cls: MagicMock,
    mock_asyncio_run: MagicMock,
) -> None:
    mock_asyncio_run.return_value = SAMPLE_VECTOR
    mock_store = MagicMock()
    mock_vs_cls.return_value = mock_store
    mock_store.search.return_value = []

    iface = KBQueryInterface()
    results = iface.query_all_collections("test")
    assert results == []


# ---------------------------------------------------------------------------
# format_results()
# ---------------------------------------------------------------------------


def test_format_results_basic() -> None:
    results = [
        QueryResult(
            chunk_text="Interesting content here",
            score=0.9,
            source="docs/file.md",
            doc_id="d1",
            collection="general",
        ),
        QueryResult(
            chunk_text="Another relevant chunk",
            score=0.7,
            source="docs/other.md",
            doc_id="d2",
            collection="ml-ai",
        ),
    ]
    iface = KBQueryInterface()
    formatted = iface.format_results(results)

    assert "Interesting content here" in formatted
    assert "Another relevant chunk" in formatted
    assert "docs/file.md" in formatted
    assert "0.900" in formatted
    assert "general" in formatted
    assert "[1]" in formatted
    assert "[2]" in formatted


def test_format_results_empty() -> None:
    iface = KBQueryInterface()
    assert iface.format_results([]) == "No results found."


def test_format_results_truncates_at_max_chars() -> None:
    long_chunk = "x" * 3000
    results = [
        QueryResult(
            chunk_text=long_chunk,
            score=0.9,
            source="src",
            doc_id="id",
            collection="general",
        )
    ]
    iface = KBQueryInterface()
    formatted = iface.format_results(results, max_chars=500)

    assert len(formatted) <= 503  # allow a little for "..."
    assert formatted.endswith("...")


def test_format_results_no_truncation_when_short() -> None:
    results = [
        QueryResult(
            chunk_text="Short text",
            score=0.8,
            source="file.md",
            doc_id="d1",
            collection="general",
        )
    ]
    iface = KBQueryInterface()
    formatted = iface.format_results(results, max_chars=2000)
    assert "..." not in formatted
    assert "Short text" in formatted


def test_format_results_includes_score() -> None:
    results = [
        QueryResult(
            chunk_text="chunk",
            score=0.12345,
            source="s",
            doc_id="d",
            collection="c",
        )
    ]
    iface = KBQueryInterface()
    formatted = iface.format_results(results)
    assert "0.123" in formatted


def test_format_results_unknown_source() -> None:
    results = [
        QueryResult(
            chunk_text="chunk",
            score=0.5,
            source="",
            doc_id="d",
            collection="general",
        )
    ]
    iface = KBQueryInterface()
    formatted = iface.format_results(results)
    assert "unknown" in formatted


# ---------------------------------------------------------------------------
# KBQueryInterface init
# ---------------------------------------------------------------------------


def test_default_config_when_none() -> None:
    iface = KBQueryInterface()
    assert isinstance(iface.config, QueryConfig)
    assert iface.config.default_collection == "general"


def test_custom_config_stored() -> None:
    cfg = QueryConfig(default_collection="quant-trading", default_limit=3)
    iface = KBQueryInterface(config=cfg)
    assert iface.config.default_collection == "quant-trading"
    assert iface.config.default_limit == 3
