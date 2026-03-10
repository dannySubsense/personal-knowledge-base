"""Tests for personal_knowledge_base.processing.clustering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from personal_knowledge_base.processing.clustering import (
    _STOP_WORDS,
    Cluster,
    ClusteringConfig,
    ClusteringResult,
    KBClusterer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vectors(n: int, dims: int = 8) -> list[list[float]]:
    """Return *n* random unit vectors with *dims* dimensions."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dims)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    return vecs.tolist()


def _make_payloads(
    n: int,
    texts: list[str] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    payloads = []
    for i in range(n):
        text = texts[i] if texts else f"document {i} about topic {i % 3}"
        src = sources[i] if sources else f"https://example.com/doc{i}"
        payloads.append({"chunk_text": text, "source": src})
    return payloads


def _clustered_labels(n: int, n_clusters: int = 3) -> list[int]:
    """Assign each of *n* docs to one of *n_clusters* clusters round-robin."""
    return [i % n_clusters for i in range(n)]


# ---------------------------------------------------------------------------
# ClusteringConfig
# ---------------------------------------------------------------------------


class TestClusteringConfig:
    def test_defaults(self) -> None:
        cfg = ClusteringConfig()
        assert cfg.min_docs_to_cluster == 20
        assert cfg.min_cluster_size == 3
        assert cfg.min_samples == 2
        assert cfg.qdrant_url == "http://localhost:6333"

    def test_custom(self) -> None:
        cfg = ClusteringConfig(min_docs_to_cluster=10, min_cluster_size=2)
        assert cfg.min_docs_to_cluster == 10
        assert cfg.min_cluster_size == 2


# ---------------------------------------------------------------------------
# KBClusterer — initialisation
# ---------------------------------------------------------------------------


class TestKBClustererInit:
    def test_default_config(self) -> None:
        c = KBClusterer()
        assert isinstance(c.config, ClusteringConfig)

    def test_custom_config(self) -> None:
        cfg = ClusteringConfig(min_docs_to_cluster=5)
        c = KBClusterer(config=cfg)
        assert c.config.min_docs_to_cluster == 5


# ---------------------------------------------------------------------------
# cluster_collection — fewer than min_docs
# ---------------------------------------------------------------------------


class TestClusterCollectionNotEnoughDocs:
    def test_empty(self) -> None:
        c = KBClusterer()
        result = c.cluster_collection("test", [], [])
        assert result.total_docs == 0
        assert result.clusters == []
        assert "Not enough" in result.summary

    def test_just_below_threshold(self) -> None:
        c = KBClusterer(ClusteringConfig(min_docs_to_cluster=20))
        vectors = _make_vectors(19)
        payloads = _make_payloads(19)
        result = c.cluster_collection("mycol", vectors, payloads)
        assert result.total_docs == 19
        assert result.clusters == []
        assert "Not enough" in result.summary

    def test_noise_count_equals_total_when_not_enough(self) -> None:
        c = KBClusterer()
        vectors = _make_vectors(5)
        payloads = _make_payloads(5)
        result = c.cluster_collection("col", vectors, payloads)
        assert result.noise_count == 5

    def test_collection_name_preserved(self) -> None:
        c = KBClusterer()
        result = c.cluster_collection("my-collection", [], [])
        assert result.collection == "my-collection"


# ---------------------------------------------------------------------------
# cluster_collection — with enough docs (mock _run_clustering)
# ---------------------------------------------------------------------------


class TestClusterCollectionWithClusters:
    def _make_clusterer_with_labels(self, labels: list[int]) -> KBClusterer:
        c = KBClusterer(ClusteringConfig(min_docs_to_cluster=20))
        c._run_clustering = MagicMock(return_value=labels)  # type: ignore[method-assign]
        return c

    def test_exactly_at_threshold(self) -> None:
        labels = _clustered_labels(20, n_clusters=2)
        c = self._make_clusterer_with_labels(labels)
        result = c.cluster_collection("col", _make_vectors(20), _make_payloads(20))
        assert result.total_docs == 20
        assert len(result.clusters) == 2

    def test_cluster_sizes_correct(self) -> None:
        # 24 docs, 3 clusters → 8 each (round-robin)
        labels = _clustered_labels(24, n_clusters=3)
        c = self._make_clusterer_with_labels(labels)
        result = c.cluster_collection("col", _make_vectors(24), _make_payloads(24))
        sizes = sorted(cl.size for cl in result.clusters)
        assert sizes == [8, 8, 8]

    def test_noise_count(self) -> None:
        # first 3 docs are noise (-1), rest go to cluster 0
        labels = [-1, -1, -1] + [0] * 20
        c = self._make_clusterer_with_labels(labels)
        vectors = _make_vectors(23)
        payloads = _make_payloads(23)
        result = c.cluster_collection("col", vectors, payloads)
        assert result.noise_count == 3
        assert len(result.clusters) == 1

    def test_all_noise(self) -> None:
        labels = [-1] * 20
        c = self._make_clusterer_with_labels(labels)
        result = c.cluster_collection("col", _make_vectors(20), _make_payloads(20))
        assert result.noise_count == 20
        assert result.clusters == []

    def test_sample_sources_capped_at_three(self) -> None:
        # all 20 docs go to same cluster, each with a unique source
        labels = [0] * 20
        c = self._make_clusterer_with_labels(labels)
        result = c.cluster_collection("col", _make_vectors(20), _make_payloads(20))
        assert len(result.clusters[0].sample_sources) <= 3

    def test_summary_is_meaningful(self) -> None:
        labels = _clustered_labels(20, n_clusters=2)
        c = self._make_clusterer_with_labels(labels)
        result = c.cluster_collection("col", _make_vectors(20), _make_payloads(20))
        assert "Found" in result.summary or len(result.summary) > 5

    def test_cluster_ids_match_labels(self) -> None:
        labels = _clustered_labels(21, n_clusters=3)
        c = self._make_clusterer_with_labels(labels)
        result = c.cluster_collection("col", _make_vectors(21), _make_payloads(21))
        cluster_ids = {cl.id for cl in result.clusters}
        assert cluster_ids == {0, 1, 2}

    def test_top_terms_populated(self) -> None:
        labels = [0] * 25
        texts = ["neural network transformer attention" for _ in range(25)]
        c = self._make_clusterer_with_labels(labels)
        payloads = _make_payloads(25, texts=texts)
        result = c.cluster_collection("col", _make_vectors(25), payloads)
        assert len(result.clusters[0].top_terms) > 0


# ---------------------------------------------------------------------------
# _extract_top_terms
# ---------------------------------------------------------------------------


class TestExtractTopTerms:
    def setup_method(self) -> None:
        self.c = KBClusterer()

    def test_basic_extraction(self) -> None:
        texts = ["neural network attention transformer model"] * 3
        terms = self.c._extract_top_terms(texts, n=3)
        assert len(terms) <= 3
        assert all(isinstance(t, str) for t in terms)

    def test_stop_words_removed(self) -> None:
        texts = ["the a an is are was it its this that for with by"]
        terms = self.c._extract_top_terms(texts, n=10)
        for stop in _STOP_WORDS:
            assert stop not in terms

    def test_short_tokens_removed(self) -> None:
        texts = ["ai ml dl go to the for at in on"]
        # "ai", "ml", "dl", "go" are ≤2 chars
        terms = self.c._extract_top_terms(texts, n=10)
        for t in terms:
            assert len(t) > 2

    def test_most_frequent_first(self) -> None:
        # "python" appears 5x, "java" appears 2x
        texts = ["python python python python python java java other"]
        terms = self.c._extract_top_terms(texts, n=3)
        assert terms[0] == "python"
        assert "java" in terms

    def test_returns_at_most_n(self) -> None:
        texts = ["word1 word2 word3 word4 word5 word6 word7"]
        terms = self.c._extract_top_terms(texts, n=5)
        assert len(terms) <= 5

    def test_empty_texts(self) -> None:
        terms = self.c._extract_top_terms([], n=5)
        assert terms == []

    def test_case_insensitive(self) -> None:
        texts = ["Python PYTHON python"]
        terms = self.c._extract_top_terms(texts, n=1)
        assert terms == ["python"]


# ---------------------------------------------------------------------------
# _generate_label
# ---------------------------------------------------------------------------


class TestGenerateLabel:
    def setup_method(self) -> None:
        self.c = KBClusterer()

    def test_basic_label(self) -> None:
        label = self.c._generate_label(["neural", "transformers", "attention"])
        assert label == "Neural Transformers"

    def test_single_term(self) -> None:
        label = self.c._generate_label(["machine"])
        assert label == "Machine"

    def test_empty_terms(self) -> None:
        label = self.c._generate_label([])
        assert label == "Unlabelled"

    def test_capitalized(self) -> None:
        label = self.c._generate_label(["deep", "learning"])
        assert label[0].isupper()

    def test_uses_only_first_two(self) -> None:
        label = self.c._generate_label(["alpha", "beta", "gamma", "delta"])
        assert "Gamma" not in label
        assert "Delta" not in label


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    def setup_method(self) -> None:
        self.c = KBClusterer()

    def _make_result(self, n_clusters: int = 2) -> ClusteringResult:
        clusters = [
            Cluster(
                id=i,
                label=f"Label {i}",
                size=10,
                top_terms=["term1", "term2"],
                sample_sources=[f"https://example.com/{i}"],
            )
            for i in range(n_clusters)
        ]
        return ClusteringResult(
            collection="test-col",
            total_docs=20,
            clusters=clusters,
            noise_count=0,
            summary="Found 2 sub-categories across 20 documents (0 uncategorised).",
        )

    def test_contains_collection_name(self) -> None:
        result = self._make_result()
        text = self.c.format_results(result)
        assert "test-col" in text

    def test_contains_total_docs(self) -> None:
        result = self._make_result()
        text = self.c.format_results(result)
        assert "20" in text

    def test_contains_cluster_labels(self) -> None:
        result = self._make_result()
        text = self.c.format_results(result)
        assert "Label 0" in text
        assert "Label 1" in text

    def test_contains_top_terms(self) -> None:
        result = self._make_result()
        text = self.c.format_results(result)
        assert "term1" in text

    def test_contains_sources(self) -> None:
        result = self._make_result()
        text = self.c.format_results(result)
        assert "https://example.com/0" in text

    def test_empty_clusters_format(self) -> None:
        result = ClusteringResult(
            collection="empty",
            total_docs=5,
            clusters=[],
            noise_count=5,
            summary="Not enough documents to cluster.",
        )
        text = self.c.format_results(result)
        assert "empty" in text
        assert "5" in text
        assert "Not enough" in text

    def test_returns_string(self) -> None:
        result = self._make_result()
        assert isinstance(self.c.format_results(result), str)


# ---------------------------------------------------------------------------
# _run_clustering — both paths tested via mocks
# ---------------------------------------------------------------------------


class TestRunClusteringFallback:
    """Test both the HDBSCAN and DBSCAN fallback paths using mocks."""

    def test_hdbscan_path_via_mock(self) -> None:
        """HDBSCAN path: mock the hdbscan module so we exercise lines 272-277."""
        import personal_knowledge_base.processing.clustering as mod

        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.labels_ = np.array([0, 0, 1, 1, 2, 2])
        mock_hdbscan_cls = MagicMock(return_value=mock_hdbscan_instance)
        mock_hdbscan_module = MagicMock()
        mock_hdbscan_module.HDBSCAN = mock_hdbscan_cls

        original_flag = mod.HAS_HDBSCAN
        original_module = mod.__dict__.get("hdbscan")
        try:
            mod.HAS_HDBSCAN = True
            mod.hdbscan = mock_hdbscan_module  # type: ignore[attr-defined]
            c = KBClusterer()
            arr = np.zeros((6, 4), dtype=np.float32)
            labels = c._run_clustering(arr)
            assert labels == [0, 0, 1, 1, 2, 2]
            mock_hdbscan_cls.assert_called_once()
        finally:
            mod.HAS_HDBSCAN = original_flag
            if original_module is not None:
                mod.hdbscan = original_module  # type: ignore[attr-defined]
            else:
                mod.__dict__.pop("hdbscan", None)

    def test_dbscan_fallback_via_mock(self) -> None:
        """DBSCAN fallback: mock sklearn so we exercise lines 279-286."""
        import personal_knowledge_base.processing.clustering as mod

        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.labels_ = np.array([0, 0, 1, 1, -1, -1])
        mock_dbscan_cls = MagicMock(return_value=mock_dbscan_instance)
        mock_sklearn_cluster = MagicMock()
        mock_sklearn_cluster.DBSCAN = mock_dbscan_cls

        original_flag = mod.HAS_HDBSCAN
        try:
            mod.HAS_HDBSCAN = False
            with patch.dict(
                "sys.modules",
                {
                    "sklearn": MagicMock(),
                    "sklearn.cluster": mock_sklearn_cluster,
                },
            ):
                c = KBClusterer()
                arr = np.zeros((6, 4), dtype=np.float32)
                labels = c._run_clustering(arr)
            assert labels == [0, 0, 1, 1, -1, -1]
            mock_dbscan_cls.assert_called_once()
        finally:
            mod.HAS_HDBSCAN = original_flag

    def test_hdbscan_real_if_available(self) -> None:
        import personal_knowledge_base.processing.clustering as mod

        if not mod.HAS_HDBSCAN:
            pytest.skip("hdbscan not installed — tested via mock above")

        c = KBClusterer()
        rng = np.random.default_rng(1)
        centres = np.array([[0.0, 0.0], [10.0, 0.0]])
        parts = [centres[i] + rng.standard_normal((15, 2)) * 0.05 for i in range(2)]
        arr = np.vstack(parts).astype(np.float32)
        labels = c._run_clustering(arr)
        assert len(labels) == 30


# ---------------------------------------------------------------------------
# Dataclass field integrity
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_cluster_fields(self) -> None:
        cl = Cluster(
            id=0,
            label="Test",
            size=5,
            top_terms=["a", "b"],
            sample_sources=["http://x.com"],
        )
        assert cl.id == 0
        assert cl.label == "Test"
        assert cl.size == 5

    def test_clustering_result_default_clusters(self) -> None:
        r = ClusteringResult(collection="c", total_docs=0)
        assert r.clusters == []
        assert r.noise_count == 0
        assert r.summary == ""
