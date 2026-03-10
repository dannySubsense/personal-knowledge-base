"""Tests for the content classifier module (Slice R4-B — topic-based KBs)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from personal_knowledge_base.processing.classifier import (
    DEFAULT_KBS,
    ClassifierConfig,
    ContentClassifier,
    _kb_description,
    _kb_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, idx: int) -> list[float]:
    """Return a unit vector of length ``dim`` with 1.0 at position ``idx``."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def _make_kb_objects(kbs: list[dict[str, str]]) -> list[SimpleNamespace]:
    """Convert a list of dicts to SimpleNamespace objects (duck-typed KB)."""
    return [SimpleNamespace(id=kb["id"], description=kb["description"]) for kb in kbs]


def _make_registry(kbs: list[dict[str, str]] | None = None) -> MagicMock:
    """Return a mock registry whose list_kbs() returns duck-typed KB objects."""
    registry = MagicMock()
    if kbs is None:
        registry.list_kbs.return_value = _make_kb_objects(DEFAULT_KBS)
    else:
        registry.list_kbs.return_value = _make_kb_objects(kbs)
    return registry


def _desc_embed_map(kbs: list[dict[str, str]], n: int) -> dict[str, list[float]]:
    """Return mapping of description text → unit vector for fake_embed."""
    return {kb["description"]: _unit_vec(n, i) for i, kb in enumerate(kbs)}


# ---------------------------------------------------------------------------
# Tests: ClassifierConfig
# ---------------------------------------------------------------------------


class TestClassifierConfig:
    def test_defaults(self) -> None:
        cfg = ClassifierConfig()
        assert cfg.similarity_threshold == 0.3
        assert cfg.ollama_url == "http://localhost:11434"
        assert cfg.embedding_model == "nomic-embed-text"
        assert cfg.db_path == "~/pkb-data/pkb_metadata.db"

    def test_custom_values(self) -> None:
        cfg = ClassifierConfig(
            similarity_threshold=0.5,
            ollama_url="http://myhost:11434",
            embedding_model="all-minilm",
            db_path="/data/my.db",
        )
        assert cfg.similarity_threshold == 0.5
        assert cfg.ollama_url == "http://myhost:11434"
        assert cfg.embedding_model == "all-minilm"
        assert cfg.db_path == "/data/my.db"


# ---------------------------------------------------------------------------
# Tests: DEFAULT_KBS constants
# ---------------------------------------------------------------------------


class TestDefaultKBs:
    def test_non_empty(self) -> None:
        assert len(DEFAULT_KBS) >= 3

    def test_required_ids_present(self) -> None:
        ids = {kb["id"] for kb in DEFAULT_KBS}
        assert "quant-trading" in ids
        assert "ml-ai" in ids
        assert "general" in ids

    def test_no_media_type_collections(self) -> None:
        ids = {kb["id"] for kb in DEFAULT_KBS}
        forbidden = {"videos", "papers", "code"}
        assert ids.isdisjoint(forbidden), f"Found forbidden collection ids: {ids & forbidden}"

    def test_descriptions_non_empty(self) -> None:
        for kb in DEFAULT_KBS:
            assert kb["description"], f"Empty description for KB '{kb['id']}'"


# ---------------------------------------------------------------------------
# Tests: _kb_id / _kb_description helpers
# ---------------------------------------------------------------------------


class TestDuckTypingHelpers:
    def test_kb_id_dict(self) -> None:
        assert _kb_id({"id": "quant-trading", "description": "..."}) == "quant-trading"

    def test_kb_id_object(self) -> None:
        obj = SimpleNamespace(id="ml-ai", description="...")
        assert _kb_id(obj) == "ml-ai"

    def test_kb_description_dict(self) -> None:
        assert _kb_description({"id": "x", "description": "hello"}) == "hello"

    def test_kb_description_object(self) -> None:
        obj = SimpleNamespace(id="x", description="world")
        assert _kb_description(obj) == "world"


# ---------------------------------------------------------------------------
# Tests: ContentClassifier – construction
# ---------------------------------------------------------------------------


class TestContentClassifierConstruction:
    def test_default_construction(self) -> None:
        clf = ContentClassifier()
        assert isinstance(clf.config, ClassifierConfig)
        assert clf._registry is None
        assert clf._desc_embeddings == {}

    def test_custom_config(self) -> None:
        cfg = ClassifierConfig(similarity_threshold=0.5)
        clf = ContentClassifier(config=cfg)
        assert clf.config.similarity_threshold == 0.5

    def test_none_config_uses_default(self) -> None:
        clf = ContentClassifier(config=None)
        assert isinstance(clf.config, ClassifierConfig)
        assert clf.config.similarity_threshold == 0.3

    def test_registry_stored(self) -> None:
        registry = _make_registry()
        clf = ContentClassifier(registry=registry)
        assert clf._registry is registry

    def test_none_registry_stored(self) -> None:
        clf = ContentClassifier(registry=None)
        assert clf._registry is None


# ---------------------------------------------------------------------------
# Tests: ContentClassifier._cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def setup_method(self) -> None:
        self.clf = ContentClassifier()

    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert self.clf._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert self.clf._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert self.clf._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a(self) -> None:
        assert self.clf._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b(self) -> None:
        assert self.clf._cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_both_zero(self) -> None:
        assert self.clf._cosine_similarity([0.0], [0.0]) == 0.0

    def test_partial_similarity(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        score = self.clf._cosine_similarity(a, b)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Tests: ContentClassifier._get_kbs
# ---------------------------------------------------------------------------


class TestGetKBs:
    def test_returns_default_kbs_when_no_registry(self) -> None:
        clf = ContentClassifier()
        kbs = clf._get_kbs()
        ids = [_kb_id(kb) for kb in kbs]
        assert "quant-trading" in ids
        assert "ml-ai" in ids
        assert "general" in ids

    def test_uses_registry_when_provided(self) -> None:
        custom_kbs = [
            {"id": "my-kb", "description": "Custom topic"},
            {"id": "general", "description": "General"},
        ]
        registry = _make_registry(custom_kbs)
        clf = ContentClassifier(registry=registry)
        kbs = clf._get_kbs()
        registry.list_kbs.assert_called_once()
        assert _kb_id(kbs[0]) == "my-kb"

    def test_falls_back_to_default_when_registry_returns_empty(self) -> None:
        registry = MagicMock()
        registry.list_kbs.return_value = []
        clf = ContentClassifier(registry=registry)
        kbs = clf._get_kbs()
        ids = [_kb_id(kb) for kb in kbs]
        assert "quant-trading" in ids

    def test_falls_back_to_default_when_registry_raises(self) -> None:
        registry = MagicMock()
        registry.list_kbs.side_effect = RuntimeError("DB unavailable")
        clf = ContentClassifier(registry=registry)
        kbs = clf._get_kbs()
        ids = [_kb_id(kb) for kb in kbs]
        assert "quant-trading" in ids


# ---------------------------------------------------------------------------
# Tests: ContentClassifier.classify_by_url
# ---------------------------------------------------------------------------


class TestClassifyByUrl:
    def setup_method(self) -> None:
        self.clf = ContentClassifier()

    # Unambiguous quant-finance signals → "quant-trading"
    def test_quant_in_url(self) -> None:
        assert self.clf.classify_by_url("https://quantopian.com/research") == "quant-trading"

    def test_trading_in_url(self) -> None:
        assert self.clf.classify_by_url("https://example.com/trading-strategies") == "quant-trading"

    def test_finance_in_url(self) -> None:
        assert self.clf.classify_by_url("https://myfinance.io/blog/post") == "quant-trading"

    def test_factor_in_url(self) -> None:
        assert self.clf.classify_by_url("https://alpha-factor.com/models") == "quant-trading"

    def test_quant_case_insensitive(self) -> None:
        assert self.clf.classify_by_url("https://QUANTCONNECT.COM/algo") == "quant-trading"

    # Ambiguous → None
    def test_arxiv_returns_none(self) -> None:
        assert self.clf.classify_by_url("https://arxiv.org/abs/2301.00001") is None

    def test_scholar_google_returns_none(self) -> None:
        assert self.clf.classify_by_url("https://scholar.google.com/scholar?q=llm") is None

    def test_youtube_returns_none(self) -> None:
        assert self.clf.classify_by_url("https://www.youtube.com/watch?v=abc") is None

    def test_youtu_be_returns_none(self) -> None:
        assert self.clf.classify_by_url("https://youtu.be/abc123") is None

    def test_github_returns_none(self) -> None:
        assert self.clf.classify_by_url("https://github.com/user/repo") is None

    def test_unknown_url_returns_none(self) -> None:
        assert self.clf.classify_by_url("https://example.com/article") is None

    def test_empty_string_returns_none(self) -> None:
        assert self.clf.classify_by_url("") is None


# ---------------------------------------------------------------------------
# Tests: ContentClassifier.classify_by_content
# ---------------------------------------------------------------------------


class TestClassifyByContent:
    """All tests patch ContentClassifier._embed to avoid Ollama dependency."""

    def _make_clf(self, threshold: float = 0.3) -> ContentClassifier:
        return ContentClassifier(config=ClassifierConfig(similarity_threshold=threshold))

    def _make_embed_fn(
        self, kbs: list[dict[str, str]], winner_id: str
    ) -> tuple[dict[str, list[float]], object]:
        """Build a fake_embed that maps descriptions to orthogonal unit vectors
        and maps any query to the unit vector of ``winner_id``."""
        n = len(kbs) + 1  # extra dim to keep query orthogonal to all if needed
        desc_map = {kb["description"]: _unit_vec(n, i) for i, kb in enumerate(kbs)}
        winner_idx = next(i for i, kb in enumerate(kbs) if kb["id"] == winner_id)

        def fake_embed(text: str) -> list[float]:
            return desc_map.get(text, _unit_vec(n, winner_idx))

        return desc_map, fake_embed  # type: ignore[return-value]

    def test_quant_content_routed_to_quant_trading(self) -> None:
        clf = self._make_clf()
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "quant-trading")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content(
                "Backtesting factor models for equity long-short strategy",
                "Risk management and portfolio construction tutorial",
                kbs=_make_kb_objects(kbs),
            )
        assert result == "quant-trading"

    def test_ml_content_routed_to_ml_ai(self) -> None:
        clf = self._make_clf()
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "ml-ai")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content(
                "Attention is All You Need — transformer architecture overview",
                "Deep learning research paper on LLMs",
                kbs=_make_kb_objects(kbs),
            )
        assert result == "ml-ai"

    def test_ambiguous_content_routed_to_general(self) -> None:
        """When all similarities are below threshold → 'general'."""
        clf = self._make_clf(threshold=0.9)
        kbs = list(DEFAULT_KBS)
        n = len(kbs) + 1

        # Query vector points to unused last dimension → orthogonal to all KBs
        def fake_embed(text: str) -> list[float]:
            for i, kb in enumerate(kbs):
                if text == kb["description"]:
                    return _unit_vec(n, i)
            return _unit_vec(n, n - 1)

        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content(
                "Random blog post about cooking",
                kbs=_make_kb_objects(kbs),
            )
        assert result == "general"

    def test_empty_title_and_description_returns_general(self) -> None:
        clf = self._make_clf()
        with patch.object(clf, "_embed") as mock_embed:
            result = clf.classify_by_content("", "")
        mock_embed.assert_not_called()
        assert result == "general"

    def test_only_title_used_when_description_empty(self) -> None:
        clf = self._make_clf()
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "ml-ai")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content(
                "Neural network architecture search", kbs=_make_kb_objects(kbs)
            )
        assert result == "ml-ai"

    def test_desc_embeddings_cached_across_calls(self) -> None:
        """KB description embeddings should be computed only once."""
        clf = self._make_clf()
        kbs = list(DEFAULT_KBS)
        n = len(kbs) + 1
        call_log: list[str] = []

        def fake_embed(text: str) -> list[float]:
            call_log.append(text)
            for i, kb in enumerate(kbs):
                if text == kb["description"]:
                    return _unit_vec(n, i)
            return _unit_vec(n, 0)

        with patch.object(clf, "_embed", side_effect=fake_embed):
            clf.classify_by_content("first query", kbs=_make_kb_objects(kbs))
            after_first = len(call_log)
            clf.classify_by_content("second query", kbs=_make_kb_objects(kbs))
            after_second = len(call_log)

        # First call: 1 query + N descriptions; second call: 1 query (cache hit)
        n_kbs = len(kbs)
        assert after_first == n_kbs + 1
        assert after_second == n_kbs + 2

    def test_below_threshold_returns_general(self) -> None:
        """Similarity exactly at threshold is NOT enough — must strictly exceed it."""
        clf = self._make_clf(threshold=1.0)  # impossible to beat
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "quant-trading")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content("quant strategy", kbs=_make_kb_objects(kbs))
        assert result == "general"

    def test_uses_default_kbs_when_kbs_arg_is_none(self) -> None:
        """classify_by_content(kbs=None) should call _get_kbs() internally."""
        clf = self._make_clf()
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "quant-trading")

        with (
            patch.object(clf, "_get_kbs", return_value=_make_kb_objects(kbs)) as mock_get,
            patch.object(clf, "_embed", side_effect=fake_embed),
        ):
            result = clf.classify_by_content("factor model research", kbs=None)
        mock_get.assert_called_once()
        assert result == "quant-trading"

    def test_registry_empty_falls_back_to_default_kbs(self) -> None:
        """When registry returns empty list, DEFAULT_KBS are used."""
        registry = MagicMock()
        registry.list_kbs.return_value = []
        clf = ContentClassifier(registry=registry)
        kbs_used: list[Any] = []

        original_get_kbs = clf._get_kbs

        def _capturing_get_kbs() -> list:
            result = original_get_kbs()
            kbs_used.extend(result)
            return result

        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "ml-ai")

        with (
            patch.object(clf, "_get_kbs", side_effect=_capturing_get_kbs),
            patch.object(clf, "_embed", side_effect=fake_embed),
        ):
            result = clf.classify_by_content("deep learning tutorial", kbs=None)

        # Fallback KBs should include quant-trading (from DEFAULT_KBS)
        fallback_ids = [_kb_id(kb) for kb in kbs_used]
        assert "quant-trading" in fallback_ids
        assert result == "ml-ai"


# ---------------------------------------------------------------------------
# Tests: ContentClassifier.classify() — integration
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for the main classify() entry point."""

    def setup_method(self) -> None:
        self.clf = ContentClassifier()

    # --- URL hints ---

    def test_quant_url_hint_used(self) -> None:
        """URL with 'quant' → hint used, _embed not called."""
        registry = _make_registry()
        clf = ContentClassifier(registry=registry)
        with patch.object(clf, "_embed") as mock_embed:
            result = clf.classify("https://quantconnect.com/algo", title="ML paper")
        mock_embed.assert_not_called()
        assert result == "quant-trading"

    def test_trading_url_hint_used(self) -> None:
        registry = _make_registry()
        clf = ContentClassifier(registry=registry)
        with patch.object(clf, "_embed") as mock_embed:
            result = clf.classify("https://example.com/trading-systems")
        mock_embed.assert_not_called()
        assert result == "quant-trading"

    def test_finance_url_hint_used(self) -> None:
        registry = _make_registry()
        clf = ContentClassifier(registry=registry)
        with patch.object(clf, "_embed") as mock_embed:
            result = clf.classify("https://openfinance.io/overview")
        mock_embed.assert_not_called()
        assert result == "quant-trading"

    # --- Ambiguous URLs fall through to content ---

    def test_arxiv_url_falls_through_to_content(self) -> None:
        """arxiv.org → no URL hint → classify_by_content called."""
        kbs = list(DEFAULT_KBS)
        clf = ContentClassifier()
        _, fake_embed = self._make_embed_fn(kbs, "ml-ai")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify(
                "https://arxiv.org/abs/2301.00001",
                title="Transformer architecture for language modelling",
            )
        assert result == "ml-ai"

    def test_github_url_falls_through_to_content(self) -> None:
        kbs = list(DEFAULT_KBS)
        clf = ContentClassifier()
        _, fake_embed = self._make_embed_fn(kbs, "ml-ai")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify(
                "https://github.com/user/repo",
                title="LLM training framework",
            )
        assert result == "ml-ai"

    def test_youtube_url_falls_through_to_content(self) -> None:
        kbs = list(DEFAULT_KBS)
        clf = ContentClassifier()
        _, fake_embed = self._make_embed_fn(kbs, "quant-trading")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify(
                "https://www.youtube.com/watch?v=abc",
                title="Backtesting quant strategies in Python",
            )
        assert result == "quant-trading"

    # --- Empty URL falls to content ---

    def test_empty_url_empty_title_returns_general(self) -> None:
        with patch.object(self.clf, "_embed") as mock_embed:
            result = self.clf.classify("", title="", description="")
        mock_embed.assert_not_called()
        assert result == "general"

    def test_empty_url_with_ml_title(self) -> None:
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "ml-ai")
        with patch.object(self.clf, "_embed", side_effect=fake_embed):
            result = self.clf.classify("", title="Deep neural network for image recognition")
        assert result == "ml-ai"

    def test_empty_url_with_quant_title(self) -> None:
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "quant-trading")
        with patch.object(self.clf, "_embed", side_effect=fake_embed):
            result = self.clf.classify("", title="Factor model performance backtesting")
        assert result == "quant-trading"

    # --- Registry injection ---

    def test_classify_loads_kbs_from_registry(self) -> None:
        registry = _make_registry()
        clf = ContentClassifier(registry=registry)
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "ml-ai")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify("https://example.com/article", title="Machine learning overview")
        registry.list_kbs.assert_called()
        assert result == "ml-ai"

    def test_classify_url_hint_bypasses_embedding(self) -> None:
        """When URL hint matches, classify_by_content must NOT be invoked."""
        registry = _make_registry()
        clf = ContentClassifier(registry=registry)
        with patch.object(clf, "classify_by_content") as mock_content:
            result = clf.classify("https://quant-research.io/paper", title="Deep learning paper")
        mock_content.assert_not_called()
        assert result == "quant-trading"

    # --- No hardcoded media-type collections ---

    def test_no_videos_collection_in_results(self) -> None:
        clf = ContentClassifier()
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "general")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify("https://www.youtube.com/watch?v=abc", title="tutorial video")
        assert result != "videos"

    def test_no_papers_collection_in_results(self) -> None:
        clf = ContentClassifier()
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "general")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify("https://arxiv.org/abs/1234.5678", title="research paper")
        assert result != "papers"

    def test_no_code_collection_in_results(self) -> None:
        clf = ContentClassifier()
        kbs = list(DEFAULT_KBS)
        _, fake_embed = self._make_embed_fn(kbs, "general")
        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify("https://github.com/user/repo", title="library")
        assert result != "code"

    # --- Utility ---

    def _make_embed_fn(
        self, kbs: list[dict[str, str]], winner_id: str
    ) -> tuple[dict[str, list[float]], object]:
        n = len(kbs) + 1
        desc_map = {kb["description"]: _unit_vec(n, i) for i, kb in enumerate(kbs)}
        winner_idx = next(i for i, kb in enumerate(kbs) if kb["id"] == winner_id)

        def fake_embed(text: str) -> list[float]:
            return desc_map.get(text, _unit_vec(n, winner_idx))

        return desc_map, fake_embed  # type: ignore[return-value]


# Make Any available for type hint in test body
from typing import Any  # noqa: E402
