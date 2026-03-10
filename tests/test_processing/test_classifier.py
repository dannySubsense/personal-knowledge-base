"""Tests for the content classifier module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from personal_knowledge_base.processing.classifier import (
    COLLECTION_DESCRIPTIONS,
    URL_RULES,
    ClassifierConfig,
    ContentClassifier,
    _cosine_similarity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, idx: int) -> list[float]:
    """Return a unit vector of length ``dim`` with 1.0 at position ``idx``."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Tests: ClassifierConfig
# ---------------------------------------------------------------------------


class TestClassifierConfig:
    def test_defaults(self) -> None:
        cfg = ClassifierConfig()
        assert cfg.similarity_threshold == 0.3
        assert cfg.ollama_url == "http://localhost:11434"
        assert cfg.embedding_model == "nomic-embed-text"

    def test_custom_values(self) -> None:
        cfg = ClassifierConfig(
            similarity_threshold=0.5,
            ollama_url="http://myhost:11434",
            embedding_model="all-minilm",
        )
        assert cfg.similarity_threshold == 0.5
        assert cfg.ollama_url == "http://myhost:11434"
        assert cfg.embedding_model == "all-minilm"


# ---------------------------------------------------------------------------
# Tests: _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_both_zero(self) -> None:
        assert _cosine_similarity([0.0], [0.0]) == 0.0

    def test_partial_similarity(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        score = _cosine_similarity(a, b)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Tests: ContentClassifier – Tier 1 (URL rules)
# ---------------------------------------------------------------------------


class TestClassifyByUrl:
    def setup_method(self) -> None:
        self.clf = ContentClassifier()

    def test_youtube_standard(self) -> None:
        assert self.clf.classify_by_url("https://www.youtube.com/watch?v=abc") == "videos"

    def test_youtu_be_short(self) -> None:
        assert self.clf.classify_by_url("https://youtu.be/abc123") == "videos"

    def test_arxiv(self) -> None:
        assert self.clf.classify_by_url("https://arxiv.org/abs/2301.00001") == "papers"

    def test_github(self) -> None:
        assert self.clf.classify_by_url("https://github.com/user/repo") == "code"

    def test_pdf_url(self) -> None:
        assert self.clf.classify_by_url("https://example.com/file.pdf") == "papers"

    def test_pdf_url_case_insensitive(self) -> None:
        assert self.clf.classify_by_url("https://example.com/FILE.PDF") == "papers"

    def test_unknown_url_returns_none(self) -> None:
        assert self.clf.classify_by_url("https://example.com/article") is None

    def test_empty_string_returns_none(self) -> None:
        assert self.clf.classify_by_url("") is None

    def test_none_like_empty_returns_none(self) -> None:
        # Passing empty string (closest to None without type violation)
        assert self.clf.classify_by_url("") is None

    def test_youtube_subdomain(self) -> None:
        assert self.clf.classify_by_url("https://m.youtube.com/watch?v=xyz") == "videos"


# ---------------------------------------------------------------------------
# Tests: ContentClassifier – Tier 2 (embedding similarity)
# ---------------------------------------------------------------------------


class TestClassifyByContent:
    """All tests patch ``ContentClassifier._embed`` to avoid Ollama dependency."""

    def _make_clf(self, threshold: float = 0.3) -> ContentClassifier:
        return ContentClassifier(config=ClassifierConfig(similarity_threshold=threshold))

    def test_high_similarity_returns_best_collection(self) -> None:
        """When query embedding is close to 'videos', should return 'videos'."""
        clf = self._make_clf()
        collections = list(COLLECTION_DESCRIPTIONS.keys())

        # Each collection gets a unique orthogonal unit vector.
        videos_idx = collections.index("videos")
        desc_embeddings = {c: _unit_vec(len(collections), i) for i, c in enumerate(collections)}

        def fake_embed(text: str) -> list[float]:
            if text in COLLECTION_DESCRIPTIONS.values():
                # Return the precomputed description vector for that collection
                for c, d in COLLECTION_DESCRIPTIONS.items():
                    if text == d:
                        return desc_embeddings[c]
            # Query: same as videos vector → similarity 1.0
            return _unit_vec(len(collections), videos_idx)

        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content("Some video tutorial", "watch on YouTube")
        assert result == "videos"

    def test_low_similarity_returns_general(self) -> None:
        """When all similarities are below threshold, return 'general'."""
        clf = self._make_clf(threshold=0.9)

        # All embeddings are orthogonal → similarity = 0.0 for query vs. any desc
        collections = list(COLLECTION_DESCRIPTIONS.keys())
        n = len(collections) + 1  # extra dim so query is orthogonal to everything

        def fake_embed(text: str) -> list[float]:
            for i, (_c, d) in enumerate(COLLECTION_DESCRIPTIONS.items()):
                if text == d:
                    return _unit_vec(n, i)
            # query vector points in the unused last dimension
            return _unit_vec(n, n - 1)

        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content("random text that matches nothing")
        assert result == "general"

    def test_empty_title_and_description_returns_general(self) -> None:
        clf = self._make_clf()
        # _embed should never be called for empty text
        with patch.object(clf, "_embed") as mock_embed:
            result = clf.classify_by_content("", "")
        mock_embed.assert_not_called()
        assert result == "general"

    def test_only_title_used_when_description_empty(self) -> None:
        clf = self._make_clf()
        collections = list(COLLECTION_DESCRIPTIONS.keys())
        papers_idx = collections.index("papers")
        desc_embeddings = {c: _unit_vec(len(collections), i) for i, c in enumerate(collections)}

        def fake_embed(text: str) -> list[float]:
            for c, d in COLLECTION_DESCRIPTIONS.items():
                if text == d:
                    return desc_embeddings[c]
            return _unit_vec(len(collections), papers_idx)

        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content("arxiv research paper on LLMs")
        assert result == "papers"

    def test_desc_embeddings_cached(self) -> None:
        """_embed should only be called once per description across two classify calls."""
        clf = self._make_clf()
        collections = list(COLLECTION_DESCRIPTIONS.keys())
        n = len(collections)

        call_count: dict[str, int] = {"n": 0}

        def fake_embed(text: str) -> list[float]:
            call_count["n"] += 1
            for i, (_c, d) in enumerate(COLLECTION_DESCRIPTIONS.items()):
                if text == d:
                    return _unit_vec(n, i)
            return _unit_vec(n, 0)

        with patch.object(clf, "_embed", side_effect=fake_embed):
            clf.classify_by_content("first query")
            first_count = call_count["n"]
            clf.classify_by_content("second query")
            second_count = call_count["n"]

        # Descriptions only embedded once; second call adds 1 for query only
        assert second_count == first_count + 1

    def test_code_collection(self) -> None:
        clf = self._make_clf()
        collections = list(COLLECTION_DESCRIPTIONS.keys())
        code_idx = collections.index("code")
        desc_embeddings = {c: _unit_vec(len(collections), i) for i, c in enumerate(collections)}

        def fake_embed(text: str) -> list[float]:
            for c, d in COLLECTION_DESCRIPTIONS.items():
                if text == d:
                    return desc_embeddings[c]
            return _unit_vec(len(collections), code_idx)

        with patch.object(clf, "_embed", side_effect=fake_embed):
            result = clf.classify_by_content("Python library for data analysis")
        assert result == "code"


# ---------------------------------------------------------------------------
# Tests: ContentClassifier – classify() (integration of tiers)
# ---------------------------------------------------------------------------


class TestClassify:
    def setup_method(self) -> None:
        self.clf = ContentClassifier()

    def test_youtube_url_wins_over_content(self) -> None:
        """URL rule should short-circuit; _embed should NOT be called."""
        with patch.object(self.clf, "_embed") as mock_embed:
            result = self.clf.classify("https://youtube.com/watch?v=1", title="arxiv paper")
        mock_embed.assert_not_called()
        assert result == "videos"

    def test_arxiv_url(self) -> None:
        result = self.clf.classify("https://arxiv.org/abs/2301.00001")
        assert result == "papers"

    def test_github_url(self) -> None:
        result = self.clf.classify("https://github.com/owner/repo")
        assert result == "code"

    def test_pdf_url(self) -> None:
        result = self.clf.classify("https://cdn.example.com/paper.pdf")
        assert result == "papers"

    def test_unknown_url_falls_back_to_content(self) -> None:
        """Unknown URL → Tier-2 should be invoked (mocked here)."""
        collections = list(COLLECTION_DESCRIPTIONS.keys())
        papers_idx = collections.index("papers")
        desc_embeddings = {c: _unit_vec(len(collections), i) for i, c in enumerate(collections)}

        def fake_embed(text: str) -> list[float]:
            for c, d in COLLECTION_DESCRIPTIONS.items():
                if text == d:
                    return desc_embeddings[c]
            return _unit_vec(len(collections), papers_idx)

        with patch.object(self.clf, "_embed", side_effect=fake_embed):
            result = self.clf.classify(
                "https://example.com/research",
                title="Deep learning survey paper",
            )
        assert result == "papers"

    def test_empty_url_falls_back_to_content_then_general(self) -> None:
        """Empty URL + empty title → 'general' without calling _embed."""
        with patch.object(self.clf, "_embed") as mock_embed:
            result = self.clf.classify("", title="", description="")
        mock_embed.assert_not_called()
        assert result == "general"

    def test_no_url_with_title_uses_content(self) -> None:
        collections = list(COLLECTION_DESCRIPTIONS.keys())
        videos_idx = collections.index("videos")
        desc_embeddings = {c: _unit_vec(len(collections), i) for i, c in enumerate(collections)}

        def fake_embed(text: str) -> list[float]:
            for c, d in COLLECTION_DESCRIPTIONS.items():
                if text == d:
                    return desc_embeddings[c]
            return _unit_vec(len(collections), videos_idx)

        with patch.object(self.clf, "_embed", side_effect=fake_embed):
            result = self.clf.classify("", title="lecture video recording")
        assert result == "videos"

    def test_default_title_and_description_empty(self) -> None:
        """classify() with no title/description should still work, defaulting to 'general'."""
        result_url_match = self.clf.classify("https://github.com/x/y")
        assert result_url_match == "code"

    def test_classify_uses_url_rule_first(self) -> None:
        """Explicitly verify Tier-1 priority when URL + misleading title."""
        with patch.object(self.clf, "classify_by_content") as mock_content:
            result = self.clf.classify("https://arxiv.org/abs/1234.5678", title="GitHub project")
        mock_content.assert_not_called()
        assert result == "papers"


# ---------------------------------------------------------------------------
# Tests: ContentClassifier – construction edge cases
# ---------------------------------------------------------------------------


class TestContentClassifierConstruction:
    def test_default_construction(self) -> None:
        clf = ContentClassifier()
        assert isinstance(clf.config, ClassifierConfig)

    def test_custom_config(self) -> None:
        cfg = ClassifierConfig(similarity_threshold=0.5)
        clf = ContentClassifier(config=cfg)
        assert clf.config.similarity_threshold == 0.5

    def test_none_config_uses_default(self) -> None:
        clf = ContentClassifier(config=None)  # type: ignore[arg-type]
        assert clf.config.similarity_threshold == 0.3

    def test_desc_embeddings_start_empty(self) -> None:
        clf = ContentClassifier()
        assert clf._desc_embeddings == {}


# ---------------------------------------------------------------------------
# Tests: module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_url_rules_non_empty(self) -> None:
        assert len(URL_RULES) >= 4

    def test_collection_descriptions_keys(self) -> None:
        assert set(COLLECTION_DESCRIPTIONS.keys()) == {"videos", "papers", "code", "general"}

    def test_collection_descriptions_non_empty_values(self) -> None:
        for key, value in COLLECTION_DESCRIPTIONS.items():
            assert value, f"Description for '{key}' is empty"
