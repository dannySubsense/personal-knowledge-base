"""Tests for the staleness detection module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from personal_knowledge_base.processing.staleness import (
    StalenessConfig,
    StalenessDetector,
    StalenessResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(UTC)


def _days_ago(days: int) -> datetime:
    return _NOW - timedelta(days=days)


def _years_ago(years: int) -> datetime:
    return _NOW - timedelta(days=365 * years)


# ---------------------------------------------------------------------------
# StalenessConfig
# ---------------------------------------------------------------------------


class TestStalenessConfig:
    def test_defaults(self) -> None:
        cfg = StalenessConfig()
        assert cfg.stale_threshold == 0.7
        assert cfg.age_weight == 0.4
        assert cfg.topic_weight == 0.4
        assert cfg.source_weight == 0.2

    def test_custom(self) -> None:
        cfg = StalenessConfig(
            stale_threshold=0.5, age_weight=0.3, topic_weight=0.5, source_weight=0.2
        )
        assert cfg.stale_threshold == 0.5
        assert cfg.age_weight == 0.3
        assert cfg.topic_weight == 0.5
        assert cfg.source_weight == 0.2


# ---------------------------------------------------------------------------
# StalenessResult
# ---------------------------------------------------------------------------


class TestStalenessResult:
    def test_fields_present(self) -> None:
        r = StalenessResult(
            score=0.5,
            is_stale=False,
            age_score=0.4,
            topic_score=0.6,
            source_score=0.5,
            reason="test reason",
        )
        assert r.score == 0.5
        assert not r.is_stale
        assert r.reason == "test reason"


# ---------------------------------------------------------------------------
# Age scoring
# ---------------------------------------------------------------------------


class TestAgeScore:
    def setup_method(self) -> None:
        self.detector = StalenessDetector()

    def test_less_than_30_days(self) -> None:
        assert self.detector._age_score(_days_ago(10)) == 0.0

    def test_exactly_0_days(self) -> None:
        assert self.detector._age_score(_NOW) == 0.0

    def test_30_to_90_days(self) -> None:
        assert self.detector._age_score(_days_ago(45)) == 0.2

    def test_90_to_180_days(self) -> None:
        assert self.detector._age_score(_days_ago(120)) == 0.4

    def test_180_to_365_days(self) -> None:
        assert self.detector._age_score(_days_ago(270)) == 0.6

    def test_1_to_2_years(self) -> None:
        assert self.detector._age_score(_days_ago(500)) == 0.8

    def test_more_than_2_years(self) -> None:
        assert self.detector._age_score(_days_ago(800)) == 1.0

    def test_boundary_exactly_30_days(self) -> None:
        assert self.detector._age_score(_days_ago(30)) == 0.2

    def test_boundary_exactly_90_days(self) -> None:
        assert self.detector._age_score(_days_ago(90)) == 0.4

    def test_boundary_exactly_180_days(self) -> None:
        assert self.detector._age_score(_days_ago(180)) == 0.6

    def test_boundary_exactly_365_days(self) -> None:
        assert self.detector._age_score(_days_ago(365)) == 0.8

    def test_boundary_exactly_730_days(self) -> None:
        assert self.detector._age_score(_days_ago(730)) == 1.0

    def test_naive_datetime_treated_as_utc(self) -> None:
        naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=800)
        score = self.detector._age_score(naive)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Topic scoring
# ---------------------------------------------------------------------------


class TestTopicScore:
    def setup_method(self) -> None:
        self.detector = StalenessDetector()

    # High volatility
    def test_ai_keywords(self) -> None:
        assert self.detector._topic_score(["ai", "neural"]) == 0.9

    def test_llm_keywords(self) -> None:
        assert self.detector._topic_score(["llm", "gpt", "transformer"]) == 0.9

    def test_crypto_keywords(self) -> None:
        assert self.detector._topic_score(["crypto", "blockchain", "nft", "defi"]) == 0.9

    def test_model_benchmark(self) -> None:
        assert self.detector._topic_score(["model", "benchmark"]) == 0.9

    def test_framework_library(self) -> None:
        assert self.detector._topic_score(["framework", "library"]) == 0.7

    def test_api_sdk_version(self) -> None:
        assert self.detector._topic_score(["api", "sdk", "version"]) == 0.7

    def test_tutorial_guide(self) -> None:
        assert self.detector._topic_score(["tutorial", "guide", "howto"]) == 0.5

    # Low volatility
    def test_philosophy(self) -> None:
        assert self.detector._topic_score(["philosophy"]) == 0.1

    def test_history(self) -> None:
        assert self.detector._topic_score(["history"]) == 0.1

    def test_mathematics_statistics(self) -> None:
        assert self.detector._topic_score(["mathematics", "statistics"]) == 0.1

    def test_psychology_biology_physics(self) -> None:
        assert self.detector._topic_score(["psychology", "biology", "physics"]) == 0.1

    def test_classic_fundamentals_principles(self) -> None:
        assert self.detector._topic_score(["classic", "fundamentals", "principles"]) == 0.1

    # Mixed — should return max
    def test_mixed_returns_max(self) -> None:
        score = self.detector._topic_score(["philosophy", "ai"])
        assert score == 0.9

    # Default
    def test_empty_keywords_default(self) -> None:
        assert self.detector._topic_score([]) == 0.5

    def test_unknown_keywords_default(self) -> None:
        assert self.detector._topic_score(["randomword", "anotherthing"]) == 0.5

    def test_case_insensitive(self) -> None:
        assert self.detector._topic_score(["AI", "LLM"]) == 0.9


# ---------------------------------------------------------------------------
# Source scoring
# ---------------------------------------------------------------------------


class TestSourceScore:
    def setup_method(self) -> None:
        self.detector = StalenessDetector()

    def test_arxiv(self) -> None:
        assert self.detector._source_score("https://arxiv.org/abs/2101.00001") == 0.2

    def test_github(self) -> None:
        assert self.detector._source_score("https://github.com/user/repo") == 0.5

    def test_medium(self) -> None:
        assert self.detector._source_score("https://medium.com/@user/article") == 0.6

    def test_substack(self) -> None:
        assert self.detector._source_score("https://user.substack.com/p/post") == 0.6

    def test_blog_url(self) -> None:
        assert self.detector._source_score("https://example.com/blog/post") == 0.6

    def test_news_subdomain(self) -> None:
        assert self.detector._source_score("https://news.ycombinator.com/item") == 0.9

    def test_dotnews_tld(self) -> None:
        assert self.detector._source_score("https://example.news/story") == 0.9

    def test_reuters(self) -> None:
        assert self.detector._source_score("https://reuters.com/article") == 0.9

    def test_bbc(self) -> None:
        assert self.detector._source_score("https://bbc.com/news/article") == 0.9

    def test_cnn(self) -> None:
        assert self.detector._source_score("https://cnn.com/article") == 0.9

    def test_youtube(self) -> None:
        assert self.detector._source_score("https://youtube.com/watch?v=abc") == 0.4

    def test_youtu_be(self) -> None:
        assert self.detector._source_score("https://youtu.be/abc") == 0.4

    def test_default_unknown_source(self) -> None:
        assert self.detector._source_score("https://example.com/article") == 0.5

    def test_empty_url(self) -> None:
        assert self.detector._source_score("") == 0.5


# ---------------------------------------------------------------------------
# score() integration tests
# ---------------------------------------------------------------------------


class TestScore:
    def setup_method(self) -> None:
        self.detector = StalenessDetector()

    def test_old_ai_paper_arxiv_is_stale(self) -> None:
        """Old AI paper (>2 years, AI keywords, arxiv) → high score, stale."""
        result = self.detector.score(
            content_date=_years_ago(3),
            topic_keywords=["ai", "neural", "benchmark"],
            source_url="https://arxiv.org/abs/2101.00001",
        )
        # age=1.0, topic=0.9, source=0.2 → 0.4*1.0 + 0.4*0.9 + 0.2*0.2 = 0.4+0.36+0.04 = 0.80
        assert result.is_stale is True
        assert result.score >= 0.7
        assert result.age_score == 1.0
        assert result.topic_score == 0.9
        assert result.source_score == 0.2

    def test_old_philosophy_article_not_stale(self) -> None:
        """Old philosophy article (>2 years, philosophy keywords) → low score, not stale."""
        result = self.detector.score(
            content_date=_years_ago(3),
            topic_keywords=["philosophy", "history"],
            source_url="https://example.com/philosophy",
        )
        # age=1.0, topic=0.1, source=0.5 → 0.4*1.0 + 0.4*0.1 + 0.2*0.5 = 0.4+0.04+0.10 = 0.54
        assert result.is_stale is False
        assert result.score < 0.7
        assert result.topic_score == 0.1

    def test_recent_content_low_age_score(self) -> None:
        """Recent content (< 30 days) → age_score = 0.0."""
        result = self.detector.score(
            content_date=_days_ago(5),
            topic_keywords=["ai"],
            source_url="",
        )
        assert result.age_score == 0.0

    def test_score_at_exactly_threshold_is_stale(self) -> None:
        """Score == threshold → is_stale=True."""
        # Score exactly at threshold: use threshold=0.8 and age_weight=1.0
        # so combined = 1.0 * age_score; age_score for 1-2 years = 0.8 == threshold
        cfg2 = StalenessConfig(
            stale_threshold=0.8, age_weight=1.0, topic_weight=0.0, source_weight=0.0
        )
        detector2 = StalenessDetector(config=cfg2)
        result = detector2.score(
            content_date=_days_ago(500),  # 1-2 years → age_score=0.8
            topic_keywords=[],
            source_url="",
        )
        assert result.score == pytest.approx(0.8)
        assert result.is_stale is True

    def test_score_below_threshold_not_stale(self) -> None:
        """Score below threshold → is_stale=False."""
        cfg = StalenessConfig(stale_threshold=0.9)
        detector = StalenessDetector(config=cfg)
        result = detector.score(
            content_date=_days_ago(5),
            topic_keywords=["philosophy"],
            source_url="https://arxiv.org/abs/123",
        )
        assert result.is_stale is False
        assert result.score < 0.9

    def test_reason_is_nonempty(self) -> None:
        result = self.detector.score(
            content_date=_days_ago(100),
            topic_keywords=["ai"],
            source_url="https://medium.com/article",
        )
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_reason_contains_meaningful_content(self) -> None:
        result = self.detector.score(
            content_date=_days_ago(800),
            topic_keywords=["philosophy"],
            source_url="https://arxiv.org/abs/1234",
        )
        assert "stale" in result.reason or "fresh" in result.reason

    def test_no_url_uses_default_source_score(self) -> None:
        result = self.detector.score(
            content_date=_days_ago(10),
            topic_keywords=["ai"],
            source_url="",
        )
        assert result.source_score == 0.5

    def test_custom_config_threshold(self) -> None:
        cfg = StalenessConfig(stale_threshold=0.3)
        detector = StalenessDetector(config=cfg)
        # Even recent content with default topic/source should exceed 0.3
        result = detector.score(
            content_date=_days_ago(45),  # age=0.2
            topic_keywords=["api"],  # topic=0.7
            source_url="https://medium.com/post",  # source=0.6
        )
        # 0.4*0.2 + 0.4*0.7 + 0.2*0.6 = 0.08+0.28+0.12 = 0.48 > 0.3
        assert result.is_stale is True

    def test_score_clamped_to_range(self) -> None:
        result = self.detector.score(
            content_date=_years_ago(5),
            topic_keywords=["crypto", "nft", "defi"],
            source_url="https://reuters.com/story",
        )
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# batch_score()
# ---------------------------------------------------------------------------


class TestBatchScore:
    def setup_method(self) -> None:
        self.detector = StalenessDetector()

    def test_batch_returns_correct_count(self) -> None:
        items = [
            {"date": _days_ago(10), "keywords": ["ai"], "url": "https://arxiv.org/abs/1"},
            {"date": _days_ago(400), "keywords": ["philosophy"], "url": ""},
            {"date": _days_ago(800), "keywords": ["crypto"], "url": "https://reuters.com/x"},
        ]
        results = self.detector.batch_score(items)
        assert len(results) == 3

    def test_batch_empty_list(self) -> None:
        assert self.detector.batch_score([]) == []

    def test_batch_preserves_order(self) -> None:
        items = [
            {"date": _years_ago(3), "keywords": ["ai"], "url": "https://arxiv.org/abs/1"},
            {"date": _days_ago(5), "keywords": ["philosophy"], "url": ""},
        ]
        results = self.detector.batch_score(items)
        # First item is old AI → stale; second is recent philosophy → fresh
        assert results[0].is_stale is True
        assert results[1].is_stale is False

    def test_batch_url_optional(self) -> None:
        """Items without 'url' key should not raise."""
        items = [
            {"date": _days_ago(10), "keywords": ["ai"]},
        ]
        results = self.detector.batch_score(items)
        assert len(results) == 1
        assert results[0].source_score == 0.5

    def test_batch_single_item(self) -> None:
        items = [{"date": _years_ago(3), "keywords": ["ai"], "url": "https://arxiv.org/abs/1"}]
        results = self.detector.batch_score(items)
        assert len(results) == 1
        assert results[0].is_stale is True


# ---------------------------------------------------------------------------
# Default config (no config passed)
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_none_config_uses_defaults(self) -> None:
        detector = StalenessDetector(config=None)
        assert detector.config.stale_threshold == 0.7
        assert detector.config.age_weight == 0.4

    def test_no_config_arg_uses_defaults(self) -> None:
        detector = StalenessDetector()
        assert detector.config.stale_threshold == 0.7
