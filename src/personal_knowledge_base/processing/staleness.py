"""Staleness detection for KB content.

Scores content for staleness based on three factors:
- Age of the content
- Volatility of the topic (how quickly the topic goes stale)
- Source reliability signal (how time-sensitive the source type is)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StalenessConfig:
    """Configuration for the staleness detector.

    Attributes:
        stale_threshold: Score >= this → flagged as stale (0.0–1.0).
        age_weight: Weight applied to the age factor.
        topic_weight: Weight applied to the topic volatility factor.
        source_weight: Weight applied to the source reliability signal.
    """

    stale_threshold: float = 0.7
    age_weight: float = 0.4
    topic_weight: float = 0.4
    source_weight: float = 0.2


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class StalenessResult:
    """Result of a staleness scoring operation.

    Attributes:
        score: Combined staleness score, 0.0 (fresh) to 1.0 (very stale).
        is_stale: True when score >= the configured threshold.
        age_score: Raw age sub-score (0.0–1.0).
        topic_score: Raw topic-volatility sub-score (0.0–1.0).
        source_score: Raw source-type sub-score (0.0–1.0).
        reason: Human-readable explanation of the score.
    """

    score: float
    is_stale: bool
    age_score: float
    topic_score: float
    source_score: float
    reason: str


# ---------------------------------------------------------------------------
# Keyword maps
# ---------------------------------------------------------------------------

# (keyword_lower → score)  — only the *max* matched score is used
_HIGH_VOLATILITY_KEYWORDS: dict[str, float] = {
    "ai": 0.9,
    "ml": 0.9,
    "llm": 0.9,
    "gpt": 0.9,
    "neural": 0.9,
    "transformer": 0.9,
    "model": 0.9,
    "benchmark": 0.9,
    "crypto": 0.9,
    "blockchain": 0.9,
    "nft": 0.9,
    "defi": 0.9,
    "framework": 0.7,
    "library": 0.7,
    "api": 0.7,
    "sdk": 0.7,
    "version": 0.7,
    "tutorial": 0.5,
    "guide": 0.5,
    "howto": 0.5,
}

_LOW_VOLATILITY_KEYWORDS: dict[str, float] = {
    "philosophy": 0.1,
    "history": 0.1,
    "mathematics": 0.1,
    "statistics": 0.1,
    "psychology": 0.1,
    "biology": 0.1,
    "physics": 0.1,
    "classic": 0.1,
    "fundamentals": 0.1,
    "principles": 0.1,
}

_DEFAULT_TOPIC_SCORE: float = 0.5

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class StalenessDetector:
    """Scores content for staleness.

    The overall score is a weighted combination of three sub-scores:

    * **age_score** — how old is the content?
    * **topic_score** — how quickly does this type of topic go stale?
    * **source_score** — how time-sensitive is the source type?

    Example::

        detector = StalenessDetector()
        result = detector.score(
            content_date=datetime(2021, 1, 1, tzinfo=timezone.utc),
            topic_keywords=["ai", "benchmark"],
            source_url="https://arxiv.org/abs/2101.00001",
        )
        print(result.is_stale, result.score)
    """

    def __init__(self, config: StalenessConfig | None = None) -> None:
        self.config: StalenessConfig = config if config is not None else StalenessConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        content_date: datetime,
        topic_keywords: list[str],
        source_url: str = "",
    ) -> StalenessResult:
        """Score content for staleness.

        Args:
            content_date: Publication / creation date of the content.
            topic_keywords: Keywords describing the content topic.
            source_url: URL of the source (used to infer source type).

        Returns:
            A :class:`StalenessResult` with combined score and sub-scores.
        """
        age_s = self._age_score(content_date)
        topic_s = self._topic_score(topic_keywords)
        source_s = self._source_score(source_url)

        cfg = self.config
        combined = (
            cfg.age_weight * age_s + cfg.topic_weight * topic_s + cfg.source_weight * source_s
        )
        # Clamp to [0.0, 1.0] for safety
        combined = max(0.0, min(1.0, combined))

        reason = self._build_reason(age_s, topic_s, source_s, combined)

        return StalenessResult(
            score=combined,
            is_stale=combined >= cfg.stale_threshold,
            age_score=age_s,
            topic_score=topic_s,
            source_score=source_s,
            reason=reason,
        )

    def batch_score(
        self,
        items: list[dict[str, Any]],
    ) -> list[StalenessResult]:
        """Score multiple items.

        Each item dict must have:

        * ``'date'`` — :class:`datetime`
        * ``'keywords'`` — ``list[str]``
        * ``'url'`` — ``str`` (optional, defaults to empty string)

        Args:
            items: List of content item dicts.

        Returns:
            List of :class:`StalenessResult`, one per input item, preserving order.
        """
        results: list[StalenessResult] = []
        for item in items:
            results.append(
                self.score(
                    content_date=item["date"],
                    topic_keywords=item["keywords"],
                    source_url=item.get("url", ""),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _age_score(self, content_date: datetime) -> float:
        """Score based on content age.

        Brackets:
        - < 30 days  → 0.0
        - 30–90 days → 0.2
        - 90–180 days → 0.4
        - 180–365 days → 0.6
        - 1–2 years  → 0.8
        - > 2 years  → 1.0

        Args:
            content_date: Publication date of the content.

        Returns:
            Age score in [0.0, 1.0].
        """
        now = datetime.now(UTC)
        # Ensure content_date is timezone-aware for comparison
        if content_date.tzinfo is None:
            content_date = content_date.replace(tzinfo=UTC)

        delta_days = (now - content_date).days

        if delta_days < 30:
            return 0.0
        elif delta_days < 90:
            return 0.2
        elif delta_days < 180:
            return 0.4
        elif delta_days < 365:
            return 0.6
        elif delta_days < 730:  # 2 years
            return 0.8
        else:
            return 1.0

    def _topic_score(self, keywords: list[str]) -> float:
        """Score based on topic volatility.

        Returns the *maximum* score across all matched keywords.  If no
        keywords match any known category the default (0.5) is returned.

        High-volatility (go stale quickly) → scores near 0.9 / 0.7 / 0.5
        Low-volatility (timeless) → scores near 0.1
        Default (unknown) → 0.5

        Args:
            keywords: Topic keywords for the content.

        Returns:
            Topic volatility score in [0.0, 1.0].
        """
        if not keywords:
            return _DEFAULT_TOPIC_SCORE

        best: float | None = None

        all_keyword_map: dict[str, float] = {
            **_HIGH_VOLATILITY_KEYWORDS,
            **_LOW_VOLATILITY_KEYWORDS,
        }

        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in all_keyword_map:
                val = all_keyword_map[kw_lower]
                if best is None or val > best:
                    best = val

        return best if best is not None else _DEFAULT_TOPIC_SCORE

    def _source_score(self, source_url: str) -> float:
        """Score based on source type inferred from URL.

        Mapping:
        - arxiv.org          → 0.2 (academic, stable reference)
        - github.com         → 0.5 (code changes but repos persist)
        - medium.com / substack.com / blog → 0.6 (opinion, may date quickly)
        - news sites         → 0.9 (very time-sensitive)
        - youtube.com        → 0.4 (videos persist but content may date)
        - default            → 0.5

        Args:
            source_url: URL of the source.

        Returns:
            Source-type score in [0.0, 1.0].
        """
        if not source_url:
            return 0.5

        url_lower = source_url.lower()

        if "arxiv.org" in url_lower:
            return 0.2
        if "github.com" in url_lower:
            return 0.5
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return 0.4
        if any(x in url_lower for x in ("medium.com", "substack.com", "blog")):
            return 0.6
        # News heuristics
        if any(
            x in url_lower
            for x in ("news.", ".news", "reuters", "bbc", "cnn", "nytimes", "theguardian")
        ):
            return 0.9

        return 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reason(
        age_score: float,
        topic_score: float,
        source_score: float,
        combined: float,
    ) -> str:
        """Build a human-readable explanation for the staleness score.

        Args:
            age_score: The age sub-score.
            topic_score: The topic sub-score.
            source_score: The source sub-score.
            combined: The final combined score.

        Returns:
            A non-empty descriptive string.
        """
        parts: list[str] = []

        # Age description
        if age_score == 0.0:
            parts.append("content is recent (< 30 days)")
        elif age_score == 0.2:
            parts.append("content is 1–3 months old")
        elif age_score == 0.4:
            parts.append("content is 3–6 months old")
        elif age_score == 0.6:
            parts.append("content is 6–12 months old")
        elif age_score == 0.8:
            parts.append("content is 1–2 years old")
        else:
            parts.append("content is more than 2 years old")

        # Topic description
        if topic_score >= 0.9:
            parts.append("covers a highly volatile topic")
        elif topic_score >= 0.7:
            parts.append("covers a moderately volatile topic")
        elif topic_score >= 0.5:
            parts.append("topic volatility is average")
        else:
            parts.append("covers a stable/timeless topic")

        # Source description
        if source_score >= 0.9:
            parts.append("from a time-sensitive news source")
        elif source_score >= 0.6:
            parts.append("from a blog/opinion source")
        elif source_score <= 0.2:
            parts.append("from a stable academic source")
        else:
            parts.append("from a moderately stable source")

        freshness = "stale" if combined >= 0.7 else "fresh"
        return f"Score {combined:.2f} ({freshness}): " + "; ".join(parts) + "."
