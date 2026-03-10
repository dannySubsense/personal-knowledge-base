"""Suggestions engine for related content and knowledge gap identification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from personal_knowledge_base.processing.embedder import EmbedderConfig, OllamaEmbedder
from personal_knowledge_base.storage.vector_store import VectorStore, VectorStoreConfig

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "him",
    "her",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "about",
    "using",
    "use",
    "get",
    "set",
    "all",
}

_CONNECTORS = {"and", "or", "with", "using", "about", "for"}

DEFAULT_COLLECTIONS = ["videos", "papers", "code", "general"]


@dataclass
class SuggestionsConfig:
    """Configuration for the suggestions engine.

    Attributes:
        qdrant_url: URL of the Qdrant server.
        ollama_url: URL of the Ollama API.
        embedding_model: Model name for generating embeddings.
        max_suggestions: Maximum number of suggestions to return.
        min_score: Minimum relevance score for related content.
        gap_threshold: Score below this value is considered a knowledge gap.
    """

    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    max_suggestions: int = 5
    min_score: float = 0.3
    gap_threshold: float = 0.5


@dataclass
class Suggestion:
    """A single suggestion item.

    Attributes:
        text: The suggestion text.
        source: Where the suggestion came from ("related" or "gap").
        score: Relevance score.
        collection: Which KB collection this came from.
    """

    text: str
    source: str
    score: float
    collection: str


@dataclass
class SuggestionResult:
    """Result from the suggestions engine.

    Attributes:
        query: The original query.
        related: Content already in KB related to the query.
        gaps: Topics the query touches that aren't well covered.
        summary: Human-readable summary of the results.
    """

    query: str
    related: list[Suggestion] = field(default_factory=list)
    gaps: list[Suggestion] = field(default_factory=list)
    summary: str = ""


class SuggestionsEngine:
    """Engine for generating content suggestions and identifying knowledge gaps.

    Given a query, this engine:
    1. Embeds the query using Ollama.
    2. Searches each collection for related content (score >= min_score).
    3. Identifies sub-topics that are not well covered (score < gap_threshold).
    4. Returns a SuggestionResult with related items and gaps.
    """

    def __init__(self, config: SuggestionsConfig | None = None) -> None:
        """Initialise the engine.

        Args:
            config: Optional configuration. Defaults are used if not provided.
        """
        self.config = config or SuggestionsConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, query: str, collections: list[str] | None = None) -> SuggestionResult:
        """Generate suggestions for a query.

        Args:
            query: The topic or question to explore.
            collections: Collections to search. Defaults to DEFAULT_COLLECTIONS.

        Returns:
            SuggestionResult with related content and identified gaps.
        """
        if collections is None:
            collections = DEFAULT_COLLECTIONS

        embedder = OllamaEmbedder(
            EmbedderConfig(
                model=self.config.embedding_model,
                ollama_url=self.config.ollama_url,
            )
        )
        query_vector: list[float] = asyncio.run(embedder.embed_text(query))

        related = self._find_related(query_vector, collections)
        gaps = self._find_gaps(query, query_vector, collections)

        result = SuggestionResult(query=query, related=related, gaps=gaps)
        result.summary = self.format_suggestions(result)
        return result

    def format_suggestions(self, result: SuggestionResult) -> str:
        """Format a SuggestionResult as readable text.

        Args:
            result: The suggestion result to format.

        Returns:
            A human-readable string summarising the suggestions.
        """
        lines: list[str] = [f'Suggestions for: "{result.query}"', ""]

        if result.related:
            lines.append("📚 Related content in your knowledge base:")
            for i, s in enumerate(result.related, 1):
                lines.append(f"  {i}. [{s.collection}] {s.text} (score: {s.score:.2f})")
        else:
            lines.append("📚 No related content found in your knowledge base.")

        lines.append("")

        if result.gaps:
            lines.append("🔍 Potential knowledge gaps:")
            for i, g in enumerate(result.gaps, 1):
                lines.append(
                    f"  {i}. [{g.collection}] '{g.text}' is not well covered"
                    f" (best score: {g.score:.2f})"
                )
        else:
            lines.append("✅ No significant knowledge gaps detected.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_related(self, query_vector: list[float], collections: list[str]) -> list[Suggestion]:
        """Search collections for related content.

        Args:
            query_vector: Embedding of the query.
            collections: Collections to search.

        Returns:
            Up to max_suggestions Suggestion objects sorted by score descending.
        """
        all_results: list[Suggestion] = []

        for collection in collections:
            store = VectorStore(
                VectorStoreConfig(
                    qdrant_url=self.config.qdrant_url,
                    collection_name=collection,
                )
            )
            try:
                store.connect()
                hits = store.search(
                    query_vector=query_vector,
                    limit=self.config.max_suggestions,
                    score_threshold=self.config.min_score,
                )
                for hit in hits:
                    all_results.append(
                        Suggestion(
                            text=hit.chunk_text,
                            source="related",
                            score=hit.score,
                            collection=collection,
                        )
                    )
            except Exception:
                # Collection may not exist; skip gracefully.
                pass

        all_results.sort(key=lambda s: s.score, reverse=True)
        return all_results[: self.config.max_suggestions]

    def _find_gaps(
        self,
        query: str,
        query_vector: list[float],  # noqa: ARG002  – kept for API symmetry
        collections: list[str],
    ) -> list[Suggestion]:
        """Identify sub-topics in the query that are not well covered.

        For each sub-topic extracted from the query, the best match across all
        collections is retrieved. If that best score is below gap_threshold,
        the sub-topic is flagged as a gap.

        Args:
            query: Original query string.
            query_vector: Embedding of the full query (unused directly but
                kept for interface symmetry).
            collections: Collections to search.

        Returns:
            Suggestion objects representing knowledge gaps.
        """
        subtopics = self._extract_subtopics(query)
        embedder = OllamaEmbedder(
            EmbedderConfig(
                model=self.config.embedding_model,
                ollama_url=self.config.ollama_url,
            )
        )

        gaps: list[Suggestion] = []

        for subtopic in subtopics:
            subtopic_vector: list[float] = asyncio.run(embedder.embed_text(subtopic))
            best_score = 0.0
            best_collection = collections[0] if collections else "general"

            for collection in collections:
                store = VectorStore(
                    VectorStoreConfig(
                        qdrant_url=self.config.qdrant_url,
                        collection_name=collection,
                    )
                )
                try:
                    store.connect()
                    hits = store.search(
                        query_vector=subtopic_vector,
                        limit=1,
                    )
                    if hits and hits[0].score > best_score:
                        best_score = hits[0].score
                        best_collection = collection
                except Exception:
                    pass

            if best_score < self.config.gap_threshold:
                gaps.append(
                    Suggestion(
                        text=subtopic,
                        source="gap",
                        score=best_score,
                        collection=best_collection,
                    )
                )

        return gaps

    def _extract_subtopics(self, query: str) -> list[str]:
        """Extract sub-topics from a query string.

        The query is split on common connector words, then short and stop words
        are removed. Up to 3 sub-topics are returned.

        Args:
            query: The query string to analyse.

        Returns:
            A list of up to 3 sub-topic strings.
        """
        # Replace connectors with a split marker
        normalised = query.lower()
        for connector in _CONNECTORS:
            normalised = normalised.replace(f" {connector} ", " | ")

        parts = [p.strip() for p in normalised.split("|") if p.strip()]

        # Within each part, also do a word-level filter for noun-like tokens
        subtopics: list[str] = []
        for part in parts:
            words = [w for w in part.split() if len(w) > 3 and w not in _STOP_WORDS]
            if words:
                candidate = " ".join(words)
                if candidate not in subtopics:
                    subtopics.append(candidate)

        return subtopics[:3]
