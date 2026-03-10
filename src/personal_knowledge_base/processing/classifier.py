"""Content classifier for routing URLs and text to KB collections."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Tier 1: URL rules (deterministic, fast)
# ---------------------------------------------------------------------------

URL_RULES: list[tuple[str, str]] = [
    (r"youtube\.com|youtu\.be", "videos"),
    (r"arxiv\.org", "papers"),
    (r"github\.com", "code"),
    (r".*\.pdf$", "papers"),
]

# ---------------------------------------------------------------------------
# Tier 2: Collection descriptions for embedding similarity
# ---------------------------------------------------------------------------

COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "videos": "YouTube videos, video tutorials, lecture recordings, video content",
    "papers": "Academic papers, research articles, arxiv preprints, PDFs, scientific literature",
    "code": "GitHub repositories, code projects, software libraries, programming resources",
    "general": "General web articles, blog posts, news, miscellaneous content",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ClassifierConfig:
    """Configuration for the content classifier.

    Attributes:
        similarity_threshold: Minimum cosine similarity score to accept a
            collection match from Tier-2 (embedding similarity). If all
            scores are below this threshold the classifier falls back to the
            ``"general"`` collection.
        ollama_url: Base URL for the Ollama API used when generating embeddings.
        embedding_model: Name of the Ollama model used for embeddings.
    """

    similarity_threshold: float = 0.3
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in the range [-1, 1].  Returns 0.0 for zero vectors.
    """
    dot: float = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a: float = sum(x * x for x in a) ** 0.5
    norm_b: float = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@dataclass
class ContentClassifier:
    """Routes content (URLs and text) to the appropriate KB collection.

    Classification is performed in three tiers:

    1. **URL rules** – fast, deterministic regex matching against known
       domains / file extensions.
    2. **Embedding similarity** – the content title/description is embedded
       and compared to per-collection description embeddings.  Requires a
       running Ollama instance.
    3. **Default fallback** – returns ``"general"`` when the similarity score
       is below :attr:`ClassifierConfig.similarity_threshold`.

    Example::

        classifier = ContentClassifier()

        # Tier-1: URL rule
        kb = classifier.classify("https://youtu.be/abc123", title="My video")
        assert kb == "videos"

        # Tier-3: fallback
        kb = classifier.classify("", title="")
        assert kb == "general"
    """

    config: ClassifierConfig = field(default_factory=ClassifierConfig)

    # Cache for collection description embeddings (populated lazily)
    _desc_embeddings: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Allow ``None`` config and substitute the default."""
        if self.config is None:
            self.config = ClassifierConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_by_url(self, url: str) -> str | None:
        """Return the KB collection name if the URL matches a known rule.

        Args:
            url: The URL to inspect.  May be an empty string or ``None``-like
                 value; both are treated as "no match".

        Returns:
            The collection name (e.g. ``"videos"``) if a rule matches,
            otherwise ``None``.
        """
        if not url:
            return None
        for pattern, collection in URL_RULES:
            if re.search(pattern, url, re.IGNORECASE):
                return collection
        return None

    def classify_by_content(self, title: str, description: str = "") -> str:
        """Return the KB collection name using embedding similarity.

        Embeds the combined ``title`` + ``description`` text and compares it
        against pre-computed embeddings of each collection description.  If
        the best cosine similarity is below
        :attr:`ClassifierConfig.similarity_threshold` the method returns
        ``"general"``.

        Args:
            title: Content title (used as primary signal).
            description: Optional supplementary description text.

        Returns:
            The best-matching collection name, or ``"general"`` on fallback.
        """
        text = " ".join(filter(None, [title, description])).strip()
        if not text:
            return "general"

        query_embedding = self._embed(text)
        desc_embeddings = self._get_desc_embeddings()

        best_collection = "general"
        best_score = self.config.similarity_threshold  # must beat threshold

        for collection, desc_embedding in desc_embeddings.items():
            score = _cosine_similarity(query_embedding, desc_embedding)
            if score > best_score:
                best_score = score
                best_collection = collection

        return best_collection

    def classify(self, url: str, title: str = "", description: str = "") -> str:
        """Main entry point: classify content and return a KB collection name.

        Attempts classification in order:

        1. URL rules (fast, deterministic).
        2. Embedding similarity on ``title`` + ``description``.
        3. Default ``"general"``.

        Args:
            url: The content URL.  May be empty.
            title: Content title.  Used for Tier-2 if Tier-1 does not match.
            description: Optional description.  Used alongside ``title`` in
                Tier-2.

        Returns:
            The target KB collection name (always a non-empty string).
        """
        # Tier 1
        result = self.classify_by_url(url)
        if result is not None:
            return result

        # Tier 2 + 3
        return self.classify_by_content(title, description)

    # ------------------------------------------------------------------
    # Embedding helpers (thin wrappers — easy to mock in tests)
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Embed a single text string using Ollama (synchronous convenience wrapper).

        This method is intentionally thin so that tests can patch it without
        requiring a running Ollama instance.

        Args:
            text: The text to embed.

        Returns:
            An embedding vector as a list of floats.

        Raises:
            RuntimeError: If the Ollama request fails.
        """
        import asyncio

        from personal_knowledge_base.processing.embedder import EmbedderConfig, OllamaEmbedder

        embedder_config = EmbedderConfig(
            model=self.config.embedding_model,
            ollama_url=self.config.ollama_url,
        )
        embedder = OllamaEmbedder(config=embedder_config)

        async def _run() -> list[float]:
            async with embedder:
                return await embedder.embed_text(text)

        return asyncio.run(_run())

    def _get_desc_embeddings(self) -> dict[str, list[float]]:
        """Return (and cache) embeddings for each collection description.

        Returns:
            Mapping of collection name → embedding vector.
        """
        if not self._desc_embeddings:
            for collection, desc in COLLECTION_DESCRIPTIONS.items():
                self._desc_embeddings[collection] = self._embed(desc)
        return self._desc_embeddings
