"""Content classifier for routing URLs and text to topic-based Knowledge Bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Fallback KB list (used when registry is empty or unavailable)
# ---------------------------------------------------------------------------

DEFAULT_KBS: list[dict[str, str]] = [
    {
        "id": "quant-trading",
        "description": (
            "Quantitative finance, trading strategies, factor models, risk management,"
            " portfolio construction, backtesting"
        ),
    },
    {
        "id": "ml-ai",
        "description": (
            "Machine learning, deep learning, large language models, transformers,"
            " neural networks, AI research"
        ),
    },
    {
        "id": "general",
        "description": "General web content, articles, blog posts, miscellaneous topics",
    },
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ClassifierConfig:
    """Configuration for the content classifier.

    Attributes:
        similarity_threshold: Minimum cosine similarity score to accept a
            KB match from embedding similarity. If all scores are below this
            threshold the classifier falls back to the ``"general"`` KB.
        ollama_url: Base URL for the Ollama API used when generating embeddings.
        embedding_model: Name of the Ollama model used for embeddings.
        db_path: Path to the PKB metadata SQLite database used by KBRegistry.
    """

    similarity_threshold: float = 0.3
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    db_path: str = "~/pkb-data/pkb_metadata.db"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ContentClassifier:
    """Routes content (URLs and text) to the appropriate topic-based Knowledge Base.

    Classification is performed in two tiers:

    1. **URL hint** – fast, optional pattern matching for *unambiguous* URLs
       (e.g. a URL containing "quant" is likely quant-trading).
    2. **Embedding similarity** – the content title/description is embedded
       and compared to KB description embeddings.  Requires a running Ollama
       instance.
    3. **Default fallback** – returns ``"general"`` when the similarity score
       is below :attr:`ClassifierConfig.similarity_threshold`.

    The ``registry`` argument accepts any object with a ``list_kbs()`` method
    that returns objects having ``.id`` and ``.description`` attributes.
    Pass ``None`` to use the built-in :data:`DEFAULT_KBS` fallback.

    Example::

        classifier = ContentClassifier()

        # URL hint (quant finance URL)
        kb = classifier.classify("https://quantopian.com/research", title="")
        # → "quant-trading"

        # Fallback
        kb = classifier.classify("", title="")
        # → "general"
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        registry: Any | None = None,
    ) -> None:
        """Initialise the classifier.

        Args:
            config: Classifier configuration.  Uses defaults when ``None``.
            registry: Optional KBRegistry instance.  When provided its
                ``list_kbs()`` method is called to retrieve the topic KBs.
                When ``None`` (or the registry returns an empty list) the
                classifier falls back to :data:`DEFAULT_KBS`.
        """
        self.config: ClassifierConfig = config if config is not None else ClassifierConfig()
        self._registry: Any | None = registry

        # Cache for KB description embeddings: kb_id → vector
        self._desc_embeddings: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, url: str, title: str = "", description: str = "") -> str:
        """Route content to a topic KB id.

        Steps:
        1. Load KBs from registry (or fallback list).
        2. Try URL hint — only for unambiguous patterns (fast, optional).
        3. Embed ``title + description`` and compare to KB descriptions.
        4. Return kb_id of best match, or ``"general"`` below threshold.

        Args:
            url: The content URL.  May be empty.
            title: Content title.  Used for embedding similarity.
            description: Optional description.  Combined with title for similarity.

        Returns:
            The target KB id (always a non-empty string).
        """
        kbs = self._get_kbs()

        # Tier 1: URL hint (fast, only for unambiguous cases)
        hint = self.classify_by_url(url)
        if hint is not None and any(_kb_id(kb) == hint for kb in kbs):
            return hint

        # Tier 2 + 3: embedding similarity → fallback "general"
        return self.classify_by_content(title, description, kbs=kbs)

    def classify_by_url(self, url: str) -> str | None:
        """Return a topic KB hint from URL, or ``None`` if ambiguous.

        Only returns a hint for *unambiguous* URL patterns:

        - URL contains ``"quant"``, ``"trading"``, ``"finance"``, or ``"factor"``
          → ``"quant-trading"``
        - URL contains ``"arxiv"`` or ``"scholar.google"``
          → ``None`` (could be quant *or* ML)
        - ``youtube.com``, ``github.com``
          → ``None`` (too ambiguous without content)

        These are hints only — :meth:`classify` may override with content
        similarity when the hint is below threshold.

        Args:
            url: The URL to inspect.

        Returns:
            A KB id hint, or ``None`` when the URL is ambiguous or empty.
        """
        if not url:
            return None

        url_lower = url.lower()

        # Unambiguous quant-finance signals in the URL
        quant_tokens = ("quant", "trading", "finance", "factor")
        if any(token in url_lower for token in quant_tokens):
            return "quant-trading"

        # Everything else is ambiguous — return None
        return None

    def classify_by_content(
        self,
        title: str,
        description: str = "",
        kbs: list[Any] | None = None,
    ) -> str:
        """Embed title + description and return the best-matching KB id.

        Compares cosine similarity of the content embedding against each KB's
        description embedding.  Returns ``"general"`` when all scores are
        below :attr:`ClassifierConfig.similarity_threshold`.

        Args:
            title: Content title (primary signal).
            description: Optional supplementary text.
            kbs: KB list to compare against.  When ``None`` the registry
                (or fallback list) is used.

        Returns:
            The best-matching KB id, or ``"general"`` as fallback.
        """
        text = " ".join(filter(None, [title, description])).strip()
        if not text:
            return "general"

        if kbs is None:
            kbs = self._get_kbs()

        query_embedding = self._embed(text)

        best_kb_id = "general"
        best_score = self.config.similarity_threshold  # must strictly beat threshold

        for kb in kbs:
            kb_id = _kb_id(kb)
            kb_desc = _kb_description(kb)

            # Lazily compute and cache description embedding
            if kb_id not in self._desc_embeddings:
                self._desc_embeddings[kb_id] = self._embed(kb_desc)

            score = self._cosine_similarity(query_embedding, self._desc_embeddings[kb_id])
            if score > best_score:
                best_score = score
                best_kb_id = kb_id

        return best_kb_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_kbs(self) -> list[Any]:
        """Load KBs from registry, falling back to :data:`DEFAULT_KBS`.

        Returns:
            List of KnowledgeBase-like objects (with ``.id`` and
            ``.description`` attributes) or plain dicts from the fallback.
        """
        if self._registry is not None:
            try:
                kbs: list[Any] = self._registry.list_kbs()
                if kbs:
                    return kbs
            except Exception:  # noqa: BLE001
                pass
        return list(DEFAULT_KBS)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
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

    def _embed(self, text: str) -> list[float]:
        """Embed a single text string using Ollama (synchronous wrapper).

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


# ---------------------------------------------------------------------------
# Duck-typing helpers (work for both dict and object KBs)
# ---------------------------------------------------------------------------


def _kb_id(kb: Any) -> str:
    """Return the id of a KB object or dict."""
    if isinstance(kb, dict):
        return str(kb["id"])
    return str(kb.id)


def _kb_description(kb: Any) -> str:
    """Return the description of a KB object or dict."""
    if isinstance(kb, dict):
        return str(kb["description"])
    return str(kb.description)
