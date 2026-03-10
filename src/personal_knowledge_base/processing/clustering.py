"""Sub-category auto-suggestion via clustering of document embeddings."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# Common English stop words to filter out during term extraction
_STOP_WORDS: frozenset[str] = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
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
        "can",
        "not",
        "no",
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
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "which",
        "who",
        "what",
        "when",
        "where",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
    ]
)


@dataclass
class ClusteringConfig:
    """Configuration for the KB clusterer.

    Attributes:
        min_docs_to_cluster: Minimum number of documents required before
            clustering is attempted.
        min_cluster_size: Minimum number of documents to form a cluster
            (used as HDBSCAN min_cluster_size).
        min_samples: HDBSCAN min_samples parameter.
        qdrant_url: URL of the Qdrant instance (metadata only; clustering
            itself does not connect to Qdrant).
    """

    min_docs_to_cluster: int = 20
    min_cluster_size: int = 3
    min_samples: int = 2
    qdrant_url: str = "http://localhost:6333"


@dataclass
class Cluster:
    """A single cluster of documents.

    Attributes:
        id: Numeric cluster identifier (0-based).
        label: Suggested sub-category name generated from top terms.
        size: Number of documents belonging to this cluster.
        top_terms: Representative terms extracted from chunk_text.
        sample_sources: Up to 3 source URLs as examples.
    """

    id: int
    label: str
    size: int
    top_terms: list[str]
    sample_sources: list[str]


@dataclass
class ClusteringResult:
    """Result of a clustering run.

    Attributes:
        collection: Name of the KB collection that was clustered.
        total_docs: Total number of documents processed.
        clusters: List of discovered clusters.
        noise_count: Number of documents not assigned to any cluster.
        summary: Human-readable summary of the clustering run.
    """

    collection: str
    total_docs: int
    clusters: list[Cluster] = field(default_factory=list)
    noise_count: int = 0
    summary: str = ""


class KBClusterer:
    """Clusters KB document embeddings to suggest sub-categories.

    The caller is responsible for fetching vectors and payloads from Qdrant;
    this class operates purely on the data it receives.
    """

    def __init__(self, config: ClusteringConfig | None = None) -> None:
        """Initialise with optional config (defaults used if omitted)."""
        self.config = config or ClusteringConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_collection(
        self,
        collection: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> ClusteringResult:
        """Cluster vectors and return sub-category suggestions.

        Args:
            collection: Name of the Qdrant collection being clustered.
            vectors: List of embedding vectors, one per document.
            payloads: List of payload dicts corresponding to each vector.
                      Expected to contain a ``chunk_text`` field and
                      optionally a ``source`` field.

        Returns:
            A :class:`ClusteringResult`.  If fewer than
            ``config.min_docs_to_cluster`` vectors are provided, a result
            with an empty cluster list and an explanatory summary is
            returned immediately.
        """
        total = len(vectors)

        if total < self.config.min_docs_to_cluster:
            return ClusteringResult(
                collection=collection,
                total_docs=total,
                clusters=[],
                noise_count=total,
                summary="Not enough documents to cluster.",
            )

        arr = np.array(vectors, dtype=np.float32)
        labels = self._run_clustering(arr)

        unique_labels = sorted(set(labels))
        noise_count = int(np.sum(np.array(labels) == -1))

        clusters: list[Cluster] = []
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue

            mask = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            texts = [str(payloads[i].get("chunk_text", "")) for i in mask]
            sources: list[str] = []
            for i in mask:
                src = payloads[i].get("source", "")
                if src and src not in sources:
                    sources.append(str(src))
                if len(sources) >= 3:
                    break

            top_terms = self._extract_top_terms(texts)
            label = self._generate_label(top_terms)

            clusters.append(
                Cluster(
                    id=cluster_id,
                    label=label,
                    size=len(mask),
                    top_terms=top_terms,
                    sample_sources=sources,
                )
            )

        n_clusters = len(clusters)
        summary = (
            f"Found {n_clusters} sub-categor{'y' if n_clusters == 1 else 'ies'} "
            f"across {total} documents "
            f"({noise_count} uncategorised)."
        )

        return ClusteringResult(
            collection=collection,
            total_docs=total,
            clusters=clusters,
            noise_count=noise_count,
            summary=summary,
        )

    def format_results(self, result: ClusteringResult) -> str:
        """Format a :class:`ClusteringResult` as human-readable text.

        Args:
            result: The clustering result to format.

        Returns:
            A multi-line string suitable for display.
        """
        lines: list[str] = [
            f"Collection : {result.collection}",
            f"Total docs : {result.total_docs}",
            f"Clusters   : {len(result.clusters)}",
            f"Noise docs : {result.noise_count}",
            f"Summary    : {result.summary}",
        ]
        if result.clusters:
            lines.append("")
            lines.append("Sub-categories:")
            for cluster in result.clusters:
                lines.append(f"  [{cluster.id}] {cluster.label}  ({cluster.size} docs)")
                lines.append(f"       Terms  : {', '.join(cluster.top_terms)}")
                if cluster.sample_sources:
                    lines.append(f"       Sources: {', '.join(cluster.sample_sources[:3])}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_clustering(self, arr: np.ndarray) -> list[int]:
        """Run HDBSCAN (or DBSCAN fallback) and return per-document labels."""
        if HAS_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
            )
            clusterer.fit(arr)
            return [int(lbl) for lbl in clusterer.labels_]
        else:
            from sklearn.cluster import DBSCAN

            clusterer = DBSCAN(
                eps=0.5,
                min_samples=self.config.min_samples,
            )
            clusterer.fit(arr)
            return [int(lbl) for lbl in clusterer.labels_]

    def _generate_label(self, top_terms: list[str]) -> str:
        """Generate a human-readable cluster label from the top terms.

        Takes up to the first two terms and title-cases them.

        Args:
            top_terms: List of representative terms for the cluster.

        Returns:
            A short label string, e.g. ``"Neural Transformers"``.
        """
        if not top_terms:
            return "Unlabelled"
        selected = top_terms[:2]
        return " ".join(t.capitalize() for t in selected)

    def _extract_top_terms(self, texts: list[str], n: int = 5) -> list[str]:
        """Extract the top *n* terms from a list of texts.

        Uses a simple term-frequency approach: tokenise each text,
        remove stop words and short tokens, count occurrences, return
        the *n* most common.

        Args:
            texts: Raw text strings to analyse.
            n: Number of top terms to return.

        Returns:
            A list of up to *n* term strings, most frequent first.
        """
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = re.findall(r"[a-zA-Z]+", text.lower())
            for token in tokens:
                if token not in _STOP_WORDS and len(token) > 2:
                    counter[token] += 1
        return [term for term, _ in counter.most_common(n)]
