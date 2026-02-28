"""Embedding providers for the Zenki memory system."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vector representations.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, one per input text.
        """
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers locally.

    Uses the ``all-MiniLM-L6-v2`` model by default (384 dimensions).
    The model is lazy-loaded on the first call to :meth:`embed`.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install 'zenki[embeddings]' or "
                "pip install sentence-transformers"
            )
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the local sentence-transformers model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors (384 dimensions by default).
        """
        if self._model is None:
            self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]


class DummyEmbeddingProvider(EmbeddingProvider):
    """Dummy embedding provider for testing.

    Returns random unit vectors of the specified dimension.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return random unit vectors for each input text.

        The vectors are normalised so that cosine similarity calculations
        produce meaningful (though random) values.

        Args:
            texts: List of text strings (content is ignored).

        Returns:
            List of random float vectors.
        """
        results: list[list[float]] = []
        for _ in texts:
            vec = [random.gauss(0, 1) for _ in range(self.dimension)]
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


def get_embedding_provider(config: dict) -> EmbeddingProvider:
    """Factory function to create an embedding provider from configuration.

    Args:
        config: Dictionary with at least a ``provider`` key.
            - ``"local"``: Uses sentence-transformers (default).
            - ``"dummy"``: Returns random vectors (for testing).
            Additional keys like ``model`` are passed to the provider.

    Returns:
        An :class:`EmbeddingProvider` instance.
    """
    provider_type = config.get("provider", "local")

    if provider_type == "dummy":
        dimension = config.get("dimension", 384)
        return DummyEmbeddingProvider(dimension=dimension)

    if provider_type == "local":
        model_name = config.get("model", LocalEmbeddingProvider.DEFAULT_MODEL)
        return LocalEmbeddingProvider(model_name=model_name)

    raise ValueError(f"Unknown embedding provider: {provider_type!r}")
