"""Ollama embedding service for text-based semantic search.
Converts text descriptions into vector embeddings.
Includes caching for fast repeated search queries.
"""

import hashlib

import numpy as np

from app.config import settings
from app.services.http_client import post_with_retry

# In-memory cache for query embeddings (makes repeated searches instant)
_embedding_cache: dict[str, list[float]] = {}
_CACHE_MAX_SIZE = 1000


class OllamaEmbeddingService:
    """
    Service for generating text embeddings via Ollama API.

    Uses a singleton pattern to reuse the same instance across requests.
    Embeddings are L2-normalized for consistent cosine similarity calculations.
    Query embeddings are cached for fast repeated searches.
    """

    _instance: "OllamaEmbeddingService | None" = None

    def __new__(cls) -> "OllamaEmbeddingService":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate a vector embedding for the given text.

        Args:
            text: The text to embed (description or search query)
            use_cache: If True, check/store in cache (use for search queries)

        Returns:
            L2-normalized embedding vector as a list of floats

        Raises:
            RuntimeError: If embedding fails after all retries
        """
        # Check cache first (instant for repeated queries)
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if use_cache and cache_key in _embedding_cache:
            return _embedding_cache[cache_key]

        # Call Ollama API
        payload = {
            "model": settings.ollama_embedding_model,
            "prompt": text,
        }

        response = post_with_retry("/api/embeddings", payload, "Embedding")
        embedding = response["embedding"]

        # L2 normalize for consistent cosine similarity
        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        result = embedding.tolist()

        # Cache the result (evict old entries if full)
        if use_cache:
            if len(_embedding_cache) >= _CACHE_MAX_SIZE:
                # Simple eviction: remove first 100 entries
                keys_to_remove = list(_embedding_cache.keys())[:100]
                for key in keys_to_remove:
                    del _embedding_cache[key]
            _embedding_cache[cache_key] = result

        return result

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (for indexing, not search).

        Note: Ollama API doesn't support true batching, so this processes sequentially.
        Caching disabled for batch operations.
        """
        return [self.embed_text(text, use_cache=False) for text in texts]


# Singleton instance used throughout the application
ollama_embedding_service = OllamaEmbeddingService()
