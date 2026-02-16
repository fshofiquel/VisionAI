"""
Ollama embedding service for semantic search.

This module provides text-to-vector embedding functionality using Ollama's
embedding API. Embeddings are used to enable semantic similarity search
across image descriptions.

Key features:
- L2 normalization for consistent cosine similarity calculations
- In-memory caching for fast repeated queries
- Singleton pattern for efficient resource usage

Example:
    service = OllamaEmbeddingService()
    embedding = service.embed_text("a photo of a sunset")
    # Returns: [0.023, -0.156, 0.089, ...] (4096 floats)
"""

import hashlib
from typing import Final

import numpy as np

from app.config import settings
from app.services.http_client import post_with_retry

# =============================================================================
# Configuration
# =============================================================================

# Maximum number of cached embeddings (LRU-style eviction when exceeded)
_CACHE_MAX_SIZE: Final[int] = 1000

# Number of entries to evict when cache is full
_CACHE_EVICTION_COUNT: Final[int] = 100

# In-memory cache for query embeddings {hash -> embedding}
_embedding_cache: dict[str, list[float]] = {}


# =============================================================================
# Embedding Service
# =============================================================================

class OllamaEmbeddingService:
    """
    Service for generating text embeddings via Ollama API.

    Converts text strings into dense vector representations that capture
    semantic meaning. These vectors enable similarity search by comparing
    distances in the embedding space.

    Features:
        - Singleton pattern: Only one instance exists application-wide
        - L2 normalization: All embeddings are unit vectors for cosine similarity
        - Caching: Query embeddings are cached for instant repeated searches

    Attributes:
        _instance: Class-level singleton instance

    Example:
        service = OllamaEmbeddingService()
        vec = service.embed_text("sunset over mountains")
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

        The embedding is L2-normalized to unit length, ensuring consistent
        cosine similarity calculations regardless of text length.

        Args:
            text: The text to embed (description or search query)
            use_cache: If True, check cache first and store result.
                      Set False for indexing operations.

        Returns:
            L2-normalized embedding vector as a list of floats.
            Length matches OLLAMA_EMBEDDING_DIMENSION config.

        Raises:
            RuntimeError: If embedding fails after all retries
        """
        # Generate cache key from text hash
        cache_key = hashlib.md5(text.encode()).hexdigest()

        # Return cached embedding if available
        if use_cache and cache_key in _embedding_cache:
            return _embedding_cache[cache_key]

        # Call Ollama embedding API
        payload = {
            "model": settings.ollama_embedding_model,
            "prompt": text,
        }
        response = post_with_retry("/api/embeddings", payload, "Embedding")
        embedding = response["embedding"]

        # L2 normalize for consistent cosine similarity
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm

        result = embedding_array.tolist()

        # Cache the result with simple LRU-style eviction
        if use_cache:
            self._cache_embedding(cache_key, result)

        return result

    def _cache_embedding(self, cache_key: str, embedding: list[float]) -> None:
        """
        Store an embedding in the cache with overflow handling.

        Evicts oldest entries when cache exceeds maximum size.

        Args:
            cache_key: Hash key for the embedding
            embedding: Vector to cache
        """
        if len(_embedding_cache) >= _CACHE_MAX_SIZE:
            # Evict oldest entries (dict maintains insertion order in Python 3.7+)
            keys_to_remove = list(_embedding_cache.keys())[:_CACHE_EVICTION_COUNT]
            for key in keys_to_remove:
                del _embedding_cache[key]

        _embedding_cache[cache_key] = embedding

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Processes texts sequentially since Ollama API doesn't support
        true batching. Caching is disabled for batch operations.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text
        """
        return [self.embed_text(text, use_cache=False) for text in texts]


# Singleton instance used throughout the application
ollama_embedding_service = OllamaEmbeddingService()
