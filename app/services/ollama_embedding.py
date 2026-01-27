"""Ollama embedding service for text-based semantic search.
Converts text descriptions into vector embeddings using llama3.1.
"""

import numpy as np

from app.config import settings
from app.services.http_client import post_with_retry


class OllamaEmbeddingService:
    """
    Service for generating text embeddings via Ollama API.

    Uses a singleton pattern to reuse the same instance across requests.
    Embeddings are L2-normalized for consistent cosine similarity calculations.
    """

    _instance: "OllamaEmbeddingService | None" = None

    def __new__(cls) -> "OllamaEmbeddingService":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def embed_text(self, text: str) -> list[float]:
        """
        Generate a vector embedding for the given text.

        Args:
            text: The text to embed (typically an image description)

        Returns:
            L2-normalized embedding vector as a list of floats

        Raises:
            RuntimeError: If embedding fails after all retries
        """
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

        return embedding.tolist()

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Note: Ollama API doesn't support true batching, so this processes sequentially.
        """
        return [self.embed_text(text) for text in texts]


# Singleton instance used throughout the application
ollama_embedding_service = OllamaEmbeddingService()
