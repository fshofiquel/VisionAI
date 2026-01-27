"""Ollama embedding service for text-based semantic search.
Converts text descriptions into vector embeddings using llama3.1.
"""

import time

import numpy as np
import requests

from app.config import settings

# Request configuration
TIMEOUT = 120  # Maximum seconds to wait for API response
MAX_RETRIES = 3  # Number of retry attempts on failure
RETRY_DELAY = 5  # Seconds to wait between retries


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

    @property
    def _headers(self) -> dict[str, str]:
        """Build request headers, including auth if API key is configured."""
        headers = {"Content-Type": "application/json"}
        if settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
        return headers

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

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{settings.ollama_base_url}/api/embeddings",
                    json=payload,
                    headers=self._headers,
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                embedding = resp.json()["embedding"]

                # L2 normalize for consistent cosine similarity
                embedding = np.array(embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                return embedding.tolist()
            except (requests.Timeout, requests.ConnectionError):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY)

        raise RuntimeError("Failed to get embedding after retries")

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Note: Ollama API doesn't support true batching, so this processes sequentially.
        """
        return [self.embed_text(text) for text in texts]


# Singleton instance used throughout the application
ollama_embedding_service = OllamaEmbeddingService()
