"""Ollama embedding service for text-based semantic search."""

import time
import requests
import numpy as np

from app.config import settings

TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 5


class OllamaEmbeddingService:
    _instance: "OllamaEmbeddingService | None" = None

    def __new__(cls) -> "OllamaEmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
        return headers

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama's embedding API."""
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

                # L2 normalize
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
        """Generate embeddings for multiple texts (sequential, not batched by API)."""
        return [self.embed_text(text) for text in texts]


ollama_embedding_service = OllamaEmbeddingService()
