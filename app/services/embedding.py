import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    _instance: "EmbeddingService | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(settings.clip_model_name)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.load_model()
        return self._model

    def embed_image(self, image: Image.Image) -> list[float]:
        embedding = self.model.encode(image, convert_to_numpy=True)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()


embedding_service = EmbeddingService()