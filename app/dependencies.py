from collections.abc import Generator

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services.vision import VisionService, vision_service
from app.services.ollama_embedding import OllamaEmbeddingService, ollama_embedding_service


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_vision_service() -> VisionService:
    return vision_service


def get_ollama_embedding_service() -> OllamaEmbeddingService:
    return ollama_embedding_service
