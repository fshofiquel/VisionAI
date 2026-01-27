"""
FastAPI dependency injection functions.
Provides database sessions and AI service instances to route handlers.
"""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services.vision import VisionService, vision_service
from app.services.ollama_embedding import OllamaEmbeddingService, ollama_embedding_service


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    Automatically closes the session when the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_vision_service() -> VisionService:
    """Dependency that provides the vision model service for image descriptions."""
    return vision_service


def get_ollama_embedding_service() -> OllamaEmbeddingService:
    """Dependency that provides the embedding service for semantic search."""
    return ollama_embedding_service
