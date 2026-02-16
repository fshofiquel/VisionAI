"""
FastAPI dependency injection functions.

This module provides dependency functions that are injected into route
handlers using FastAPI's Depends() system. Dependencies include:

- Database session management with automatic cleanup
- AI service instances (vision model, embedding model)

Usage in route handlers:
    @router.get("/items")
    def get_items(db: Session = Depends(get_db)):
        return db.query(Item).all()
"""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services.vision import VisionService, vision_service
from app.services.ollama_embedding import OllamaEmbeddingService, ollama_embedding_service


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.

    Creates a new SQLAlchemy session for each request and ensures it is
    properly closed when the request completes, even if an error occurs.

    Yields:
        Session: SQLAlchemy database session

    Example:
        @router.get("/images")
        def list_images(db: Session = Depends(get_db)):
            return db.query(Image).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_vision_service() -> VisionService:
    """
    Dependency that provides the vision model service.

    Returns the singleton VisionService instance used for generating
    image descriptions via multimodal models (LLaVA, Qwen2.5-VL).

    Returns:
        VisionService: Singleton vision service instance
    """
    return vision_service


def get_ollama_embedding_service() -> OllamaEmbeddingService:
    """
    Dependency that provides the embedding service.

    Returns the singleton OllamaEmbeddingService instance used for
    generating text embeddings for semantic search.

    Returns:
        OllamaEmbeddingService: Singleton embedding service instance
    """
    return ollama_embedding_service
