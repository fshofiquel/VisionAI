from collections.abc import Generator

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services.embedding import EmbeddingService, embedding_service
from app.services.llava import LLaVAService, llava_service


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_embedding_service() -> EmbeddingService:
    return embedding_service


def get_llava_service() -> LLaVAService:
    return llava_service
