"""
Database models for VisionAI image storage and search.

This module defines the SQLAlchemy ORM models for storing image metadata
and vector embeddings. Uses PostgreSQL with pgvector extension for
efficient similarity search.
"""

import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.database import Base


class Image(Base):
    """
    SQLAlchemy model for storing image metadata and embeddings.

    Attributes:
        id: Primary key, auto-incremented.
        filename: Stored filename (UUID-based for uniqueness).
        original_filename: User's original filename.
        filepath: Full path to the stored image file.
        content_type: MIME type (e.g., "image/jpeg").
        file_size: File size in bytes.
        description: AI-generated description from vision model.
        embedding: Vector embedding for semantic search (4096 dimensions).
        created_at: Timestamp when the image was uploaded.

    The embedding column enables semantic similarity search using
    pgvector's cosine distance function.
    """

    __tablename__ = "images"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # File information
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)

    # AI-generated description from vision model (LLaVA)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Vector embedding of description for semantic search
    embedding = mapped_column(
        Vector(settings.ollama_embedding_dimension), nullable=True
    )

    # Timestamp of when the image was added
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"<Image(id={self.id}, filename='{self.original_filename}')>"

