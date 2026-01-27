"""
Image database model with vector embedding support.
Stores image metadata and description embeddings for semantic search.
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

    The embedding column stores a vector representation of the image description,
    enabling semantic similarity search using pgvector's cosine distance.
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

    # AI-generated description from vision model (LLaVA/Qwen2.5-VL)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Vector embedding of the description for semantic search
    embedding = mapped_column(
        Vector(settings.ollama_embedding_dimension), nullable=True
    )

    # Timestamp of when the image was added
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
