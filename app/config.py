"""
Application configuration using Pydantic settings.
Loads environment variables from .env file with sensible defaults.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for VisionAI.
    Values are loaded from environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database connection string for PostgreSQL with pgvector extension
    visionai_database_url: str = "postgresql+psycopg://postgres:1337@localhost:5432/visionai_db"

    # Directory where uploaded images are stored
    upload_dir: Path = Path("uploads")

    # Ollama API configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = ""

    # Vision model for generating image descriptions
    ollama_model: str = "qwen2.5vl:latest"

    # Embedding model for semantic search (llava has CLIP, optimized for similarity)
    ollama_embedding_model: str = "llava:latest"

    # Dimension of embedding vectors (llava:latest outputs 4096)
    ollama_embedding_dimension: int = 4096


# Singleton settings instance used throughout the application
settings = Settings()