"""
Application configuration using Pydantic settings.

This module provides centralized configuration management for VisionAI.
Settings are loaded from environment variables or a .env file, with
sensible defaults for local development.

Environment Variables:
    VISIONAI_DATABASE_URL: PostgreSQL connection string with pgvector
    UPLOAD_DIR: Directory for storing uploaded images
    OLLAMA_BASE_URL: Ollama API server URL
    OLLAMA_API_KEY: Optional API key for Ollama
    OLLAMA_MODEL: Vision model for image descriptions (e.g., llava:latest)
    OLLAMA_EMBEDDING_MODEL: Text embedding model (e.g., llama3.1:latest)
    OLLAMA_EMBEDDING_DIMENSION: Vector dimension (must match model output)
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for VisionAI application.

    All settings can be overridden via environment variables or .env file.
    Use UPPERCASE_WITH_UNDERSCORES for environment variable names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================================
    # Database Configuration
    # ==========================================================================

    # PostgreSQL connection string (requires pgvector extension)
    visionai_database_url: str = (
        "postgresql+psycopg://postgres:1337@localhost:5432/visionai_db"
    )

    # ==========================================================================
    # File Storage Configuration
    # ==========================================================================

    # Directory where uploaded images are stored
    upload_dir: Path = Path("uploads")

    # ==========================================================================
    # Ollama API Configuration
    # ==========================================================================

    # Base URL for Ollama API server
    ollama_base_url: str = "http://localhost:11434"

    # Optional API key (leave empty if not required)
    ollama_api_key: str = ""

    # Vision model for generating image descriptions
    ollama_model: str = "llava:latest"

    # Text embedding model for semantic search
    ollama_embedding_model: str = "llama3.1:latest"

    # Dimension of embedding vectors (must match model output)
    # Common values: 4096 (llama), 768 (sentence-transformers)
    ollama_embedding_dimension: int = 4096


# Singleton settings instance used throughout the application
settings = Settings()