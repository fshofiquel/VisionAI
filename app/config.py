from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    visionai_database_url: str = "postgresql+psycopg://postgres:1337@localhost:5432/visionai_db"
    upload_dir: Path = Path("uploads")
    clip_model_name: str = "clip-ViT-B-32"
    embedding_dimension: int = 512

    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = ""
    ollama_model: str = "llava:latest"
    ollama_embedding_model: str = "llama3.1:latest"
    ollama_embedding_dimension: int = 4096


settings = Settings()