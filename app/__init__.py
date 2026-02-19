"""
FastAPI application factory and configuration.
Sets up the API with routes, static file serving, and database initialization.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import Base, engine
from app.models import Image  # noqa: F401 - Required for table creation
from app.routers.api import router as images_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs setup code on startup (create tables, directories) and cleanup on shutdown.
    """
    # Startup: Create database tables and upload directory
    Base.metadata.create_all(bind=engine)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: Add cleanup code here if needed


def create_app() -> FastAPI:
    """
    Application factory - creates and configures the FastAPI app.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="VisionAI",
        description="AI-powered image search using LLaVA descriptions and Ollama embeddings",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Serve uploaded images as static files
    app.mount(
        "/static/uploads",
        StaticFiles(directory=str(settings.upload_dir)),
        name="uploads",
    )

    # Include API routes
    app.include_router(images_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"message": "VisionAI API is running"}

    return app
