from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import Base, engine
from app.models import Image  # noqa: F401
from app.routers.images import router as images_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="VisionAI",
        description="AI-powered image search using LLaVA descriptions and Ollama embeddings",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.mount(
        "/static/uploads",
        StaticFiles(directory=str(settings.upload_dir)),
        name="uploads",
    )

    app.include_router(images_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"message": "VisionAI API is running"}

    return app


app = create_app()
