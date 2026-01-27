"""
Image API endpoints for upload, search, and management.
Handles image processing with AI-generated descriptions and semantic search.
"""

import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_vision_service, get_ollama_embedding_service
from app.models.db_models import Image
from app.schemas.api_schemas import (
    ImageResponse,
    ImageSearchResult,
    ImageUploadResponse,
    SearchResponse,
)
from app.services.storage import delete_file, save_to_disk, validate_image
from app.services.vision import VisionService
from app.services.ollama_embedding import OllamaEmbeddingService

router = APIRouter(prefix="/images", tags=["images"])


# =============================================================================
# Upload Endpoint
# =============================================================================

@router.post("/upload", response_model=ImageUploadResponse, status_code=201)
async def upload_image(
        file: UploadFile = File(...),
        description: str | None = Form(default=None),
        db: Session = Depends(get_db),
        vision_svc: VisionService = Depends(get_vision_service),
        ollama_embed_svc: OllamaEmbeddingService = Depends(get_ollama_embedding_service),
):
    """
    Upload an image with automatic AI processing.

    1. Validates the image file (type, size)
    2. Generates a description using the vision model (if not provided)
    3. Creates a vector embedding of the description for search
    4. Saves the file and metadata to the database
    """
    content = await file.read()

    # Validate file type and size
    try:
        validate_image(content, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert to PIL Image for vision model
    try:
        pil_image = PILImage.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process image: {e}")

    # Generate description with vision model if not provided
    if description is None:
        try:
            description = vision_svc.describe_image(pil_image)
        except Exception:
            pass  # Continue without description if vision model fails

    # Generate embedding from description for semantic search
    embedding = None
    if description:
        try:
            embedding = ollama_embed_svc.embed_text(description)
        except Exception:
            pass  # Continue without embedding if it fails

    # Save file to disk
    stored_filename, filepath, file_size = save_to_disk(
        content, file.filename
    )

    # Create database record
    db_image = Image(
        filename=stored_filename,
        original_filename=file.filename or "unknown",
        filepath=filepath,
        content_type=file.content_type or "image/jpeg",
        file_size=file_size,
        description=description,
        embedding=embedding,
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    return ImageUploadResponse(
        message="Image uploaded and indexed successfully.",
        image=ImageResponse.model_validate(db_image),
    )


# =============================================================================
# Search Endpoint
# =============================================================================

@router.get("/search/text", response_model=SearchResponse)
def search_by_text(
        q: str = Query(..., min_length=1, max_length=500),
        limit: int = Query(default=10, ge=1, le=100),
        min_score: float = Query(default=0.0, ge=0.0, le=1.0),
        db: Session = Depends(get_db),
        ollama_embed_svc: OllamaEmbeddingService = Depends(get_ollama_embedding_service),
):
    """
    Search images using natural language queries.

    The query text is embedded using llama3.1 and compared against
    stored description embeddings using cosine similarity.
    Higher scores indicate more relevant matches.
    """
    # Embed the search query
    query_embedding = ollama_embed_svc.embed_text(q)

    # Find similar images using pgvector cosine distance
    results = db.execute(
        select(
            Image,
            (1 - Image.embedding.cosine_distance(query_embedding)).label("score"),
        )
        .where(Image.embedding.isnot(None))
        .order_by(Image.embedding.cosine_distance(query_embedding))
        .limit(limit)
    ).all()

    # Filter by minimum score and format results
    search_results = [
        ImageSearchResult(
            id=row.Image.id,
            filename=row.Image.filename,
            original_filename=row.Image.original_filename,
            filepath=row.Image.filepath,
            description=row.Image.description,
            score=round(float(row.score), 4),
        )
        for row in results
        if float(row.score) >= min_score
    ]

    return SearchResponse(query=q, results=search_results, total=len(search_results))


# =============================================================================
# Individual Image Endpoints
# =============================================================================

@router.get("/{image_id}", response_model=ImageResponse)
def get_image(image_id: int, db: Session = Depends(get_db)):
    """Get details of a specific image by ID."""
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageResponse.model_validate(image)


@router.delete("/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Delete an image by ID (removes file and database record)."""
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete file from disk
    delete_file(image.filepath)

    # Remove from database
    db.delete(image)
    db.commit()
