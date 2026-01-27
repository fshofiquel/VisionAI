"""
Image API endpoints for upload, search, and management.
Handles image processing with AI-generated descriptions and semantic search.
"""

import io
import logging

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

logger = logging.getLogger(__name__)

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
        except Exception as e:
            logger.warning("Vision model failed for %s: %s", file.filename, e)

    # Generate embedding from description for semantic search
    embedding = None
    if description:
        try:
            embedding = ollama_embed_svc.embed_text(description)
        except Exception as e:
            logger.warning("Embedding failed for %s: %s", file.filename, e)

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
        min_score: float = Query(default=0.15, ge=0.0, le=1.0, description="Minimum similarity score (0-1). Default 0.15 filters low-relevance results."),
        db: Session = Depends(get_db),
        ollama_embed_svc: OllamaEmbeddingService = Depends(get_ollama_embedding_service),
):
    """
    Search images using natural language queries.

    The query text is embedded using LLaVA/CLIP and compared against
    stored description embeddings using cosine similarity.
    Higher scores indicate more relevant matches.
    Results are boosted if the query terms appear in the description.
    """
    # Embed the search query
    query_embedding = ollama_embed_svc.embed_text(q)
    query_lower = q.lower()

    # Find similar images using pgvector cosine distance
    results = db.execute(
        select(
            Image,
            (1 - Image.embedding.cosine_distance(query_embedding)).label("score"),
        )
        .where(Image.embedding.isnot(None))
        .order_by(Image.embedding.cosine_distance(query_embedding))
        .limit(limit * 2)  # Fetch extra for re-ranking
    ).all()

    # Calculate boosted scores (boost if query terms appear in description)
    scored_results = []
    for row in results:
        base_score = float(row.score)
        description_lower = (row.Image.description or "").lower()

        # Boost score by 0.1 if query appears in description
        boost = 0.1 if query_lower in description_lower else 0.0
        final_score = min(base_score + boost, 1.0)  # Cap at 1.0

        if final_score >= min_score:
            scored_results.append((row.Image, final_score))

    # Sort by boosted score and limit results
    scored_results.sort(key=lambda x: x[1], reverse=True)
    scored_results = scored_results[:limit]

    # Format results
    search_results = [
        ImageSearchResult(
            id=img.id,
            filename=img.filename,
            original_filename=img.original_filename,
            filepath=img.filepath,
            description=img.description,
            score=round(score, 4),
        )
        for img, score in scored_results
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
