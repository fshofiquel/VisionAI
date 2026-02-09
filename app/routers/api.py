"""
Image API endpoints for upload, search, and management.
Handles image processing with AI-generated descriptions and semantic search.
"""

import io
import logging

import numpy as np
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
from app.services.keyword_extractor import KeywordExtractor

logger = logging.getLogger(__name__)

# Generic query used to compute baseline scores for images
# Images similar to this are "generic" and will be penalized in search
GENERIC_QUERY = "scene, photo, image, picture, view"

router = APIRouter(prefix="/images", tags=["images"])

# Singleton keyword extractor for query expansion
_keyword_extractor = KeywordExtractor()


def get_keyword_extractor() -> KeywordExtractor:
    """Dependency injection for keyword extractor service."""
    return _keyword_extractor


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
        keyword_svc: KeywordExtractor = Depends(get_keyword_extractor),
):
    """
    Upload an image with automatic AI processing.

    1. Validates the image file (type, size)
    2. Generates a description using the vision model (if not provided)
    3. Extracts search keywords from description using LLM
    4. Creates a vector embedding of keywords (not full description) for search
    5. Saves the file and metadata to the database
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

    # Extract keywords and generate embedding for semantic search
    embedding = None
    baseline_score = None
    search_keywords = None

    if description:
        try:
            # Extract keywords from description (short text for better embedding)
            search_keywords = keyword_svc.extract_keywords(description)

            # Embed the keywords (NOT the full description)
            # Short ↔ short comparison produces better similarity scores
            text_to_embed = search_keywords if search_keywords else description
            embedding = ollama_embed_svc.embed_text(text_to_embed)

            # Compute baseline score for search ranking normalization
            generic_embedding = ollama_embed_svc.embed_text(GENERIC_QUERY)
            baseline_score = float(np.dot(
                np.array(embedding),
                np.array(generic_embedding)
            ))
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
        search_keywords=search_keywords,
        embedding=embedding,
        baseline_score=baseline_score,
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
        min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score (0-1). Default 0 returns all results."),
        db: Session = Depends(get_db),
        ollama_embed_svc: OllamaEmbeddingService = Depends(get_ollama_embedding_service),
        keyword_svc: KeywordExtractor = Depends(get_keyword_extractor),
):
    """
    Search images using natural language queries (Pure AI with LLaVA).

    Short queries (1-2 words) are expanded by LLM to match keyword format.
    Example: "bunny" -> "bunny, rabbit, hare, fluffy animal, cute pet"
    This improves embedding similarity matching.
    """
    # Expand short queries to match keyword format
    query_for_embedding = keyword_svc.expand_query(q)

    # Embed the search query
    query_embedding = ollama_embed_svc.embed_text(query_for_embedding)

    # PURE AI SEARCH: Vector similarity with baseline normalization
    results = db.execute(
        select(
            Image,
            (1 - Image.embedding.cosine_distance(query_embedding)).label("score"),
        )
        .where(Image.embedding.isnot(None))
        .order_by(Image.embedding.cosine_distance(query_embedding))
        .limit(limit * 3)  # Get extra for filtering
    ).all()

    # Apply baseline normalization to prevent generic images from dominating
    # Images with high baseline scores (similar to generic queries) are penalized
    scored_results = []

    for row in results:
        raw_score = float(row.score)
        img = row.Image

        # Baseline normalization: subtract baseline score to get "specificity"
        # Images that match everything (high baseline) get penalized
        # Images that match only specific queries (low baseline) get boosted
        baseline = img.baseline_score if img.baseline_score else 0.0

        # Normalized score = raw_score - (baseline * penalty_factor)
        # penalty_factor of 0.5 means half the baseline is subtracted
        # This reduces score for generic images without completely removing them
        penalty_factor = 0.3
        normalized_score = raw_score - (baseline * penalty_factor)

        # Ensure score doesn't go negative
        final_score = max(0.0, normalized_score)

        if final_score >= min_score:
            scored_results.append((img, final_score))


    # Sort by score and limit results
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
            search_keywords=img.search_keywords,
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
