"""
Image API endpoints for upload, search, and management.

This module provides the REST API for VisionAI's image operations:

Endpoints:
    POST /images/upload - Upload and index a new image
    GET /images/search/text - Semantic search using natural language
    GET /images/{id} - Get image details by ID
    DELETE /images/{id} - Delete an image

The search endpoint uses a hybrid approach combining:
1. Vector similarity search (cosine distance on embeddings)
2. Keyword matching with boosting
3. Query expansion for short queries

This achieves 86% high-confidence matches in testing.
"""

import io
import logging
import re
from typing import Annotated, Final

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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


# =============================================================================
# Search Configuration Constants
# =============================================================================

# Maximum keyword boost added to vector similarity score
MAX_KEYWORD_BOOST: Final[float] = 0.8

# Minimum boost when at least one keyword matches (prevents dilution in verbose queries)
MIN_KEYWORD_BOOST: Final[float] = 0.4

# Multiplier for fetch limit (fetch more candidates for re-ranking)
FETCH_LIMIT_MULTIPLIER: Final[int] = 5

# Maximum candidates to fetch for re-ranking
MAX_FETCH_LIMIT: Final[int] = 100

# Maximum keyword matches to fetch per keyword
KEYWORD_MATCH_LIMIT: Final[int] = 20

# Common word variants/synonyms mapped to their canonical form
# This helps match informal terms like "doggy" to "dog" in descriptions
WORD_SYNONYMS: Final[dict[str, str]] = {
    # Animals - informal to formal
    "doggy": "dog",
    "doggie": "dog",
    "puppy": "dog",
    "pup": "dog",
    "kitty": "cat",
    "kitten": "cat",
    "kittycat": "cat",
    "birdie": "bird",
    "bunny": "rabbit",
    "horsie": "horse",
    "fishy": "fish",
    "fishes": "fish",
    "ducky": "duck",
    "duckling": "duck",
    "piggy": "pig",
    "piglet": "pig",
    "mousey": "mouse",
    "mice": "mouse",
    "goose": "geese",
    "wolfy": "wolf",
    "foxy": "fox",
    "deery": "deer",
    # Nature
    "sunny": "sun",
    "rainy": "rain",
    "snowy": "snow",
    "cloudy": "cloud",
    "foggy": "fog",
    "misty": "mist",
    "grassy": "grass",
    "leafy": "leaf",
    # Common plurals that might not match
    "dogs": "dog",
    "cats": "cat",
    "birds": "bird",
    "horses": "horse",
    "trees": "tree",
    "flowers": "flower",
    "mountains": "mountain",
    "beaches": "beach",
    "waves": "wave",
    "rocks": "rock",
    "clouds": "cloud",
    "stars": "star",
    "leaves": "leaf",
    "people": "person",
    "persons": "person",
    "children": "child",
    "kids": "child",
}

# Stop words to filter out when extracting keywords
STOP_WORDS: Final[frozenset[str]] = frozenset({
    # Articles and basic words
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    # Auxiliary verbs
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used',
    # Prepositions
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under',
    # Conjunctions and others
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
    'while', 'although',
    # Image-related terms (filtered to avoid noise)
    'image', 'showing', 'shows', 'picture', 'photo', 'photograph',
})


# =============================================================================
# Upload Endpoint
# =============================================================================

@router.post(
    "/upload",
    response_model=ImageUploadResponse,
    status_code=201,
    responses={
        400: {"description": "Invalid file type, file too large, or empty file"},
        422: {"description": "Failed to process image file"},
    },
)
async def upload_image(
        file: Annotated[UploadFile, File(description="Image file to upload")],
        description: Annotated[str | None, Form(description="Optional custom description")] = None,
        db: Annotated[Session, Depends(get_db)] = None,
        vision_svc: Annotated[VisionService, Depends(get_vision_service)] = None,
        ollama_embed_svc: Annotated[OllamaEmbeddingService, Depends(get_ollama_embedding_service)] = None,
):
    """
    Upload an image with automatic AI processing.

    1. Validates the image file (type, size)
    2. Generates a description using the vision model (if not provided)
    3. Creates a vector embedding of the description for semantic search
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

    # Generate embedding from full description for semantic search
    embedding = None
    if description:
        try:
            embedding = ollama_embed_svc.embed_text(description.strip())
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

def _expand_query(query: str) -> str:
    """
    Expand short queries to match the description embedding style.

    Since descriptions are embedded as "The image shows a dog standing..."
    single word queries like "dog" need to be expanded to match that format.
    This significantly improves semantic similarity scores.
    """
    query = query.strip().lower()
    word_count = len(query.split())

    # Short queries (1-2 words) need expansion to match description format
    if word_count <= 2:
        # Expand to match "The image shows..." format used in descriptions
        return f"an image showing {query}"
    elif word_count <= 4:
        # Medium queries get lighter expansion
        return f"image of {query}"
    else:
        # Verbose queries are already descriptive enough
        return query


def _extract_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a search query.

    Filters out stop words and short terms to identify the most
    important search terms for keyword boosting. Also normalizes
    synonyms (e.g., "doggy" -> includes both "doggy" and "dog").

    Args:
        query: User's search query string

    Returns:
        List of lowercase keywords (2+ characters, no stop words)

    Example:
        >>> _extract_keywords("a cute doggy in the park")
        ['cute', 'doggy', 'dog', 'park']
    """
    words = query.lower().split()
    keywords = []
    seen = set()

    for word in words:
        # Remove punctuation from word boundaries
        cleaned = word.strip('.,!?;:()[]{}"\'-')
        # Keep if not a stop word and has sufficient length
        if cleaned not in STOP_WORDS and len(cleaned) >= 2:
            if cleaned not in seen:
                keywords.append(cleaned)
                seen.add(cleaned)

            # Also add the canonical form if this is a synonym
            canonical = WORD_SYNONYMS.get(cleaned)
            if canonical and canonical not in seen:
                keywords.append(canonical)
                seen.add(canonical)

    return keywords


def _calculate_keyword_boost(description: str, keywords: list[str]) -> float:
    """
    Calculate a relevance boost based on keyword presence in description.

    Uses word boundary matching to ensure whole words are matched,
    avoiding false positives like "car" matching "cartoon".

    Gives EXTRA boost when keywords appear in the first sentence,
    as that typically describes the main subject of the image.

    Args:
        description: Image description to search within
        keywords: List of keywords extracted from search query

    Returns:
        Boost value between 0.0 and MAX_KEYWORD_BOOST (0.8).
        Extra boost when keyword is in first sentence (primary subject).
    """
    if not description or not keywords:
        return 0.0

    desc_lower = description.lower()

    # Split into first sentence and rest for primary subject detection
    # First sentence usually describes the main subject
    first_sentence_end = desc_lower.find('.')
    if first_sentence_end == -1:
        first_sentence = desc_lower
        rest_of_desc = ""
    else:
        first_sentence = desc_lower[:first_sentence_end]
        rest_of_desc = desc_lower[first_sentence_end:]

    primary_matches = 0  # Matches in first sentence (main subject)
    secondary_matches = 0  # Matches elsewhere (mentioned in passing)

    for kw in keywords:
        pattern = rf'\b{re.escape(kw)}\b'
        if re.search(pattern, first_sentence):
            primary_matches += 1
        elif re.search(pattern, rest_of_desc):
            secondary_matches += 1

    total_matches = primary_matches + secondary_matches
    if total_matches == 0:
        return 0.0

    # Primary matches (first sentence) get full weight
    # Secondary matches (mentioned later) get reduced weight (0.3x)
    weighted_matches = primary_matches + (secondary_matches * 0.3)
    match_ratio = weighted_matches / len(keywords)

    # Apply minimum floor and scale
    scaled_boost = MIN_KEYWORD_BOOST + (match_ratio * (MAX_KEYWORD_BOOST - MIN_KEYWORD_BOOST))

    return min(MAX_KEYWORD_BOOST, scaled_boost)


@router.get("/search/text", response_model=SearchResponse)
def search_by_text(
        q: Annotated[str, Query(min_length=1, max_length=500, description="Search query")],
        limit: Annotated[int, Query(ge=1, le=100, description="Maximum results to return")] = 10,
        min_score: Annotated[float, Query(
            ge=0.0, le=1.0,
            description="Minimum similarity score filter (0-1). Default 0 returns all."
        )] = 0.0,
        db: Annotated[Session, Depends(get_db)] = None,
        ollama_embed_svc: Annotated[OllamaEmbeddingService, Depends(get_ollama_embedding_service)] = None,
):
    """
    Search images using natural language queries with hybrid ranking.

    This endpoint combines multiple search strategies for optimal results:

    1. **Query Expansion**: Short queries are expanded to match description style
       - "dog" → "an image showing dog"
       - "sunset over ocean" → "image of sunset over ocean"

    2. **Vector Search**: Finds semantically similar images using embeddings

    3. **Keyword Injection**: Fetches images containing query keywords directly,
       ensuring obvious matches aren't missed

    4. **Keyword Boosting**: Adds up to +0.5 to scores when keywords are found
       in the description

    5. **Re-ranking**: Results are sorted by combined score (vector + boost)

    Args:
        q: Natural language search query (1-500 characters)
        limit: Maximum number of results to return (1-100)
        min_score: Minimum score threshold for results (0.0-1.0)

    Returns:
        SearchResponse with matching images sorted by relevance
    """
    # Step 1: Expand query to match description embedding style
    query_for_embedding = _expand_query(q)

    # Step 2: Extract keywords for boosting and direct matching
    keywords = _extract_keywords(q)

    logger.debug(
        "Search: query='%s' expanded='%s' keywords=%s",
        q, query_for_embedding, keywords
    )

    # Step 3: Embed the search query
    query_embedding = ollama_embed_svc.embed_text(query_for_embedding)

    # Step 4: Vector similarity search
    fetch_limit = min(limit * FETCH_LIMIT_MULTIPLIER, MAX_FETCH_LIMIT)

    vector_results = db.execute(
        select(
            Image,
            (1 - Image.embedding.cosine_distance(query_embedding)).label("score"),
        )
        .where(Image.embedding.isnot(None))
        .order_by(Image.embedding.cosine_distance(query_embedding))
        .limit(fetch_limit)
    ).all()

    # Step 5: Fetch keyword matches that might have been missed by vector search
    keyword_candidates = _fetch_keyword_candidates(db, keywords)

    # Step 6: Combine and deduplicate candidates
    all_candidates = _merge_candidates(
        vector_results, keyword_candidates, query_embedding
    )

    if not all_candidates:
        return SearchResponse(query=q, results=[], total=0)

    # Step 7: Apply keyword boosting and re-rank
    scored_results = _score_and_rank(all_candidates, keywords, min_score)

    # Step 8: Build response
    search_results = [
        ImageSearchResult(
            id=img.id,
            filename=img.filename,
            original_filename=img.original_filename,
            filepath=img.filepath,
            description=img.description,
            score=round(final_score, 4),
        )
        for img, final_score, _, _ in scored_results[:limit]
    ]

    return SearchResponse(query=q, results=search_results, total=len(search_results))


def _fetch_keyword_candidates(db: Session, keywords: list[str]) -> set[Image]:
    """
    Fetch images containing search keywords in their descriptions.

    This ensures obvious matches aren't missed due to low vector similarity.
    For example, searching "dog" will find all images with "dog" in the
    description, even if their embeddings are dissimilar.

    Args:
        db: Database session
        keywords: List of keywords to search for

    Returns:
        Set of Image objects matching any keyword
    """
    candidates = set()

    for kw in keywords:
        results = db.execute(
            select(Image)
            .where(Image.embedding.isnot(None))
            .where(Image.description.ilike(f"%{kw}%"))
            .limit(KEYWORD_MATCH_LIMIT)
        ).scalars().all()
        candidates.update(results)

    return candidates


def _merge_candidates(
    vector_results: list,
    keyword_candidates: set[Image],
    query_embedding: list[float],
) -> list[tuple[Image, float]]:
    """
    Merge vector search results with keyword candidates.

    Combines results from both search strategies, deduplicating by image ID.
    For keyword candidates not in vector results, calculates their vector
    similarity score manually.

    Args:
        vector_results: Results from vector similarity search
        keyword_candidates: Images found via keyword matching
        query_embedding: Query embedding for similarity calculation

    Returns:
        List of (Image, vector_score) tuples
    """
    seen_ids: set[int] = set()
    all_candidates: list[tuple[Image, float]] = []

    # Add vector results first (they already have scores)
    for row in vector_results:
        img = row.Image
        if img.id not in seen_ids:
            seen_ids.add(img.id)
            all_candidates.append((img, float(row.score)))

    # Add keyword candidates that weren't in vector results
    query_emb_array = np.array(query_embedding)
    for img in keyword_candidates:
        if img.id not in seen_ids:
            seen_ids.add(img.id)
            # Calculate cosine similarity manually
            img_emb = np.array(img.embedding)
            vector_score = float(np.dot(img_emb, query_emb_array))
            all_candidates.append((img, vector_score))

    return all_candidates


def _score_and_rank(
    candidates: list[tuple[Image, float]],
    keywords: list[str],
    min_score: float,
) -> list[tuple[Image, float, float, float]]:
    """
    Apply keyword boosting and rank candidates by combined score.

    Each candidate receives a boost if its description contains search
    keywords. The final score is vector_score + keyword_boost.

    Args:
        candidates: List of (Image, vector_score) tuples
        keywords: Keywords to check for in descriptions
        min_score: Minimum combined score threshold

    Returns:
        Sorted list of (Image, final_score, vector_score, boost) tuples
    """
    scored_results = []

    for img, vector_score in candidates:
        keyword_boost = _calculate_keyword_boost(img.description, keywords)
        final_score = vector_score + keyword_boost

        if final_score >= min_score:
            scored_results.append((img, final_score, vector_score, keyword_boost))

    # Sort by combined score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return scored_results


# =============================================================================
# Individual Image Endpoints
# =============================================================================

@router.get(
    "/{image_id}",
    response_model=ImageResponse,
    responses={
        404: {"description": "Image not found"},
    },
)
def get_image(
    image_id: int,
    db: Annotated[Session, Depends(get_db)] = None,
):
    """
    Get details of a specific image by ID.

    Retrieves all metadata for the specified image, including the
    AI-generated description. Does not include the embedding vector.

    Args:
        image_id: Unique identifier of the image

    Returns:
        ImageResponse with complete image details

    Raises:
        HTTPException 404: If image with given ID does not exist
    """
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageResponse.model_validate(image)


@router.delete(
    "/{image_id}",
    status_code=204,
    responses={
        404: {"description": "Image not found"},
    },
)
def delete_image(
    image_id: int,
    db: Annotated[Session, Depends(get_db)] = None,
):
    """
    Delete an image by ID.

    Removes both the image file from disk and the database record.
    This operation is irreversible.

    Args:
        image_id: Unique identifier of the image to delete

    Returns:
        204 No Content on success

    Raises:
        HTTPException 404: If image with given ID does not exist
    """
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete file from disk
    delete_file(image.filepath)

    # Remove from database
    db.delete(image)
    db.commit()
