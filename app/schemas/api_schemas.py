"""
Pydantic schemas for API request/response validation.
Defines the shape of data returned by image endpoints.
"""

import datetime

from pydantic import BaseModel, ConfigDict


class ImageResponse(BaseModel):
    """Full image details returned after upload or when fetching by ID."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    original_filename: str
    filepath: str
    content_type: str
    file_size: int
    description: str | None
    created_at: datetime.datetime


class ImageSearchResult(BaseModel):
    """Single image result from a search query, includes similarity score."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    original_filename: str
    filepath: str
    description: str | None
    search_keywords: str | None = None  # Keywords that were matched
    score: float  # Cosine similarity score (0-1, higher = more similar)


class ImageUploadResponse(BaseModel):
    """Response returned after successfully uploading an image."""

    message: str
    image: ImageResponse


class SearchResponse(BaseModel):
    """Response containing search results with the original query."""

    query: str | None = None
    results: list[ImageSearchResult]
    total: int
