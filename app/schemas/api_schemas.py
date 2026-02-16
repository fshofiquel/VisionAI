"""
Pydantic schemas for API request/response validation.

This module defines the data structures (schemas) for all API endpoints.
Pydantic handles:
- Request body validation
- Response serialization
- OpenAPI documentation generation

All schemas use ConfigDict(from_attributes=True) to enable automatic
conversion from SQLAlchemy ORM objects.
"""

import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Image Response Schemas
# =============================================================================

class ImageResponse(BaseModel):
    """
    Complete image details returned by upload and get endpoints.

    Includes all metadata about an image, excluding the embedding vector
    (which is too large for API responses).
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Unique image identifier")
    filename: str = Field(description="Stored filename (UUID-based)")
    original_filename: str = Field(description="User's original filename")
    filepath: str = Field(description="Server path to the image file")
    content_type: str = Field(description="MIME type (e.g., 'image/jpeg')")
    file_size: int = Field(description="File size in bytes")
    description: Optional[str] = Field(
        default=None,
        description="AI-generated description of the image content"
    )
    created_at: datetime.datetime = Field(description="Upload timestamp")


class ImageSearchResult(BaseModel):
    """
    Single image result from a semantic search query.

    Contains essential image info plus the similarity score indicating
    how well the image matches the search query.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Unique image identifier")
    filename: str = Field(description="Stored filename")
    original_filename: str = Field(description="User's original filename")
    filepath: str = Field(description="Server path to the image file")
    description: Optional[str] = Field(
        default=None,
        description="AI-generated description"
    )
    score: float = Field(
        description="Similarity score (0-1+). Higher values indicate "
                    "better matches. Scores above 1.0 indicate keyword "
                    "boost was applied."
    )


# =============================================================================
# Request/Response Wrappers
# =============================================================================

class ImageUploadResponse(BaseModel):
    """
    Response returned after successfully uploading an image.

    Contains a confirmation message and the complete image details.
    """

    message: str = Field(description="Success message")
    image: ImageResponse = Field(description="Uploaded image details")


class SearchResponse(BaseModel):
    """
    Response containing semantic search results.

    Includes the original query for reference and a list of matching
    images sorted by relevance (highest score first).
    """

    query: Optional[str] = Field(
        default=None,
        description="Original search query"
    )
    results: list[ImageSearchResult] = Field(
        description="List of matching images, sorted by score (descending)"
    )
    total: int = Field(description="Number of results returned")
