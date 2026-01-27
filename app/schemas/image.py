import datetime

from pydantic import BaseModel, ConfigDict


class ImageResponse(BaseModel):
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
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    original_filename: str
    filepath: str
    description: str | None
    score: float


class ImageUploadResponse(BaseModel):
    message: str
    image: ImageResponse


class SearchResponse(BaseModel):
    query: str | None = None
    results: list[ImageSearchResult]
    total: int
