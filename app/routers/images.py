import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_embedding_service, get_llava_service
from app.models.image import Image
from app.schemas.image import (
    ImageResponse,
    ImageSearchResult,
    ImageUploadResponse,
    SearchResponse,
)
from app.services.embedding import EmbeddingService
from app.services.image_storage import delete_file, save_to_disk, validate_image
from app.services.llava import LLaVAService

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload", response_model=ImageUploadResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    db: Session = Depends(get_db),
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    llava_svc: LLaVAService = Depends(get_llava_service),
):
    content = await file.read()

    try:
        validate_image(content, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        pil_image = PILImage.open(io.BytesIO(content)).convert("RGB")
        embedding = embedding_svc.embed_image(pil_image)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process image: {e}")

    if description is None:
        try:
            description = llava_svc.describe_image(pil_image)
        except Exception:
            pass

    stored_filename, filepath, file_size = save_to_disk(
        content, file.filename, file.content_type
    )

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


@router.get("/search/text", response_model=SearchResponse)
def search_by_text(
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
):
    query_embedding = embedding_svc.embed_text(q)

    results = db.execute(
        select(
            Image,
            (1 - Image.embedding.cosine_distance(query_embedding)).label("score"),
        )
        .order_by(Image.embedding.cosine_distance(query_embedding))
        .limit(limit)
    ).all()

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
    ]

    return SearchResponse(query=q, results=search_results, total=len(search_results))


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
):
    try:
        content = await file.read()
        pil_image = PILImage.open(io.BytesIO(content)).convert("RGB")
        query_embedding = embedding_svc.embed_image(pil_image)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process image: {e}")

    results = db.execute(
        select(
            Image,
            (1 - Image.embedding.cosine_distance(query_embedding)).label("score"),
        )
        .order_by(Image.embedding.cosine_distance(query_embedding))
        .limit(limit)
    ).all()

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
    ]

    return SearchResponse(query="[image upload]", results=search_results, total=len(search_results))


@router.get("/{image_id}", response_model=ImageResponse)
def get_image(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageResponse.model_validate(image)


@router.delete("/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    delete_file(image.filepath)
    db.delete(image)
    db.commit()