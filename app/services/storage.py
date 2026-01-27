"""
Image file storage utilities.
Handles saving, validating, and deleting image files on disk.
"""

import uuid
from pathlib import Path

from app.config import settings

# Supported image MIME types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

# Maximum file size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024


def get_upload_dir() -> Path:
    """Get the upload directory path, creating it if it doesn't exist."""
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def validate_image(content: bytes, content_type: str | None) -> None:
    """
    Validate uploaded image content.

    Raises:
        ValueError: If file type is unsupported, file is too large, or file is empty.
    """
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported file type: {content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
        )
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {len(content)} bytes. Maximum: {MAX_FILE_SIZE} bytes.")
    if len(content) == 0:
        raise ValueError("Empty file uploaded.")


def save_to_disk(content: bytes, original_filename: str | None) -> tuple[str, str, int]:
    """
    Save image bytes to disk with a unique filename.

    Args:
        content: Raw image bytes
        original_filename: Original filename (used to preserve extension)

    Returns:
        Tuple of (stored_filename, relative_filepath, file_size)
    """
    # Preserve original file extension or default to .jpg
    extension = Path(original_filename).suffix.lower() if original_filename else ".jpg"

    # Generate unique filename using UUID to prevent collisions
    stored_filename = f"{uuid.uuid4().hex}{extension}"

    upload_dir = get_upload_dir()
    filepath = upload_dir / stored_filename
    filepath.write_bytes(content)

    return stored_filename, str(filepath), len(content)


def delete_file(filepath: str) -> None:
    """Delete an image file from disk if it exists."""
    path = Path(filepath)
    if path.exists():
        path.unlink()
