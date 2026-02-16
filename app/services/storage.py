"""
Image file storage utilities.

This module handles all file system operations for uploaded images:
- Validating image files (type, size, content)
- Saving images to disk with unique filenames
- Deleting images when removed from the system

All images are stored in the configured upload directory with UUID-based
filenames to prevent collisions and path traversal attacks.
"""

import uuid
from pathlib import Path

from app.config import settings

# =============================================================================
# Constants
# =============================================================================

# Supported image MIME types for upload
ALLOWED_CONTENT_TYPES: set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

# Maximum allowed file size (10 MB)
MAX_FILE_SIZE: int = 10 * 1024 * 1024

# Default file extension when original cannot be determined
DEFAULT_EXTENSION: str = ".jpg"


# =============================================================================
# Directory Management
# =============================================================================

def get_upload_dir() -> Path:
    """
    Get the upload directory path, creating it if necessary.

    Returns:
        Path: Absolute path to the upload directory
    """
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


# =============================================================================
# Validation
# =============================================================================

def validate_image(content: bytes, content_type: str | None) -> None:
    """
    Validate uploaded image content before saving.

    Performs the following checks:
    1. Content type is in the allowed list
    2. File size is within the limit
    3. File is not empty

    Args:
        content: Raw image bytes
        content_type: MIME type from the upload request

    Raises:
        ValueError: If any validation check fails
    """
    # Check content type
    if content_type not in ALLOWED_CONTENT_TYPES:
        allowed_types = ", ".join(sorted(ALLOWED_CONTENT_TYPES))
        raise ValueError(
            f"Unsupported file type: {content_type}. Allowed: {allowed_types}"
        )

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE // (1024 * 1024)
        raise ValueError(
            f"File too large: {len(content):,} bytes. Maximum: {max_mb} MB"
        )

    # Check for empty file
    if len(content) == 0:
        raise ValueError("Empty file uploaded")


# =============================================================================
# File Operations
# =============================================================================

def save_to_disk(content: bytes, original_filename: str | None) -> tuple[str, str, int]:
    """
    Save image bytes to disk with a unique filename.

    Uses UUID-based naming to prevent filename collisions and ensure
    security against path traversal attacks.

    Args:
        content: Raw image bytes to save
        original_filename: Original filename (used to preserve extension)

    Returns:
        tuple: (stored_filename, filepath, file_size)
            - stored_filename: UUID-based filename with extension
            - filepath: Full path to the saved file
            - file_size: Size of the saved file in bytes
    """
    # Preserve original file extension or use default
    extension = DEFAULT_EXTENSION
    if original_filename:
        original_ext = Path(original_filename).suffix.lower()
        if original_ext:
            extension = original_ext

    # Generate unique filename using UUID
    stored_filename = f"{uuid.uuid4().hex}{extension}"

    # Save to upload directory
    upload_dir = get_upload_dir()
    filepath = upload_dir / stored_filename
    filepath.write_bytes(content)

    return stored_filename, str(filepath), len(content)


def delete_file(filepath: str) -> bool:
    """
    Delete an image file from disk.

    Safely handles cases where the file doesn't exist.

    Args:
        filepath: Path to the file to delete

    Returns:
        bool: True if file was deleted, False if it didn't exist
    """
    path = Path(filepath)
    if path.exists():
        path.unlink()
        return True
    return False
