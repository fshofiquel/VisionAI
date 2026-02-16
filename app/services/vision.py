"""
Vision model service for generating image descriptions.

This module provides AI-powered image description generation using
multimodal vision-language models via the Ollama API. Supported models
include LLaVA and Qwen2.5-VL.

The service:
- Resizes images for faster processing
- Converts images to base64 for API transmission
- Cleans up any garbled output from the model
- Uses singleton pattern for efficient resource usage

Example:
    from PIL import Image
    service = VisionService()
    img = Image.open("photo.jpg")
    description = service.describe_image(img)
    # Returns: "The image shows a sunset over mountains..."
"""

import base64
import io
import re
from typing import Final

from PIL import Image

from app.config import settings
from app.services.http_client import post_with_retry

# =============================================================================
# Configuration Constants
# =============================================================================

# Maximum image dimension (images are resized to fit within this size)
# Smaller = faster processing, larger = more detail
MAX_IMAGE_SIZE: Final[int] = 384

# Prompt sent to the vision model
DESCRIPTION_PROMPT: Final[str] = "What is in this image? Describe briefly."

# JPEG quality for image encoding (balance between size and quality)
JPEG_QUALITY: Final[int] = 85


# =============================================================================
# Output Cleaning
# =============================================================================

def clean_description(text: str) -> str:
    """
    Clean garbled prefixes from vision model output.

    Some vision models (especially LLaVA) occasionally produce artifacts
    like "feelThe image..." or "theThe image..." at the start of their
    output. This function removes these artifacts.

    Args:
        text: Raw text output from vision model

    Returns:
        Cleaned text with proper capitalization

    Examples:
        >>> clean_description("feelThe image shows a dog")
        "The image shows a dog"
        >>> clean_description("thethe IMAGE shows")
        "The image shows"
    """
    if not text:
        return text

    cleaned = text.strip()

    # Find and extract from "The image" or "This image" onwards
    match = re.search(r'(The image|This image)', cleaned, re.IGNORECASE)
    if match:
        cleaned = cleaned[match.start():]
        # Ensure proper capitalization
        if cleaned.startswith('the '):
            cleaned = 'T' + cleaned[1:]

    # Remove leading punctuation or whitespace
    cleaned = re.sub(r'^[.,;:\s]+', '', cleaned)

    # Remove garbled prefix pattern: lowercase chars followed by uppercase
    # Example: "feelThe" -> "The"
    match = re.match(r'^[a-z]+([A-Z].*)', cleaned)
    if match:
        cleaned = match.group(1)

    return cleaned.strip()


# =============================================================================
# Vision Service
# =============================================================================

class VisionService:
    """
    Service for generating image descriptions using vision-language models.

    Uses multimodal models (LLaVA, Qwen2.5-VL) to analyze images and
    generate natural language descriptions. Images are resized before
    processing to improve speed.

    Features:
        - Singleton pattern: Only one instance exists application-wide
        - Automatic image resizing: Maintains aspect ratio
        - Output cleaning: Removes model artifacts

    Attributes:
        _instance: Class-level singleton instance

    Example:
        from PIL import Image

        service = VisionService()
        img = Image.open("sunset.jpg")
        desc = service.describe_image(img)
        print(desc)  # "The image shows a beautiful sunset..."
    """

    _instance: "VisionService | None" = None

    def __new__(cls) -> "VisionService":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def describe_image(self, image: Image.Image) -> str:
        """
        Generate a text description of the given image.

        The image is resized to MAX_IMAGE_SIZE while maintaining aspect
        ratio, then sent to the vision model for analysis.

        Args:
            image: PIL Image object to describe (any mode, will be
                   converted to RGB)

        Returns:
            Natural language description of the image content

        Raises:
            requests.RequestException: If API call fails after all retries
        """
        # Resize image to speed up processing (copy to avoid mutating original)
        img_copy = image.copy()
        img_copy.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

        # Ensure RGB mode for JPEG encoding
        if img_copy.mode != "RGB":
            img_copy = img_copy.convert("RGB")

        # Convert to base64 for API transmission
        buffer = io.BytesIO()
        img_copy.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Build API request payload
        payload = {
            "model": settings.ollama_model,
            "prompt": DESCRIPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "keep_alive": "30m",  # Keep model loaded for 30 minutes
            "options": {
                "num_predict": 60,  # Limit response length for speed
            },
        }

        # Call vision model API
        response = post_with_retry("/api/generate", payload, "Vision")
        result: str = response["response"].strip()

        # Clean any garbled prefixes from the output
        result = clean_description(result)

        return result


# Singleton instance used throughout the application
vision_service = VisionService()
