"""
Vision model service for generating image descriptions.
Supports multimodal models like LLaVA and Qwen2.5-VL via Ollama API.
"""

import base64
import io
import re

from PIL import Image

from app.config import settings
from app.services.http_client import post_with_retry

# Image processing settings
MAX_IMAGE_SIZE = 384  # Resize images to this max dimension for faster processing

# Short prompt for speed - clean_description() handles any garbled output
DESCRIPTION_PROMPT = "What is in this image? Describe briefly."


def clean_description(text: str) -> str:
    """
    Clean garbled prefixes from vision model output.

    Some vision models produce artifacts like "feelThe image..." or "theThe image..."
    This function removes these artifacts to ensure clean descriptions.
    """
    if not text:
        return text

    cleaned = text.strip()

    # Remove any text before "The image" or "This image"
    # Pattern: random lowercase word(s) followed by "The image" or "This image"
    match = re.search(r'(The image|This image)', cleaned, re.IGNORECASE)
    if match:
        # Keep from "The image" onwards
        cleaned = cleaned[match.start():]
        # Ensure proper capitalization
        if cleaned.startswith('the '):
            cleaned = 'T' + cleaned[1:]

    # Remove leading punctuation or whitespace
    cleaned = re.sub(r'^[.,;:\s]+', '', cleaned)

    # Remove any remaining garbled prefix (lowercase followed by uppercase)
    match = re.match(r'^[a-z]+([A-Z].*)', cleaned)
    if match:
        cleaned = match.group(1)

    return cleaned.strip()


class VisionService:
    """
    Service for generating image descriptions using vision-language models.

    Uses a singleton pattern to reuse the same instance across requests.
    Images are resized before sending to improve processing speed.
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

        Args:
            image: PIL Image object to describe

        Returns:
            Text description of the image content

        Raises:
            requests.RequestException: If API call fails after all retries
        """
        # Resize image to speed up processing (copy to avoid mutating original)
        img_copy = image.copy()
        img_copy.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

        # Convert to base64 for API transmission
        buf = io.BytesIO()
        img_copy.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Build API request payload
        payload = {
            "model": settings.ollama_model,
            "prompt": DESCRIPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": 60,
            },
        }

        response = post_with_retry("/api/generate", payload, "Vision")
        result: str = response["response"].strip()

        # Clean any garbled prefixes from the output
        result = clean_description(result)

        return result


# Singleton instance used throughout the application
vision_service = VisionService()
