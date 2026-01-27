"""
Vision model service for generating image descriptions.
Supports multimodal models like LLaVA and Qwen2.5-VL via Ollama API.
"""

import base64
import io

from PIL import Image

from app.config import settings
from app.services.http_client import post_with_retry

# Image processing settings
MAX_IMAGE_SIZE = 384  # Resize images to this max dimension for faster processing

# Prompt designed for concise, searchable descriptions
DESCRIPTION_PROMPT = "What is in this image? Describe briefly."


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
                "temperature": 0.0,
                "num_ctx": 2048,
            },
        }

        response = post_with_retry("/api/generate", payload, "Vision")
        result: str = response["response"].strip()
        return result


# Singleton instance used throughout the application
vision_service = VisionService()
