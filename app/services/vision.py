"""
Vision model service for generating image descriptions.
Supports multimodal models like LLaVA and Qwen2.5-VL via Ollama API.
"""

import base64
import io
import time

import requests
from PIL import Image

from app.config import settings

# Request configuration
TIMEOUT = 120  # Maximum seconds to wait for API response
MAX_RETRIES = 3  # Number of retry attempts on failure
RETRY_DELAY = 5  # Seconds to wait between retries

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

    @property
    def _headers(self) -> dict[str, str]:
        """Build request headers, including auth if API key is configured."""
        headers = {"Content-Type": "application/json"}
        if settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
        return headers

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
        # Resize image to speed up processing (maintains aspect ratio)
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

        # Convert to base64 for API transmission
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Build API request payload
        payload = {
            "model": settings.ollama_model,
            "prompt": DESCRIPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "keep_alive": "30m",  # Keep model loaded between requests
            "options": {
                "num_predict": 60,  # Limit response length
                "temperature": 0.0,  # Deterministic output
                "num_ctx": 2048,  # Reduced context window for speed
            },
        }

        # Retry loop for resilience
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json=payload,
                    headers=self._headers,
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()["response"].strip()
            except (requests.Timeout, requests.ConnectionError):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY)


# Singleton instance used throughout the application
vision_service = VisionService()
