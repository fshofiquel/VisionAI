import base64
import io
import time

import requests
from PIL import Image

from app.config import settings

TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_IMAGE_SIZE = 512


class LLaVAService:
    _instance: "LLaVAService | None" = None

    def __new__(cls) -> "LLaVAService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
        return headers

    def describe_image(self, image: Image.Image) -> str:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": settings.ollama_model,
            "prompt": "Briefly describe this image in one sentence.",
            "images": [image_b64],
            "stream": False,
            "options": {
                "num_predict": 80,
            },
        }

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


llava_service = LLaVAService()
