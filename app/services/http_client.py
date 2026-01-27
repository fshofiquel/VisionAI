"""
Shared HTTP client utilities for Ollama API services.
Provides common retry logic, timeouts, and header configuration.
"""

import logging
import time
from typing import Any

import requests

from app.config import settings

logger = logging.getLogger(__name__)

# Request configuration
TIMEOUT = 120  # Maximum seconds to wait for API response
MAX_RETRIES = 3  # Number of retry attempts on failure
RETRY_DELAY = 5  # Seconds to wait between retries


def get_headers() -> dict[str, str]:
    """Build request headers, including auth if API key is configured."""
    headers = {"Content-Type": "application/json"}
    if settings.ollama_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
    return headers


def post_with_retry(
    endpoint: str,
    payload: dict[str, Any],
    error_context: str = "API",
) -> dict[str, Any]:
    """
    Make a POST request to Ollama API with retry logic.

    Args:
        endpoint: API endpoint path (e.g., "/api/generate")
        payload: JSON payload to send
        error_context: Context string for error messages (e.g., "Vision", "Embedding")

    Returns:
        JSON response as a dictionary

    Raises:
        requests.RequestException: If request fails after all retries
        RuntimeError: If all retries are exhausted
    """
    url = f"{settings.ollama_base_url}{endpoint}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=get_headers(),
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.error("%s API failed after %d attempts: %s", error_context, MAX_RETRIES, e)
                raise
            logger.warning(
                "%s API attempt %d/%d failed: %s, retrying in %ds...",
                error_context, attempt + 1, MAX_RETRIES, e, RETRY_DELAY
            )
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"{error_context} API failed after {MAX_RETRIES} retries")
