"""
Shared HTTP client utilities for Ollama API communication.

This module provides a robust HTTP client for communicating with the
Ollama API server. It includes:

- Automatic retry logic with exponential backoff
- Configurable timeouts
- Optional authentication via API key
- Consistent error handling and logging

All AI services (vision, embedding) use these utilities for API calls.
"""

import logging
import time
from typing import Any

import requests

from app.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Maximum seconds to wait for API response
TIMEOUT: int = 120

# Number of retry attempts on transient failures
MAX_RETRIES: int = 3

# Seconds to wait between retry attempts
RETRY_DELAY: int = 5


# =============================================================================
# HTTP Utilities
# =============================================================================

def get_headers() -> dict[str, str]:
    """
    Build HTTP headers for Ollama API requests.

    Includes authorization header if an API key is configured.

    Returns:
        dict: Headers dictionary with Content-Type and optional Authorization
    """
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
    Make a POST request to Ollama API with automatic retry on failure.

    Implements retry logic for transient errors (timeouts, connection errors).
    Non-transient errors (4xx, 5xx) are raised immediately.

    Args:
        endpoint: API endpoint path (e.g., "/api/generate")
        payload: JSON payload to send in request body
        error_context: Human-readable context for error messages
                      (e.g., "Vision", "Embedding")

    Returns:
        dict: JSON response parsed as dictionary

    Raises:
        requests.HTTPError: If server returns error status code
        requests.RequestException: If request fails after all retries
        RuntimeError: If all retry attempts are exhausted

    Example:
        response = post_with_retry(
            "/api/embeddings",
            {"model": "llama3.1", "prompt": "Hello"},
            "Embedding"
        )
    """
    url = f"{settings.ollama_base_url}{endpoint}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=get_headers(),
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            return response.json()

        except (requests.Timeout, requests.ConnectionError) as e:
            # Transient error - retry if attempts remaining
            if attempt == MAX_RETRIES - 1:
                logger.error(
                    "%s API failed after %d attempts: %s",
                    error_context, MAX_RETRIES, e
                )
                raise

            logger.warning(
                "%s API attempt %d/%d failed: %s. Retrying in %ds...",
                error_context, attempt + 1, MAX_RETRIES, e, RETRY_DELAY
            )
            time.sleep(RETRY_DELAY)

    # This should not be reached, but provides a safety net
    raise RuntimeError(f"{error_context} API failed after {MAX_RETRIES} retries")
