"""
VisionAI - AI-powered semantic image search.

This is the main application entry point. The FastAPI application is
created using the factory pattern in app/__init__.py.

Usage:
    Development:
        uvicorn main:app --reload

    Production:
        uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

API Endpoints:
    POST /api/v1/images/upload     - Upload and index an image
    GET  /api/v1/images/search/text - Semantic search with natural language
    GET  /api/v1/images/{id}        - Get image details
    DELETE /api/v1/images/{id}      - Delete an image

Documentation:
    GET /docs    - Interactive Swagger UI
    GET /redoc   - ReDoc documentation
"""

from app import create_app

# Create the FastAPI application instance
app = create_app()
