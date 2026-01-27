"""
VisionAI - Main application entry point.
Run with: uvicorn main:app --reload
"""

from app import create_app

app = create_app()
