"""
Database configuration and SQLAlchemy setup.
Uses PostgreSQL with pgvector extension for vector similarity search.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# Create database engine with connection pooling
engine = create_engine(
    settings.visionai_database_url,
    echo=False,  # Set True to log SQL queries for debugging
    pool_size=5,  # Number of persistent connections
    max_overflow=10,  # Additional connections allowed during high load
)

# Session factory for creating database sessions
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass