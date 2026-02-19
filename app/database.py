"""
Database configuration and SQLAlchemy setup.

This module configures the PostgreSQL database connection with pgvector
extension support for vector similarity search. It provides:

- Database engine with connection pooling
- Session factory for creating database sessions
- Base class for ORM models

Requirements:
    - PostgreSQL 12+ with pgvector extension installed
    - CREATE EXTENSION IF NOT EXISTS vector;
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# =============================================================================
# Database Engine Configuration
# =============================================================================

# Connection pool settings
_POOL_SIZE = 5  # Number of persistent connections to maintain
_MAX_OVERFLOW = 10  # Additional connections allowed during high load

engine = create_engine(
    settings.visionai_database_url,
    echo=False,  # Set True to log SQL queries for debugging
    pool_size=_POOL_SIZE,
    max_overflow=_MAX_OVERFLOW,
)

# =============================================================================
# Session Factory
# =============================================================================

# Creates new database sessions for each request
# autocommit=False: Requires explicit commit() calls
# autoflush=False: Prevents automatic flush before queries
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# =============================================================================
# ORM Base Class
# =============================================================================

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    All database models should inherit from this class to be properly
    registered with SQLAlchemy's metadata system.
    """