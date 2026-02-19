# VisionAI

AI-powered semantic image search using vision-language models and vector embeddings.

## Overview

VisionAI is a semantic image search engine that allows you to find images using natural language queries. Upload images, and the system automatically generates descriptions using vision-language models (LLaVA), then indexes them for semantic search using vector embeddings.

**Key Features:**
- 🔍 **Natural Language Search** - Find images with queries like "sunset over mountains" or just "dog"
- 🤖 **AI-Powered Descriptions** - Automatic image captioning using LLaVA vision model
- ⚡ **Hybrid Search** - Combines vector similarity with keyword matching for 86% accuracy
- 🚀 **Fast & Scalable** - PostgreSQL with pgvector for efficient similarity search
- 📦 **Simple REST API** - Easy integration with any application
- 🎨 **Modern React Frontend** - Clean, responsive UI for search and upload

## Quick Start

### Prerequisites

- Python 3.14+
- Node.js 18+ (for frontend)
- PostgreSQL 12+ with pgvector extension
- Ollama with LLaVA model

### 1. Install Ollama & Models

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llava:latest      # Vision model for descriptions
ollama pull llama3.1:latest   # Text model for embeddings
```

### 2. Setup PostgreSQL with pgvector

```sql
-- Create database
CREATE DATABASE visionai_db;

-- Connect and enable pgvector
\c visionai_db
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Install Backend

#### Option A: Using pip (Recommended for PyCharm)

```bash
# Clone the repository
git clone https://github.com/yourusername/visionai.git
cd visionai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/Mac:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using uv (Faster)

```bash
# Clone the repository
git clone https://github.com/yourusername/visionai.git
cd visionai

# Install with uv (creates .venv automatically)
uv sync
```

### 4. Install Frontend

```bash
cd frontend
npm install
```

### 5. Configure Environment

Copy the example environment file and update with your settings:

```bash
# Linux/Mac
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` with your database credentials:

```env
VISIONAI_DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/visionai_db
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:latest
OLLAMA_EMBEDDING_MODEL=llama3.1:latest
OLLAMA_EMBEDDING_DIMENSION=4096
UPLOAD_DIR=uploads
```

### 6. Run the Application

**Terminal 1 - Backend:**
```bash
uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Access the application:**

| Component | URL |
|-----------|-----|
| Frontend UI | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |

## API Reference

### Upload Image

```http
POST /api/v1/images/upload
Content-Type: multipart/form-data
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | file | Image file (JPEG, PNG, WebP, GIF, BMP) |
| `description` | string | Optional custom description |

**Response:**
```json
{
  "message": "Image uploaded and indexed successfully.",
  "image": {
    "id": 1,
    "filename": "abc123.jpg",
    "original_filename": "photo.jpg",
    "description": "The image shows a golden retriever...",
    "created_at": "2026-02-19T10:30:00Z"
  }
}
```

### Search Images

```http
GET /api/v1/images/search/text?q={query}&limit={limit}&min_score={min_score}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | required | Search query (1-500 chars) |
| `limit` | int | 10 | Max results (1-100) |
| `min_score` | float | 0.0 | Minimum score filter (0-1) |

**Response:**
```json
{
  "query": "sunset over ocean",
  "results": [
    {
      "id": 42,
      "filename": "sunset.jpg",
      "description": "The image shows a beautiful sunset...",
      "score": 0.8521
    }
  ],
  "total": 1
}
```

### Get Image Details

```http
GET /api/v1/images/{image_id}
```

### Delete Image

```http
DELETE /api/v1/images/{image_id}
```

## Project Structure

```
VisionAI/
├── main.py                     # Backend entry point (uvicorn target)
├── pyproject.toml              # Python project metadata and dependencies
├── requirements.txt            # Pip-compatible dependencies
├── .env                        # Environment configuration (not in git)
├── .env.example                # Example environment template
├── ARCHITECTURE.md             # Detailed technical documentation
│
├── app/                        # Backend application package
│   ├── __init__.py             # FastAPI app factory and lifespan
│   ├── config.py               # Pydantic settings management
│   ├── database.py             # SQLAlchemy engine and session setup
│   ├── dependencies.py         # FastAPI dependency injection
│   │
│   ├── models/                 # Database models
│   │   ├── __init__.py         # Exports Image model
│   │   └── db_models.py        # SQLAlchemy ORM model with pgvector
│   │
│   ├── schemas/                # API data schemas
│   │   ├── __init__.py         # Exports all schemas
│   │   └── api_schemas.py      # Pydantic request/response models
│   │
│   ├── routers/                # API route handlers
│   │   ├── __init__.py         # Exports router
│   │   └── api.py              # Image upload, search, CRUD endpoints
│   │
│   └── services/               # Business logic services
│       ├── __init__.py         # Exports all services
│       ├── http_client.py      # Ollama API client with retry logic
│       ├── vision.py           # Image description generation (LLaVA)
│       ├── ollama_embedding.py # Text embedding service
│       └── storage.py          # File validation and storage
│
├── frontend/                   # React frontend application
│   ├── package.json            # Node.js dependencies and scripts
│   ├── vite.config.ts          # Vite configuration with API proxy
│   ├── tsconfig.json           # TypeScript configuration
│   │
│   ├── src/
│   │   ├── main.tsx            # React entry point
│   │   ├── App.tsx             # Main app component with navigation
│   │   ├── App.css             # Global styles
│   │   │
│   │   ├── api/
│   │   │   └── client.ts       # API client for backend communication
│   │   │
│   │   ├── components/
│   │   │   ├── ImageCard.tsx   # Image display card component
│   │   │   └── ImageCard.css
│   │   │
│   │   ├── pages/
│   │   │   ├── SearchPage.tsx  # Search interface
│   │   │   ├── SearchPage.css
│   │   │   ├── UploadPage.tsx  # Upload interface
│   │   │   └── UploadPage.css
│   │   │
│   │   └── test/               # Frontend tests
│   │       └── setup.ts
│   │
│   └── public/                 # Static assets
│
└── uploads/                    # Uploaded image storage (not in git)
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VISIONAI_DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg://postgres:1337@localhost:5432/visionai_db` |
| `UPLOAD_DIR` | Image storage directory | `uploads` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Vision model name | `llava:latest` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `llama3.1:latest` |
| `OLLAMA_EMBEDDING_DIMENSION` | Vector dimension | `4096` |

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed technical documentation covering system architecture, the hybrid search algorithm, and implementation details.
- **Swagger UI** - http://localhost:8000/docs (interactive API documentation)
- **ReDoc** - http://localhost:8000/redoc (alternative API documentation)

## License

MIT License

