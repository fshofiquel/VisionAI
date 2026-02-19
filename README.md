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

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI App   │────▶│   Ollama API    │────▶│  LLaVA Model    │
│   (REST API)    │     │   (localhost)   │     │  (Vision + LLM) │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │
│   + pgvector    │
│  (Embeddings)   │
└─────────────────┘
```

### How It Works

1. **Upload**: Image is uploaded via REST API
2. **Describe**: LLaVA vision model generates a text description
3. **Embed**: Description is converted to a 4096-dimensional vector
4. **Index**: Vector is stored in PostgreSQL with pgvector
5. **Search**: Query is expanded, embedded, and matched using hybrid search

### Hybrid Search Algorithm

VisionAI uses a sophisticated hybrid search that achieves **86% high-confidence matches**:

1. **Query Expansion** - Short queries are expanded to match description style
   - `"dog"` → `"an image showing dog"`
2. **Vector Search** - Cosine similarity on embeddings
3. **Keyword Injection** - Direct SQL ILIKE search for keyword matches
4. **Keyword Boosting** - Up to +0.5 score for keyword presence
5. **Re-ranking** - Final sort by combined score

## Quick Start

### Prerequisites

- Python 3.14+
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

### 3. Install VisionAI

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

#### Option C: PyCharm Setup

1. **Clone the repository** using PyCharm:
   - File → New → Project from Version Control
   - Enter the repository URL

2. **Configure the Python interpreter**:
   - File → Settings → Project → Python Interpreter
   - Click the gear icon → Add Interpreter → Add Local Interpreter
   - Select "Virtualenv Environment" → "New"
   - Choose Python 3.10+ as the base interpreter
   - Click OK

3. **Install dependencies**:
   - Open the terminal in PyCharm (View → Tool Windows → Terminal)
   - Run: `pip install -r requirements.txt`

4. **Configure run configuration**:
   - Run → Edit Configurations → Add New → Python
   - Script path: `main.py`
   - Or use the uvicorn command: `uvicorn main:app --reload`

### 4. Configure Environment

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

### 5. Run the Server

```bash
uvicorn main:app --reload
```

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
    "created_at": "2025-02-16T10:30:00Z"
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
├── main.py                 # Application entry point
├── pyproject.toml          # Dependencies
├── .env                    # Configuration
├── app/
│   ├── __init__.py         # FastAPI app factory
│   ├── config.py           # Settings management
│   ├── database.py         # SQLAlchemy setup
│   ├── dependencies.py     # Dependency injection
│   ├── models/
│   │   └── db_models.py    # Image ORM model
│   ├── schemas/
│   │   └── api_schemas.py  # Pydantic schemas
│   ├── routers/
│   │   └── api.py          # API endpoints
│   └── services/
│       ├── http_client.py  # Ollama HTTP client
│       ├── vision.py       # Image description service
│       ├── ollama_embedding.py  # Text embedding service
│       └── storage.py      # File storage utilities
└── uploads/                # Uploaded images
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

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


