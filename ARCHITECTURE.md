# VisionAI Architecture Documentation

This document provides a comprehensive technical overview of VisionAI, including system architecture, the hybrid search algorithm, data flow, and implementation details.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [The Hybrid Search Algorithm](#the-hybrid-search-algorithm)
6. [Database Schema](#database-schema)
7. [API Endpoints](#api-endpoints)
8. [Service Layer Details](#service-layer-details)
9. [Frontend Architecture](#frontend-architecture)
10. [Configuration](#configuration)
11. [Performance Considerations](#performance-considerations)

---

## System Overview

VisionAI is a semantic image search engine that bridges the gap between visual content and natural language queries. The system allows users to:

1. **Upload images** - Images are automatically analyzed by an AI vision model
2. **Generate descriptions** - The vision model produces natural language descriptions
3. **Create embeddings** - Descriptions are converted to high-dimensional vectors
4. **Search semantically** - Users find images using natural language queries

### Technology Stack

| Component       | Technology             | Purpose                              |
|-----------------|------------------------|--------------------------------------|
| Web Framework   | FastAPI                | REST API with automatic OpenAPI docs |
| Database        | PostgreSQL + pgvector  | Vector similarity search             |
| ORM             | SQLAlchemy 2.0         | Database abstraction                 |
| Vision Model    | LLaVA (via Ollama)     | Image description generation         |
| Embedding Model | Llama 3.1 (via Ollama) | Text-to-vector conversion            |
| Validation      | Pydantic               | Request/response validation          |
| Frontend        | React 19 + TypeScript  | User interface                       |
| Build Tool      | Vite                   | Frontend bundling and dev server     |
| Testing         | Vitest + Testing Library | Frontend unit tests                |

---

## Architecture Diagram

### System Overview

```mermaid
graph LR
    FE["React Frontend<br/>:5173"]
    BE["FastAPI<br/>:8000"]
    SVC["Service Layer"]
    DATA["Data Layer"]
    OLLAMA["Ollama<br/>:11434"]

    FE -->|"Vite proxy"| BE
    BE --> SVC
    BE --> DATA
    SVC -->|"HTTP"| OLLAMA
```

### Frontend Layer

```mermaid
graph LR
    SP[SearchPage] -->|search query| AC
    UP[UploadPage] -->|file + description| AC
    IC[ImageCard] -->|renders results from| AC
    AC["api/client.ts<br/>API Client"]
    AC -->|"/api/v1/*"| BE["FastAPI :8000"]
```

### FastAPI Application Layer

```mermaid
graph TB
    APP["app/__init__.py<br/>Application Factory"] --> ROUTER

    ROUTER["routers/api.py"]
    ROUTER --> R1["POST /api/v1/images/upload"]
    ROUTER --> R2["GET  /api/v1/images/search/text"]
    ROUTER --> R3["GET  /api/v1/images/{id}"]
    ROUTER --> R4["DELETE /api/v1/images/{id}"]

    R1 & R2 & R3 & R4 --> DEP

    subgraph Support["Supporting Modules"]
        DEP["dependencies.py<br/>get_db · get_vision_svc · get_embed_svc"]
        SCHEMAS["schemas/api_schemas.py<br/>ImageResponse · SearchResponse · ImageUpload"]
        MODELS["models/db_models.py<br/>Image ORM · pgvector embedding column"]
    end
```

### Service & Data Layers

```mermaid
graph TB
    subgraph Services["Service Layer"]
        VS["vision.py<br/>VisionService<br/>Image → Text description"]
        ES["ollama_embedding.py<br/>OllamaEmbeddingService<br/>Text → 4096-dim vector<br/>L2 norm · in-memory cache"]
        HC["http_client.py<br/>post_with_retry<br/>3 attempts · 120s timeout"]
        VS & ES --> HC
    end

    subgraph Ollama["Ollama API Server (localhost:11434)"]
        LLAVA["LLaVA<br/>Multimodal vision model<br/>Generates descriptions"]
        LLAMA["Llama 3.1<br/>Embedding model<br/>Text → 4096-dim vector"]
    end

    subgraph Data["Data Layer"]
        DB["database.py<br/>SQLAlchemy engine<br/>Pool: 5 + 10 overflow"]
        PG[("PostgreSQL + pgvector<br/>images table<br/>Cosine distance search")]
        STORAGE["storage.py<br/>File validation · UUID filenames"]
        UPLOADS[("uploads/<br/>Image file storage")]
        DB --> PG
        STORAGE --> UPLOADS
    end

    HC -->|"/api/generate"| LLAVA
    HC -->|"/api/embeddings"| LLAMA
```

---

## Core Components

### 1. Application Factory (`app/__init__.py`)

The application uses the factory pattern for flexible configuration:

```python
def create_app() -> FastAPI:
    app = FastAPI(
        title="VisionAI",
        description="AI-powered image search",
        version="0.1.0",
        lifespan=lifespan,  # Handles startup/shutdown
    )
    
    # Mount static files for image serving
    app.mount("/static/uploads", StaticFiles(...))
    
    # Include API routes
    app.include_router(images_router, prefix="/api/v1")
    
    return app
```

**Lifespan Management:**
- **Startup**: Creates database tables, ensures upload directory exists
- **Shutdown**: Cleanup hooks (currently minimal)

### 2. Configuration (`app/config.py`)

Centralized configuration using Pydantic Settings:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    # Database
    visionai_database_url: str = "postgresql+psycopg://..."
    
    # File storage
    upload_dir: Path = Path("uploads")
    
    # Ollama API
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava:latest"           # Vision model
    ollama_embedding_model: str = "llama3.1:latest"  # Embedding model
    ollama_embedding_dimension: int = 4096
```

Settings are loaded from environment variables or `.env` file.

### 3. Database Layer (`app/database.py`)

SQLAlchemy 2.0 with connection pooling:

```python
engine = create_engine(
    settings.visionai_database_url,
    pool_size=5,        # Persistent connections
    max_overflow=10,    # Additional connections under load
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
```

### 4. Dependency Injection (`app/dependencies.py`)

FastAPI's dependency system provides:

- **`get_db()`**: Database session with automatic cleanup
- **`get_vision_service()`**: Singleton vision model service
- **`get_ollama_embedding_service()`**: Singleton embedding service

---

## Data Flow

### Image Upload Flow

```mermaid
flowchart LR
    A["Client<br/>Upload"] -->|"Image file"| B["Validate<br/>Image"]
    B -->|"Check type<br/>and size"| C["Describe<br/>(LLaVA)"]
    C -->|"Generate<br/>'The image shows...'"| D["Embed<br/>(Llama)"]
    D -->|"Convert description<br/>to 4096-dim vector"| E["Store<br/>DB"]
    E -->|"Save file,<br/>metadata & vector"| F([Done])
```

**Detailed Steps:**

1. **Receive Upload**: Client sends image via multipart form
2. **Validate**: Check file type (JPEG, PNG, etc.) and size (max 10MB)
3. **Generate Description**: 
   - Resize image to 384x384 for efficiency
   - Send to LLaVA with prompt "What is in this image? Describe briefly."
   - Clean any garbled output artifacts
4. **Create Embedding**: 
   - Send description to Llama 3.1 embedding endpoint
   - L2 normalize the resulting vector
5. **Store**: 
   - Save image file with UUID filename
   - Insert database record with description + embedding

### Search Flow

```mermaid
flowchart LR
    A["Search<br/>Query"] -->|"'dog'"| B["Expand<br/>Query"]
    B -->|"'an image<br/>showing dog'"| C["Embed<br/>Query"]
    C -->|"Convert to<br/>4096-dim vector"| D["Hybrid<br/>Search"]
    D -->|"Vector search +<br/>keyword injection +<br/>boosting"| E["Return<br/>Results"]
    E --> F(["Ranked results<br/>with scores"])
```

---

## The Hybrid Search Algorithm

VisionAI's search algorithm achieves **86% high-confidence matches** through a sophisticated five-step process. This section explains each step in detail.

### Why Hybrid Search?

Pure vector similarity search has limitations:

| Problem                 | Example                                                                                        | Solution          |
|-------------------------|------------------------------------------------------------------------------------------------|-------------------|
| Short query mismatch    | Query "dog" doesn't match description "The image shows a golden retriever playing in the park" | Query expansion   |
| Semantic-only ranking   | Image with "dog" in description ranks lower than semantically similar but less relevant images | Keyword boosting  |
| Missing obvious matches | Vector search may miss images that clearly contain the query term                              | Keyword injection |

### Algorithm Overview

```mermaid
flowchart TD
    IN([User query string<br/>e.g. 'dog']) --> S1

    S1["1. EXPAND QUERY<br/>Transform short queries to match<br/>description embedding style"]
    S2["2. EXTRACT KEYWORDS<br/>Pull meaningful search terms<br/>for boosting"]
    S3["3. VECTOR SEARCH<br/>Find semantically similar images<br/>via cosine similarity"]
    S4["4. KEYWORD INJECTION<br/>Fetch images containing<br/>keywords in descriptions"]
    S5["5. SCORE & RANK<br/>Combine vector scores with<br/>keyword boosts, sort by final score"]

    S1 --> S2 --> S3 --> S4 --> S5
    S5 --> OUT([Ranked list of<br/>image + score pairs])
```

### Step 1: Query Expansion

**Purpose**: Bridge the semantic gap between short queries and verbose descriptions.

**The Problem**: 
- Descriptions are generated as: "The image shows a golden retriever playing..."
- User searches with: "dog"
- These have low cosine similarity despite being semantically related

**The Solution**: Expand short queries to match the description format:

```python
def _expand_query(query: str) -> str:
    query = query.strip().lower()
    word_count = len(query.split())

    if word_count <= 2:
        # "dog" → "an image showing dog"
        return f"an image showing {query}"
    elif word_count <= 4:
        # "sunset over ocean" → "image of sunset over ocean"
        return f"image of {query}"
    else:
        # Verbose queries are already descriptive
        return query
```

**Examples**:

| Original Query                            | Expanded Query               |
|-------------------------------------------|------------------------------|
| `dog`                                     | `an image showing dog`       |
| `cute cat`                                | `an image showing cute cat`  |
| `sunset over ocean`                       | `image of sunset over ocean` |
| `person standing on a mountain at sunset` | (unchanged)                  |

### Step 2: Keyword Extraction

**Purpose**: Identify meaningful terms for direct matching and boosting.

```python
STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'image', 'showing', 'shows', 'picture', 'photo', 'photograph',
    # ... more stop words
})

def _extract_keywords(query: str) -> list[str]:
    words = query.lower().split()
    keywords = []
    
    for word in words:
        cleaned = word.strip('.,!?;:()[]{}"\'-')
        if cleaned not in STOP_WORDS and len(cleaned) >= 2:
            keywords.append(cleaned)
    
    return keywords
```

**Examples**:

| Query                          | Extracted Keywords                       |
|--------------------------------|------------------------------------------|
| `a cute dog in the park`       | `['cute', 'dog', 'park']`                |
| `the sunset over the ocean`    | `['sunset', 'ocean']`                    |
| `beautiful mountain landscape` | `['beautiful', 'mountain', 'landscape']` |

### Step 3: Vector Similarity Search

**Purpose**: Find semantically similar images regardless of exact wording.

```python
# Embed the expanded query
query_embedding = ollama_embed_svc.embed_text(query_for_embedding)

# Calculate fetch limit (fetch more for re-ranking)
fetch_limit = min(limit * 5, 100)

# Vector search using pgvector cosine distance
vector_results = db.execute(
    select(
        Image,
        (1 - Image.embedding.cosine_distance(query_embedding)).label("score"),
    )
    .where(Image.embedding.isnot(None))
    .order_by(Image.embedding.cosine_distance(query_embedding))
    .limit(fetch_limit)
).all()
```

**How Cosine Similarity Works**:

Cosine similarity measures the angle between two vectors, regardless of magnitude:

```
similarity = (A · B) / (||A|| × ||B||)
```

For L2-normalized vectors (unit vectors), this simplifies to the dot product:

```
similarity = A · B
```

**Score Range**: 0.0 (completely dissimilar) to 1.0 (identical)

### Step 4: Keyword Injection

**Purpose**: Ensure images containing exact keyword matches aren't missed.

**The Problem**: Vector search might rank semantically similar but keyword-absent images higher than exact matches.

**The Solution**: Fetch additional candidates via SQL ILIKE search:

```python
def _fetch_keyword_candidates(db: Session, keywords: list[str]) -> set[Image]:
    candidates = set()
    
    for kw in keywords:
        results = db.execute(
            select(Image)
            .where(Image.embedding.isnot(None))
            .where(Image.description.ilike(f"%{kw}%"))
            .limit(20)  # Max 20 per keyword
        ).scalars().all()
        candidates.update(results)
    
    return candidates
```

**Merging Candidates**:

```python
def _merge_candidates(vector_results, keyword_candidates, query_embedding):
    seen_ids = set()
    all_candidates = []
    
    # Add vector results first (they already have scores)
    for row in vector_results:
        if row.Image.id not in seen_ids:
            seen_ids.add(row.Image.id)
            all_candidates.append((row.Image, float(row.score)))
    
    # Add keyword candidates with calculated scores
    for img in keyword_candidates:
        if img.id not in seen_ids:
            seen_ids.add(img.id)
            # Calculate cosine similarity manually
            img_emb = np.array(img.embedding)
            vector_score = float(np.dot(img_emb, query_embedding))
            all_candidates.append((img, vector_score))
    
    return all_candidates
```

### Step 5: Keyword Boosting & Re-ranking

**Purpose**: Reward images that contain the user's search terms.

```python
MAX_KEYWORD_BOOST = 0.5  # Maximum boost value

def _calculate_keyword_boost(description: str, keywords: list[str]) -> float:
    if not description or not keywords:
        return 0.0
    
    desc_lower = description.lower()
    matches = sum(1 for kw in keywords if kw in desc_lower)
    
    if matches == 0:
        return 0.0
    
    # Proportional boost: more matches = higher boost
    match_ratio = matches / len(keywords)
    return min(MAX_KEYWORD_BOOST, match_ratio * MAX_KEYWORD_BOOST)
```

**Boost Examples**:

| Description                 | Keywords          | Matches | Boost |
|-----------------------------|-------------------|---------|-------|
| "A dog playing in the park" | `['dog', 'park']` | 2/2     | +0.50 |
| "A dog playing in the park" | `['dog', 'cat']`  | 1/2     | +0.25 |
| "A cat sleeping on a couch" | `['dog', 'park']` | 0/2     | +0.00 |

**Final Scoring & Ranking**:

```python
def _score_and_rank(candidates, keywords, min_score):
    scored_results = []
    
    for img, vector_score in candidates:
        keyword_boost = _calculate_keyword_boost(img.description, keywords)
        final_score = vector_score + keyword_boost
        
        if final_score >= min_score:
            scored_results.append((img, final_score, vector_score, keyword_boost))
    
    # Sort by combined score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    return scored_results
```

### Complete Algorithm Flow

```python
def search_by_text(q: str, limit: int, min_score: float, db, embedding_service):
    # Step 1: Expand query
    query_for_embedding = _expand_query(q)  # "dog" → "an image showing dog"
    
    # Step 2: Extract keywords
    keywords = _extract_keywords(q)  # ["dog"]
    
    # Step 3: Embed and vector search
    query_embedding = embedding_service.embed_text(query_for_embedding)
    vector_results = db.execute(
        select(Image, cosine_similarity)
        .order_by(cosine_distance)
        .limit(fetch_limit)
    ).all()
    
    # Step 4: Keyword injection
    keyword_candidates = _fetch_keyword_candidates(db, keywords)
    all_candidates = _merge_candidates(vector_results, keyword_candidates, query_embedding)
    
    # Step 5: Score and rank
    scored_results = _score_and_rank(all_candidates, keywords, min_score)
    
    return scored_results[:limit]
```

### Why This Works: A Concrete Example

**Search Query**: `"dog"`

**Image A** - Description: "The image shows a golden retriever playing in a sunny park"
- Vector similarity to "an image showing dog": 0.72
- Keyword "dog" not found (contains "retriever" not "dog"): +0.00
- **Final Score: 0.72**

**Image B** - Description: "The image shows a dog with white fur sitting on grass"
- Vector similarity to "an image showing dog": 0.68
- Keyword "dog" found: +0.50
- **Final Score: 1.18**

**Result**: Image B ranks higher despite lower vector similarity because it contains the exact search term.

---

## Database Schema

### Images Table

```sql
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,           -- UUID-based stored filename
    original_filename VARCHAR(255) NOT NULL,  -- User's original filename
    filepath VARCHAR(500) NOT NULL UNIQUE,    -- Full server path
    content_type VARCHAR(100) NOT NULL,       -- MIME type
    file_size INTEGER NOT NULL,               -- Size in bytes
    description TEXT,                          -- AI-generated description
    embedding VECTOR(4096),                    -- pgvector embedding column
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for vector similarity search
CREATE INDEX ON images USING ivfflat (embedding vector_cosine_ops);
```

### SQLAlchemy Model

```python
class Image(Base):
    __tablename__ = "images"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding = mapped_column(Vector(4096), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

---

## API Endpoints

### POST `/api/v1/images/upload`

Upload and index a new image.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (required), `description` (optional)

**Process**:
1. Validate file type and size
2. Generate description (if not provided)
3. Create embedding
4. Save file and database record

**Response**: `201 Created`
```json
{
  "message": "Image uploaded and indexed successfully.",
  "image": {
    "id": 1,
    "filename": "a1b2c3d4.jpg",
    "original_filename": "vacation.jpg",
    "description": "The image shows a beach with crystal blue water...",
    "created_at": "2025-02-19T10:30:00Z"
  }
}
```

### GET `/api/v1/images/search/text`

Semantic search using natural language.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `q` | string | Yes | - | Search query (1-500 chars) |
| `limit` | int | No | 10 | Max results (1-100) |
| `min_score` | float | No | 0.0 | Minimum score filter |

**Response**: `200 OK`
```json
{
  "query": "sunset over ocean",
  "results": [
    {
      "id": 42,
      "filename": "sunset.jpg",
      "description": "The image shows a beautiful sunset over the Pacific Ocean...",
      "score": 1.15
    }
  ],
  "total": 1
}
```

### GET `/api/v1/images/{image_id}`

Get details of a specific image.

**Response**: `200 OK` or `404 Not Found`

### DELETE `/api/v1/images/{image_id}`

Delete an image and its file.

**Response**: `204 No Content` or `404 Not Found`

---

## Service Layer Details

### VisionService (`app/services/vision.py`)

Generates text descriptions from images using LLaVA.

**Key Features**:
- **Singleton Pattern**: Only one instance for efficiency
- **Image Resizing**: Scales to 384x384 for faster processing
- **Output Cleaning**: Removes garbled prefixes from model output

```python
class VisionService:
    def describe_image(self, image: PIL.Image) -> str:
        # Resize for efficiency
        img_copy = image.copy()
        img_copy.thumbnail((384, 384))
        
        # Convert to base64
        image_b64 = base64.b64encode(img_buffer).decode()
        
        # Call LLaVA
        response = post_with_retry("/api/generate", {
            "model": "llava:latest",
            "prompt": "What is in this image? Describe briefly.",
            "images": [image_b64],
        })
        
        # Clean and return
        return clean_description(response["response"])
```

### OllamaEmbeddingService (`app/services/ollama_embedding.py`)

Converts text to vector embeddings.

**Key Features**:
- **Singleton Pattern**: Only one instance
- **L2 Normalization**: Ensures consistent cosine similarity
- **In-Memory Cache**: Speeds up repeated queries

```python
class OllamaEmbeddingService:
    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if use_cache and cache_key in _embedding_cache:
            return _embedding_cache[cache_key]
        
        # Call Ollama embedding API
        response = post_with_retry("/api/embeddings", {
            "model": "llama3.1:latest",
            "prompt": text,
        })
        
        # L2 normalize
        embedding = np.array(response["embedding"])
        embedding = embedding / np.linalg.norm(embedding)
        
        # Cache and return
        result = embedding.tolist()
        self._cache_embedding(cache_key, result)
        return result
```

### HTTP Client (`app/services/http_client.py`)

Robust HTTP client for Ollama API communication.

**Key Features**:
- **Retry Logic**: 3 attempts with 5-second delays
- **Timeout Handling**: 120-second timeout
- **Error Logging**: Detailed failure messages

```python
def post_with_retry(endpoint: str, payload: dict, error_context: str) -> dict:
    url = f"{settings.ollama_base_url}{endpoint}"
    
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(5)
                continue
            raise RuntimeError(f"{error_context} failed: {e}")
```

### Storage Service (`app/services/storage.py`)

File validation and storage operations.

**Key Features**:
- **Type Validation**: JPEG, PNG, WebP, GIF, BMP
- **Size Limit**: 10MB maximum
- **UUID Filenames**: Prevents collisions and path traversal

```python
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def save_to_disk(content: bytes, original_filename: str) -> tuple[str, str, int]:
    # Generate UUID filename
    ext = Path(original_filename).suffix or ".jpg"
    new_filename = f"{uuid.uuid4()}{ext}"
    
    # Save file
    filepath = get_upload_dir() / new_filename
    filepath.write_bytes(content)
    
    return new_filename, str(filepath), len(content)
```

---

## Frontend Architecture

The frontend is a React application built with TypeScript and Vite, providing a modern user interface for image search and upload.

### Frontend Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | React 19 | UI component library |
| Language | TypeScript | Type-safe JavaScript |
| Build Tool | Vite | Fast HMR and bundling |
| Testing | Vitest + Testing Library | Unit and integration tests |
| Styling | CSS Modules | Component-scoped styles |

### Frontend Structure

```
frontend/
├── src/
│   ├── main.tsx              # Application entry point
│   ├── App.tsx               # Root component with tab navigation
│   ├── App.css               # Global application styles
│   │
│   ├── api/
│   │   └── client.ts         # Backend API client
│   │
│   ├── components/
│   │   ├── ImageCard.tsx     # Reusable image display component
│   │   └── ImageCard.css
│   │
│   ├── pages/
│   │   ├── SearchPage.tsx    # Natural language search interface
│   │   ├── SearchPage.css
│   │   ├── UploadPage.tsx    # Image upload interface
│   │   └── UploadPage.css
│   │
│   └── test/
│       └── setup.ts          # Test configuration for Vitest
│
├── public/                   # Static assets
├── index.html                # HTML entry point
├── vite.config.ts            # Vite configuration
├── tsconfig.json             # TypeScript configuration
└── package.json              # Dependencies and scripts
```

### API Proxy Configuration

The Vite development server proxies API requests to the FastAPI backend, enabling seamless development without CORS issues:

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

**How it works:**
- Frontend runs on `http://localhost:5173`
- API calls to `/api/v1/images/...` are proxied to `http://localhost:8000/api/v1/images/...`
- Static files (uploaded images) at `/static/uploads/...` are proxied to the backend

### Key Components

#### App.tsx - Application Shell

The root component provides tab-based navigation between Search and Upload pages:

```tsx
function App() {
  const [tab, setTab] = useState<'search' | 'upload'>('search');

  return (
    <div className="app">
      <header className="app-header">
        <nav className="app-nav">
          <button onClick={() => setTab('search')}>Search</button>
          <button onClick={() => setTab('upload')}>Upload</button>
        </nav>
      </header>
      <main>
        {tab === 'search' ? <SearchPage /> : <UploadPage />}
      </main>
    </div>
  );
}
```

#### SearchPage.tsx - Search Interface

Provides natural language search with real-time results:

1. User enters a query (e.g., "sunset over mountains")
2. Query is sent to `/api/v1/images/search/text`
3. Results are displayed in a grid of ImageCard components
4. Each result shows the image, similarity score, and description

#### UploadPage.tsx - Upload Interface

Handles image upload with AI processing:

1. User selects an image file
2. Optionally provides a custom description
3. File is uploaded to `/api/v1/images/upload`
4. Backend generates AI description and embedding
5. Success/error status is displayed

#### client.ts - API Client

Type-safe API client with full TypeScript interfaces:

```typescript
interface ImageSearchResult {
  id: number;
  filename: string;
  original_filename: string;
  filepath: string;
  description: string | null;
  score: number;
}

interface SearchResponse {
  query: string | null;
  results: ImageSearchResult[];
  total: number;
}

export async function searchImages(
  q: string,
  limit = 20,
  minScore = 0.0,
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q,
    limit: String(limit),
    min_score: String(minScore),
  });
  const res = await fetch(`/api/v1/images/search/text?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'Search failed');
  }
  return res.json();
}

export function imageUrl(filename: string): string {
  return `/static/uploads/${filename}`;
}
```

### Frontend-Backend Communication Flow

```mermaid
sequenceDiagram
    participant Browser as React App (Browser :5173)
    participant Vite as Vite Dev Server (Proxy)
    participant API as FastAPI Backend (:8000)

    Browser->>Vite: fetch('/api/v1/...')
    Vite->>API: HTTP request
    API-->>Vite: JSON response
    Vite-->>Browser: JSON data
```

### Running the Frontend

**Development mode:**
```bash
cd frontend
npm install    # First time only
npm run dev    # Start dev server with HMR
```

**Production build:**
```bash
cd frontend
npm run build  # Outputs to frontend/dist/
npm run preview  # Preview production build
```

**Running tests:**
```bash
cd frontend
npm run test        # Run tests once
npm run test:watch  # Run tests in watch mode
```

---

## Configuration

### Environment Variables

| Variable                     | Type   | Default                                                         | Description                   |
|------------------------------|--------|-----------------------------------------------------------------|-------------------------------|
| `VISIONAI_DATABASE_URL`      | string | `postgresql+psycopg://postgres:1337@localhost:5432/visionai_db` | PostgreSQL connection URL     |
| `UPLOAD_DIR`                 | path   | `uploads`                                                       | Image storage directory       |
| `OLLAMA_BASE_URL`            | string | `http://localhost:11434`                                        | Ollama API server URL         |
| `OLLAMA_API_KEY`             | string | ``                                                              | Optional API key              |
| `OLLAMA_MODEL`               | string | `llava:latest`                                                  | Vision model for descriptions |
| `OLLAMA_EMBEDDING_MODEL`     | string | `llama3.1:latest`                                               | Text embedding model          |
| `OLLAMA_EMBEDDING_DIMENSION` | int    | `4096`                                                          | Vector dimension size         |

### Supported Models

**Vision Models** (for description generation):
- `llava:latest` (7B) - Default, good balance
- `llava:13b` - More detailed descriptions
- `qwen2.5vl:latest` - Alternative vision model

**Embedding Models** (for text-to-vector):
- `llama3.1:latest` - 4096 dimensions, recommended
- `nomic-embed-text` - 768 dimensions, faster
- `mxbai-embed-large` - 1024 dimensions

---

## Performance Considerations

### Embedding Cache

The embedding service maintains an in-memory cache:
- **Size**: 1000 entries maximum
- **Eviction**: LRU-style (removes oldest 100 when full)
- **Use Case**: Repeated search queries

### Database Connection Pool

SQLAlchemy connection pool settings:
- **Pool Size**: 5 persistent connections
- **Max Overflow**: 10 additional connections under load

### Search Optimization

- **Fetch Multiplier**: Fetches 5x the requested limit for re-ranking
- **Maximum Fetch**: 100 candidates (prevents excessive memory use)
- **Keyword Limit**: 20 matches per keyword

### Image Processing

- **Resize Before Processing**: 384x384 max dimension
- **JPEG Quality**: 85% (balance of quality and size)
- **Keep Alive**: Vision model kept loaded for 30 minutes

---

## Appendix: Constants Reference

### Search Configuration

```python
MAX_KEYWORD_BOOST = 0.5          # Maximum score boost from keywords
FETCH_LIMIT_MULTIPLIER = 5       # Fetch 5x limit for re-ranking
MAX_FETCH_LIMIT = 100            # Maximum candidates to fetch
KEYWORD_MATCH_LIMIT = 20         # Max matches per keyword
```

### Vision Configuration

```python
MAX_IMAGE_SIZE = 384             # Max dimension for processing
JPEG_QUALITY = 85                # JPEG encoding quality
DESCRIPTION_PROMPT = "What is in this image? Describe briefly."
```

### Storage Configuration

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"
}
```

### HTTP Client Configuration

```python
TIMEOUT = 120                    # Seconds
MAX_RETRIES = 3                  # Retry attempts
RETRY_DELAY = 5                  # Seconds between retries
```