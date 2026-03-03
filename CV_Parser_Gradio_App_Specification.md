# CV Parser Gradio Application - Technical Specification

## 1. Project Overview

### 1.1 Purpose
The CV Parser Gradio Application is an intelligent resume processing system that enables users to upload resumes, extract and analyze their content, and perform intelligent queries about candidate information through a user-friendly web interface.

### 1.2 Key Features
- **Resume Upload and Processing**: Accept PDF resumes and extract text content
- **AI-Powered Content Analysis**: Classify and structure resume information using LLM
- **Persistent Storage**: Store structured data in PostgreSQL database
- **Semantic Search**: Create embeddings for intelligent content retrieval
- **RAG-based Querying**: Answer natural language questions about candidates
- **Web Interface**: Intuitive Gradio-based user interface

## 2. System Architecture

### 2.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                         Gradio Web Interface                        │
├─────────────────────────────────────────────────────────────────────┤
│                      Application Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ PDF Text    │  │ LLM Content │  │ RAG Query   │                 │
│  │ Extractor   │  │ Classifier  │  │ Engine      │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
├─────────────────────────────────────────────────────────────────────┤
│                      Data Layer                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ PostgreSQL  │  │ FAISS Vector│  │ HuggingFace │                 │
│  │ Database    │  │ Store       │  │ Embeddings  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
├─────────────────────────────────────────────────────────────────────┤
│                      External Services                              │
│  ┌─────────────┐  ┌─────────────┐                                   │
│  │ Groq LLM    │  │ HuggingFace │                                   │
│  │ API         │  │ API         │                                   │
│  └─────────────┘  └─────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow
1. **Resume Upload** → PDF file uploaded through Gradio interface
2. **Text Extraction** → PyMuPDF extracts raw text content
3. **Content Classification** → Groq LLM structures the content into categories
4. **Database Storage** → Structured data saved to PostgreSQL
5. **Embedding Generation** → HuggingFace model creates vector embeddings
6. **Vector Storage** → Embeddings stored in FAISS index
7. **Query Processing** → User queries processed through RAG system
8. **Response Generation** → LLM generates natural language responses

## 3. Technology Stack

### 3.1 Core Technologies
- **Frontend**: Gradio 4.x
- **Backend**: Python 3.9+
- **PDF Processing**: PyMuPDF (fitz)
- **LLM Service**: Groq API (llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace Transformers API
- **Vector Database**: FAISS
- **Database**: PostgreSQL 14+
- **RAG Framework**: LangChain
- **Containerization**: Docker & Docker Compose

### 3.2 Python Dependencies
```
Up to you
```

## 4. Core Components

### 4.1 PDF Text Extraction Module
**File**: `pdf_extractor.py`

```python
class PDFExtractor:
    def extract_text(self, pdf_file_path: str) -> str:
        """Extract text content from PDF file using PyMuPDF"""
        
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes for Gradio uploads"""
```

### 4.2 CV Classification Module
**File**: `cv_classifier.py`

```python
class CVClassifier:
    def __init__(self):
        """Initialize with Groq API client"""
        
    def classify_cv_sections(self, cv_text: str) -> dict:
        """Classify CV content into structured sections"""
        
    def extract_specific_info(self, cv_data: dict) -> dict:
        """Extract specific fields like name, email, experience etc."""
```

**Classification Schema**:
- `education`: Educational background, degrees, schools
- `experience`: Professional experience, work history
- `skills`: Technical skills, programming languages, tools
- `projects`: Personal/academic projects, portfolio items
- `summary`: Professional summary, objective
- `contact`: Contact information, personal details
- `achievements`: Awards, honors, recognitions
- `languages`: Language proficiency
- `publications`: Research papers, articles
- `certifications`: Professional certifications, licenses
- `references`: Reference information
- `other`: Other relevant information

### 4.3 Database Module
**File**: `database.py`

```python
class CVDatabase:
    def __init__(self, connection_string: str):
        """Initialize PostgreSQL connection"""
        
    def create_tables(self):
        """Create necessary database tables"""
        
    def insert_cv_data(self, cv_data: dict) -> int:
        """Insert classified CV data and return CV ID"""
        
    def query_cvs(self, sql_query: str) -> List[dict]:
        """Execute SQL query and return results"""
        
    def get_cv_by_id(self, cv_id: int) -> dict:
        """Retrieve CV data by ID"""
```

### 4.4 Embedding Module
**File**: `embedding_service.py`

```python
class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize HuggingFace embedding service"""
        
    def generate_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        
    def create_cv_embeddings(self, cv_data: dict) -> List[dict]:
        """Create embeddings for different CV sections"""
```

### 4.5 Vector Store Module
**File**: `vector_store.py`

```python
class FAISSVectorStore:
    def __init__(self, dimension: int = 1024):
        """Initialize FAISS index"""
        
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]):
        """Add embeddings with metadata to index"""
        
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        """Find similar embeddings"""
        
    def save_index(self, file_path: str):
        """Persist FAISS index to disk"""
        
    def load_index(self, file_path: str):
        """Load FAISS index from disk"""
```

## 5. RAG Implementation

### 5.1 RAG Query Engine
**File**: `rag_engine.py`

```python
class RAGQueryEngine:
    def __init__(self, llm_client, vector_store, database):
        """Initialize RAG components"""
        
    def process_query(self, query: str) -> str:
        """Process user query and return answer"""
        
    def route_query(self, query: str) -> str:
        """Determine if query needs DB search or semantic search"""
```

### 5.2 Database Query Tool
```python
class DatabaseQueryTool:
    def __init__(self, database, llm_client):
        """Initialize database query tool"""
        
    def generate_sql_query(self, natural_language_query: str) -> str:
        """Convert natural language to SQL query"""
        
    def execute_and_format(self, sql_query: str) -> str:
        """Execute SQL and format results for LLM"""
```

**Example Queries**:
- "Who comes from Hanoi?"
- "How many candidates have AI experience?"
- "List all candidates with Computer Science degrees"

### 5.3 Semantic Search Tool
```python
class SemanticSearchTool:
    def __init__(self, vector_store, embedding_service, llm_client):
        """Initialize semantic search tool"""
        
    def semantic_search(self, query: str) -> str:
        """Perform semantic search and generate answer"""
        
    def find_relevant_context(self, query_embedding: np.ndarray) -> List[str]:
        """Find relevant CV sections based on similarity"""
```

**Example Queries**:
- "Who has better knowledge about AI?"
- "Find candidates with machine learning experience"
- "Who has worked on similar projects to chatbots?"

## 6. Database Schema

### 6.1 Main Tables

#### CVs Table
```sql
CREATE TABLE cvs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    location VARCHAR(255),
    current_job_title VARCHAR(255),
    educated BOOLEAN DEFAULT FALSE,
    major VARCHAR(255),
    gpa DECIMAL(3,2),
    experiment_years INTEGER,
    raw_text TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### CV Sections Table
```sql
CREATE TABLE cv_sections (
    id SERIAL PRIMARY KEY,
    cv_id INTEGER REFERENCES cvs(id) ON DELETE CASCADE,
    section_type VARCHAR(50) NOT NULL, -- education, experience, skills, etc.
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Embeddings Table
```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    cv_id INTEGER REFERENCES cvs(id) ON DELETE CASCADE,
    section_type VARCHAR(50),
    text_content TEXT,
    embedding_vector BYTEA, -- Serialized numpy array
    faiss_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2 Indexes
```sql
CREATE INDEX idx_cvs_name ON cvs(name);
CREATE INDEX idx_cvs_location ON cvs(location);
CREATE INDEX idx_cvs_major ON cvs(major);
CREATE INDEX idx_cv_sections_type ON cv_sections(section_type);
CREATE INDEX idx_embeddings_cv_section ON embeddings(cv_id, section_type);
```

## 7. Gradio Interface Design

### 7.1 Main Interface Components

#### Upload Tab
```python
def create_upload_interface():
    with gr.Tab("Upload Resume"):
        file_upload = gr.File(
            label="Upload PDF Resume",
            file_types=[".pdf"],
            file_count="single"
        )
        upload_btn = gr.Button("Process Resume", variant="primary")
        
        # Output components
        status_output = gr.Textbox(label="Processing Status")
        extracted_text = gr.Textbox(label="Extracted Text", max_lines=10)
        classified_data = gr.JSON(label="Classified Information")
```

#### Query Tab
```python
def create_query_interface():
    with gr.Tab("Query Resumes"):
        query_input = gr.Textbox(
            label="Ask a question about the resumes",
            placeholder="e.g., 'Who has experience in AI?' or 'Find candidates from Hanoi'"
        )
        query_btn = gr.Button("Search", variant="primary")
        
        # Output
        answer_output = gr.Textbox(label="Answer", max_lines=10)
        source_info = gr.JSON(label="Source Information")
```

#### Analytics Tab
```python
def create_analytics_interface():
    with gr.Tab("Analytics"):
        stats_display = gr.HTML(label="Database Statistics")
        location_chart = gr.Plot(label="Candidates by Location")
        skills_chart = gr.Plot(label="Top Skills")
        refresh_btn = gr.Button("Refresh Analytics")
```

### 7.2 Event Handlers
```python
def process_resume(file):
    """Handle resume upload and processing"""
    
def handle_query(query):
    """Handle user queries"""
    
def update_analytics():
    """Update analytics dashboard"""
```

## 8. Environment Configuration

### 8.1 Environment Variables
```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_hf_api_key_here

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=cv_parser
POSTGRES_USER=cv_user
POSTGRES_PASSWORD=cv_password

# Application Configuration
APP_PORT=7860
DEBUG_MODE=false

# Vector Store Configuration
FAISS_INDEX_PATH=./data/faiss_index
EMBEDDING_DIMENSION=1024

# LLM Configuration
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-m3
```

## 9. Docker Deployment

### 9.1 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Create data directory for FAISS
RUN mkdir -p ./data

# Run the application
CMD ["python", "app.py"]
```

### 9.2 Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: cv_parser
      POSTGRES_USER: cv_user
      POSTGRES_PASSWORD: cv_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  cv-parser-app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=cv_parser
      - POSTGRES_USER=cv_user
      - POSTGRES_PASSWORD=cv_password
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  postgres_data:
```

## 10. Application Structure

### 10.1 Directory Structure
```
app/
├── main.py                 # Gradio app entry point
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-container setup
├── .env.example          # Environment template
├── init.sql              # Database initialization
├── modules/
│   ├── __init__.py
│   ├── pdf_extractor.py   # PDF text extraction
│   ├── cv_classifier.py   # LLM classification
│   ├── database.py        # PostgreSQL operations
│   ├── embedding_service.py # HuggingFace embeddings
│   ├── vector_store.py     # FAISS operations
│   ├── rag_engine.py       # RAG query processing
│   └── gradio_interface.py # UI components
├── tools/
│   ├── __init__.py
│   ├── database_query.py   # SQL generation tool
│   └── semantic_search.py  # Semantic search tool
├── data/                  # Persistent data storage
│   ├── faiss_index/      # FAISS vector store
│   └── uploads/          # Temporary file storage
├── tests/                # Unit tests
│   ├── test_pdf_extractor.py
│   ├── test_cv_classifier.py
│   ├── test_database.py
│   └── test_rag_engine.py
└── docs/                 # Documentation
    ├── API.md            # API documentation
    ├── DEPLOYMENT.md     # Deployment guide
    └── TROUBLESHOOTING.md
```

### 10.2 Main Application File (main.py)
```python
import gradio as gr
import logging
from modules.gradio_interface import CVParserInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    try:
        # Initialize the CV Parser interface
        app = CVParserInterface()
        
        # Create Gradio app
        demo = app.create_interface()
        
        # Launch the app
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()
```

## 11. API Endpoints and LLM Integration

### 11.1 Groq LLM Configuration
```python
# LLM Configuration for all tasks
MODEL_NAME = "llama-3.3-70b-versatile"
MAX_TOKENS = 4096
TEMPERATURE = 0.1

# System prompts for different tasks
CV_CLASSIFICATION_PROMPT = """
You are an expert CV/Resume parser. Analyze the CV text and classify it into sections...
"""

SQL_GENERATION_PROMPT = """
You are a SQL expert. Convert the natural language query to SQL based on the schema...
"""

RAG_RESPONSE_PROMPT = """
Based on the retrieved context, provide a comprehensive answer to the user's question...
"""
```

### 11.2 Error Handling and Logging
```python
class CVParserError(Exception):
    """Custom exception for CV Parser application"""
    pass

class ErrorHandler:
    @staticmethod
    def handle_pdf_extraction_error(error):
        """Handle PDF extraction errors"""
        
    @staticmethod
    def handle_llm_api_error(error):
        """Handle LLM API errors"""
        
    @staticmethod
    def handle_database_error(error):
        """Handle database errors"""
```

## 12. Chunking Strategy for Embedding Generation

### 12.1 Batch Chunking Strategy

Thay vì gọi API embedding riêng lẻ cho từng section của CV, hệ thống áp dụng chiến lược **batch theo CV**: toàn bộ các section nội dung của một CV được gộp thành một batch duy nhất và gửi đi trong **một lần gọi API**.

**Nguyên tắc cốt lõi**:
- **1 lần gọi = 1 CV** (không gọi theo từng section riêng lẻ)
- Các **thông tin cá nhân định danh** (tên, email, số điện thoại, địa chỉ) được **loại trừ** khỏi batch embedding vì chúng đã được lưu riêng vào cột có cấu trúc trong PostgreSQL
- Chỉ embed các **section nội dung ngữ nghĩa**: `education`, `experience`, `skills`, `projects`, `summary`, `achievements`, `languages`, `publications`, `certifications`, `other`

### 12.2 Các Section Được Embed vs Bỏ Qua

| Section | Embed? | Lý do |
|---|---|---|
| `summary` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `experience` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `education` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `skills` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `projects` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `achievements` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `languages` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `publications` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `certifications` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `other` | ✅ | Nội dung ngữ nghĩa quan trọng |
| `contact` (tên, email, phone, địa chỉ) | ❌ | Đã lưu vào cột riêng trong DB, không cần embed |

### 12.3 Implementation

```python
# embedding_service.py

SECTIONS_TO_EMBED = {
    "summary", "experience", "education", "skills",
    "projects", "achievements", "languages",
    "publications", "certifications", "other"
}

SECTIONS_TO_SKIP = {"contact"}  # name, email, phone, address → lưu vào DB có cấu trúc

class EmbeddingService:
    def create_cv_embeddings(self, cv_id: int, cv_data: dict) -> List[dict]:
        """
        Tạo embeddings cho một CV theo chiến lược batch:
        - Tất cả section nội dung được gộp vào MỘT lần gọi API duy nhất
        - Các thông tin cá nhân (contact) bị loại trừ
        - Trả về list các chunk kèm metadata để lưu vào FAISS + DB
        """
        chunks = []
        texts_to_embed = []

        # Bước 1: Thu thập tất cả section cần embed của CV này
        for section_type, content in cv_data.items():
            if section_type not in SECTIONS_TO_EMBED:
                continue
            if not content or not content.strip():
                continue

            text = f"[{section_type.upper()}]\n{content.strip()}"
            texts_to_embed.append(text)
            chunks.append({
                "cv_id": cv_id,
                "section_type": section_type,
                "text_content": text,
            })

        if not texts_to_embed:
            return []

        # Bước 2: Gọi API embedding MỘT LẦN cho toàn bộ section của CV này
        embeddings = self.generate_embeddings(texts_to_embed)  # shape: (n_sections, dim)

        # Bước 3: Gắn vector vào từng chunk
        for i, chunk in enumerate(chunks):
            chunk["embedding_vector"] = embeddings[i]

        return chunks
```

### 12.4 Luồng Xử Lý Theo Batch

```
CV Text (raw)
     │
     ▼
LLM Classification
     │
     ▼
cv_data = {
  "summary":       "...",   ✅ → embed
  "experience":    "...",   ✅ → embed
  "education":     "...",   ✅ → embed
  "skills":        "...",   ✅ → embed
  "projects":      "...",   ✅ → embed
  "contact": {              ❌ → bỏ qua (lưu vào cột DB)
    "name":  "Nguyen Van A",
    "email": "a@email.com",
    "phone": "0912345678"
  }
}
     │
     ▼ (gộp tất cả section ✅ thành 1 batch)
HuggingFace Embedding API  ◄── 1 lần gọi duy nhất / 1 CV
     │
     ▼
[vector_section_1, vector_section_2, ..., vector_section_n]
     │
     ▼
FAISS Index + PostgreSQL embeddings table
```

### 12.5 Lợi Ích

| Tiêu chí | Gọi riêng lẻ từng section | **Batch theo CV (chiến lược này)** |
|---|---|---|
| Số lần gọi API / CV | N (bằng số section) | **1** |
| Latency | Cao (N round-trips) | **Thấp (1 round-trip)** |
| Chi phí API | Cao | **Thấp** |
| Độ phức tạp code | Trung bình | **Đơn giản, dễ kiểm soát** |
| Khả năng retry | Phức tạp | **Đơn giản (retry cả CV)** |

---

## 13. Monitoring and Maintenance

### 13.1 Health Checks
```python
def health_check():
    """Application health check endpoint"""
    checks = {
        "database": check_database_connection(),
        "groq_api": check_groq_api(),
        "huggingface_api": check_hf_api(),
        "faiss_index": check_faiss_index()
    }
    return checks
```

### 13.2 Logging Configuration
- Application logs
- Error tracking
- Performance metrics
- User interaction logs

---

This specification serves as a comprehensive guide for implementing the CV Parser Gradio application with all the specified requirements and features.