# RAG Document Q&A System

A complete Retrieval-Augmented Generation (RAG) application using Endee vector database for high-performance document search and OpenAI for answer generation.

## 🏗️ Architecture

```
Documents → [Processing] → [Embedding] → [Endee Vector DB]
                                              ↓
Question → [Embedding] → [Vector Search] → [Retrieved Docs] → [OpenAI] → Answer
```

## 🚀 Features

- **Document Ingestion**: Support for TXT, PDF, DOCX, and Markdown files
- **Smart Chunking**: Intelligent text splitting with configurable overlap
- **Vector Search**: High-performance similarity search using Endee
- **AI Answer Generation**: Context-aware answers using OpenAI
- **Web Interface**: User-friendly Streamlit dashboard
- **REST API**: Full-featured FastAPI backend
- **Multi-format Support**: Handle various document types

## 📋 Prerequisites

1. **Endee Vector Database** (already running via Docker)
2. **Python 3.8+**
3. **OpenAI API Key** (for answer generation)

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Required for answer generation
export OPENAI_API_KEY="your-openai-api-key-here"

# Optional - Endee configuration (defaults should work)
export ENDEE_HOST="localhost"
export ENDEE_PORT="8080"
```

### 3. Test the System

```bash
python run_app.py test
```

### 4. Run the Application

#### Option A: Web Interface (Recommended)
```bash
python run_app.py web
```
Then open: http://localhost:8501

#### Option B: API Server
```bash
python run_app.py api
```
API docs: http://localhost:8000/docs

## 📖 Usage Examples

### Command Line Interface

#### Process Documents
```bash
# Ingest a single document
python run_app.py ingest "path/to/document.pdf"

# Ingest to specific collection
python run_app.py ingest "document.pdf" --collection "research_papers"
```

#### Query Documents
```bash
# Ask a question
python run_app.py query "What is the main conclusion of the paper?"

# Query specific collection
python run_app.py query "How does this work?" --collection "technical_docs"
```

### Web Interface

1. **Upload Documents**: Drag and drop files in the upload tab
2. **Ask Questions**: Type questions in natural language
3. **View Sources**: See which documents were used for answers
4. **Manage Collections**: Create and organize document collections

### REST API

#### Upload Document
```bash
curl -X POST "http://localhost:8000/documents/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@document.pdf"
```

#### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the main topic?",
       "collection_name": "documents",
       "top_k": 5
     }'
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Embedding Model**: Default is `all-MiniLM-L6-v2` (384 dimensions)
- **Chunk Size**: Default 1000 tokens with 200 overlap
- **Retrieval Settings**: Top-K results and similarity thresholds
- **OpenAI Model**: Default `gpt-3.5-turbo`

## 🔧 System Components

### Core Components
- `rag_pipeline.py` - Main RAG orchestration
- `vector_store.py` - Endee vector database client
- `embedding_service.py` - Text embedding generation
- `document_processor.py` - Document parsing and chunking
- `answer_generator.py` - OpenAI answer generation

### Interfaces
- `web_app.py` - Streamlit web interface
- `api_server.py` - FastAPI REST server
- `run_app.py` - CLI runner and utilities

## 🐛 Troubleshooting

### Endee Connection Issues
```bash
# Check if Endee is running
curl http://localhost:8080/api/v1/health

# Restart Endee
docker-compose restart
```

### Python Dependencies
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### OpenAI API Issues
- Verify API key is set: `echo $OPENAI_API_KEY`
- Check API quota and billing
- Test with a simple API call

## 📊 Performance Tips

1. **Chunk Size**: Smaller chunks (500-1000 tokens) for precise answers
2. **Embedding Model**: Use larger models for better semantic understanding
3. **Top-K**: Start with 3-5 sources, increase if answers lack context
4. **Collections**: Organize documents by topic for better retrieval

## 🔐 Security Notes

- Set authentication tokens for production use
- Validate file uploads and limit file sizes
- Use environment variables for API keys
- Consider rate limiting for API endpoints

## 📈 Scaling Considerations

- Endee handles millions of vectors efficiently
- Consider batch processing for large document sets
- Monitor embedding API costs
- Use caching for frequently asked questions