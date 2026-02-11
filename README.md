# 🤖 RAG Document Q&A System

A complete Retrieval-Augmented Generation (RAG) application powered by **Endee Vector Database** and **Google Gemini AI** for intelligent document search and question answering.

## ✨ Features

- 📄 **Multi-format support**: PDF, DOCX, TXT, Markdown
- 🔍 **Semantic search**: Find relevant content using AI embeddings  
- 🤖 **AI-powered answers**: Get intelligent responses from your documents
- ⚡ **High performance**: SIMD-optimized vector search with Endee
- 🌐 **Multiple interfaces**: Web UI, CLI, and REST API
- 🆓 **Cost-effective**: Uses Google Gemini's generous free tier

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- Docker (for Endee vector database)
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

### 2. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Endee vector database
docker-compose up -d

# Add your Gemini API key to .env file
GEMINI_API_KEY=your_api_key_here
```

### 3. Upload Documents
```bash
# Upload a document
python run_app.py ingest "path/to/your/document.pdf"

# Or use the web interface
python run_app.py web
```

### 4. Ask Questions
```bash
# Command line
python run_app.py query "What is the main topic of the document?"

# Web interface (recommended)
python run_app.py web
# Then open: http://localhost:8501
```

## 💻 Usage Examples

### Command Line Interface
```bash
# Standard AI-powered query
python run_app.py query "Explain artificial intelligence"

# Ultra-fast text extraction (instant)
python run_fast.py "What is machine learning?" --mode fast

# AI mode with Gemini
python run_fast.py "Summarize the key points" --mode ai
```

### Web Interface
```bash
python run_app.py web
```
- Upload documents via drag & drop
- Ask questions in natural language
- View source documents and relevance scores
- Manage document collections

### REST API
```bash
python run_app.py api
```
Visit http://localhost:8000/docs for interactive API documentation.

## ⚙️ Configuration

Edit `.env` file to customize:

```bash
# AI Configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Vector Database
ENDEE_HOST=localhost
ENDEE_PORT=8080

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_DOCUMENTS=5
```

## 🏗️ Architecture

```
Documents → [Processing] → [Embedding] → [Endee Vector DB]
                                              ↓
Question → [Embedding] → [Vector Search] → [Retrieved Docs] → [Gemini AI] → Answer
```

**Core Components:**
- **Endee**: High-performance vector database with SIMD optimizations
- **Google Gemini**: Fast, intelligent AI for answer generation
- **SentenceTransformers**: Creates semantic embeddings for search
- **FastAPI/Streamlit**: Multiple interface options

## 🔧 Troubleshooting

### Common Issues

**"Cannot connect to Endee"**
```bash
docker-compose up -d
```

**"Gemini API error"**
- Check your API key at https://makersuite.google.com/app/apikey
- Ensure you haven't exceeded the free tier limit (15 requests/minute)

**"No relevant documents found"**
- Upload documents first: `python run_app.py ingest "document.pdf"`
- Try broader search terms
- Check document format is supported

### Getting Help
- Use fast mode for instant results: `python run_fast.py "question" --mode fast`
- Check system status: `python run_app.py test`
- View available collections in the web interface

## 📊 Performance

- **Vector Search**: Instant retrieval with 60-80% relevance scores
- **AI Responses**: ~3 seconds with Gemini
- **Document Processing**: Handles PDFs up to 10MB
- **Throughput**: 15+ queries per minute (Gemini free tier)

## 🆓 Free Usage

This system uses Google Gemini's generous free tier:
- **15 requests per minute**
- **No credit card required**
- **High-quality AI responses**

Perfect for personal projects, research, and small teams!

## 🎯 Use Cases

- 📚 **Research**: Search academic papers and documents
- 💼 **Business**: Query company documentation and reports  
- 📖 **Education**: Get answers from textbooks and materials
- 🔍 **Legal**: Search contracts and legal documents
- 📝 **Personal**: Organize and search your document library

---

