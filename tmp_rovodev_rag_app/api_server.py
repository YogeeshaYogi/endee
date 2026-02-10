"""FastAPI REST API server for the RAG application."""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from pathlib import Path
import logging

from rag_pipeline import RAGPipeline
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Q&A API",
    description="REST API for document ingestion and question answering using Endee vector database",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None
    top_k: Optional[int] = None

class QueryResponse(BaseModel):
    status: str
    question: str
    answer: str
    sources: List[dict]
    num_sources: int

class DocumentResponse(BaseModel):
    status: str
    filename: str
    chunks_created: Optional[int] = None
    collection: Optional[str] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    endee_status: str
    collections: List[str]
    embedding_model: str
    embedding_dimension: int
    default_collection: str

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_system
    try:
        rag_system = RAGPipeline()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Continue startup but mark system as unavailable

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "RAG Document Q&A API", "status": "running"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        status = rag_system.get_system_status()
        return {
            "status": "healthy",
            "endee_connected": status["endee_status"] == "connected",
            "system_info": status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"System health check failed: {e}")

@app.get("/status", response_model=StatusResponse)
async def get_system_status():
    """Get system status and configuration."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        status = rag_system.get_system_status()
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")

@app.post("/documents/upload", response_model=List[DocumentResponse])
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = Form(None)
):
    """Upload and process documents."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    collection = collection_name or Config.DEFAULT_COLLECTION
    results = []
    
    for file in files:
        try:
            # Validate file type
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in Config.ALLOWED_EXTENSIONS:
                results.append(DocumentResponse(
                    status="error",
                    filename=file.filename,
                    error=f"Unsupported file type: {file_extension}"
                ))
                continue
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Process document
                result = rag_system.ingest_document(tmp_path, collection)
                results.append(DocumentResponse(
                    status=result["status"],
                    filename=result["filename"],
                    chunks_created=result.get("chunks_created"),
                    collection=result.get("collection"),
                    error=result.get("error")
                ))
            finally:
                # Cleanup temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            results.append(DocumentResponse(
                status="error",
                filename=file.filename,
                error=str(e)
            ))
    
    return results

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with a question."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(
            request.question,
            request.collection_name,
            request.top_k
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.get("/collections")
async def list_collections():
    """List all available collections."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        collections = rag_system.vector_store.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")

@app.post("/collections/{collection_name}")
async def create_collection(collection_name: str):
    """Create a new collection."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        success = rag_system.vector_store.create_collection(
            collection_name,
            rag_system.embedding_service.get_dimension()
        )
        
        if success:
            return {"message": f"Collection '{collection_name}' created successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create collection")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection creation failed: {e}")

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection and all its documents."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        success = rag_system.delete_collection(collection_name)
        
        if success:
            return {"message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Collection not found or deletion failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection deletion failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)