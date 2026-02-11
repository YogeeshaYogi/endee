"""Complete RAG pipeline implementation."""

import os
import json
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from config import Config
from vector_store import EndeeVectorStore
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from answer_generator import AnswerGenerator
from simple_text_answer import SimpleTextAnswerer

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline."""
    
    def __init__(self):
        self.config = Config()
        self.vector_store = EndeeVectorStore()
        self.doc_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        
        # Try Ollama first, fallback to simple text answerer
        try:
            self.answer_generator = AnswerGenerator()
            self.use_ai = True
        except Exception as e:
            logger.warning(f"AI answer generator not available: {e}")
            self.answer_generator = SimpleTextAnswerer()
            self.use_ai = False
        
        # Ensure upload directory exists
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG system."""
        logger.info("Initializing RAG pipeline...")
        
        # Check Endee health
        if not self.vector_store.health_check():
            logger.warning("Endee vector database is not accessible")
            raise RuntimeError("Cannot connect to Endee vector database")
        
        # Validate configuration
        self.config.validate()
        
        # Create default collection if it doesn't exist
        collections = self.vector_store.list_collections()
        if Config.DEFAULT_COLLECTION not in collections:
            logger.info(f"Creating default collection: {Config.DEFAULT_COLLECTION}")
            self.vector_store.create_collection(
                Config.DEFAULT_COLLECTION,
                self.embedding_service.get_dimension()
            )
        
        # Create metadata storage directory
        self.metadata_dir = Path(Config.UPLOAD_DIR) / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RAG pipeline initialized successfully")
    
    def _store_metadata(self, collection_name: str, chunk_ids: List[str], chunks: List[Dict]) -> None:
        """Store metadata separately since Endee doesn't support it directly."""
        metadata_file = self.metadata_dir / f"{collection_name}_metadata.json"
        
        # Load existing metadata if it exists
        existing_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            except:
                existing_metadata = {}
        
        # Add new metadata
        for chunk_id, chunk in zip(chunk_ids, chunks):
            existing_metadata[chunk_id] = {
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            }
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
    
    def _get_metadata(self, collection_name: str, chunk_ids: List[str]) -> Dict[str, Dict]:
        """Retrieve stored metadata for given chunk IDs."""
        metadata_file = self.metadata_dir / f"{collection_name}_metadata.json"
        
        if not metadata_file.exists():
            return {}
        
        try:
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            # Return only requested chunk metadata
            return {chunk_id: all_metadata.get(chunk_id, {}) for chunk_id in chunk_ids}
        except:
            return {}
    
    def ingest_document(self, file_path: str, collection_name: str = None) -> Dict[str, Any]:
        """Ingest a document into the vector database."""
        collection_name = collection_name or Config.DEFAULT_COLLECTION
        
        try:
            logger.info(f"Ingesting document: {file_path}")
            
            # Process document into chunks
            chunks = self.doc_processor.process_document(file_path)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.encode_batch(chunk_texts)
            
            # Prepare data for vector store
            metadata = [chunk["metadata"] for chunk in chunks]
            chunk_ids = [
                f"{Path(file_path).stem}_{chunk['metadata']['chunk_index']}"
                for chunk in chunks
            ]
            
            # Store metadata separately (Endee doesn't support metadata in vectors)
            self._store_metadata(collection_name, chunk_ids, chunks)
            
            # Store in vector database
            success = self.vector_store.add_vectors(
                collection_name,
                embeddings,
                metadata,  # Will be ignored by Endee
                chunk_ids
            )
            
            if success:
                result = {
                    "status": "success",
                    "filename": Path(file_path).name,
                    "chunks_created": len(chunks),
                    "collection": collection_name,
                    "document_metadata": chunks[0]["metadata"] if chunks else {}
                }
                logger.info(f"Successfully ingested document: {result}")
                return result
            else:
                raise Exception("Failed to store vectors")
                
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "filename": Path(file_path).name if file_path else "unknown"
            }
    
    def query(self, question: str, collection_name: str = None, 
              top_k: int = None) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        collection_name = collection_name or Config.DEFAULT_COLLECTION
        top_k = top_k or Config.TOP_K_DOCUMENTS
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Generate embedding for the question
            query_embedding = self.embedding_service.encode_text(question)
            
            # Search for relevant documents
            search_results = self.vector_store.search_vectors(
                collection_name,
                query_embedding,
                top_k
            )
            
            if not search_results:
                return {
                    "status": "no_results",
                    "question": question,
                    "answer": "I couldn't find any relevant documents to answer your question.",
                    "sources": []
                }
            
            # Extract chunk IDs from search results
            chunk_ids = []
            scores = {}
            
            # Handle Endee's response format: [distance, id, vector, filter, ?, ?]
            for result in search_results:
                if isinstance(result, list) and len(result) >= 2:
                    distance = result[0]  # Cosine distance
                    chunk_id = result[1]  # Vector ID
                    
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                        # Convert cosine distance to similarity score
                        similarity_score = 1.0 - distance
                        scores[chunk_id] = similarity_score
            
            if not chunk_ids:
                return {
                    "status": "no_results",
                    "question": question,
                    "answer": "I couldn't find any relevant documents to answer your question.",
                    "sources": []
                }
            
            # Get metadata for found chunks
            metadata_map = self._get_metadata(collection_name, chunk_ids)
            
            # Extract context from metadata
            contexts = []
            sources = []
            
            for chunk_id in chunk_ids:
                chunk_data = metadata_map.get(chunk_id, {})
                score = scores.get(chunk_id, 0)
                
                # Skip results below similarity threshold
                if score < Config.SIMILARITY_THRESHOLD:
                    continue
                
                context_text = chunk_data.get("text", "")
                chunk_metadata = chunk_data.get("metadata", {})
                
                if context_text:
                    contexts.append(context_text)
                    sources.append({
                        "filename": chunk_metadata.get("filename", "unknown"),
                        "chunk_index": chunk_metadata.get("chunk_index", 0),
                        "score": score
                    })
            
            if not contexts:
                return {
                    "status": "low_relevance",
                    "question": question,
                    "answer": "I found some documents, but they don't seem relevant enough to answer your question.",
                    "sources": []
                }
            
            # Generate answer using retrieved context
            if self.use_ai:
                answer = self.answer_generator.generate_answer(question, contexts)
            else:
                answer = self.answer_generator.generate_answer(question, contexts)
                answer += "\n\n📝 Note: Using simple text extraction. Install Ollama for AI-powered answers."
            
            result = {
                "status": "success",
                "question": question,
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }
            
            logger.info(f"Query completed successfully: {len(sources)} sources used")
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "status": "error",
                "question": question,
                "error": str(e),
                "answer": "Sorry, I encountered an error while processing your question.",
                "sources": []
            }
    
    def list_documents(self, collection_name: str = None) -> List[Dict]:
        """List all documents in a collection."""
        collection_name = collection_name or Config.DEFAULT_COLLECTION
        
        try:
            stats = self.vector_store.get_collection_stats(collection_name)
            return {
                "collection": collection_name,
                "stats": stats,
                "documents": []  # Endee may not provide document listing
            }
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {"collection": collection_name, "error": str(e)}
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its documents."""
        return self.vector_store.delete_collection(collection_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        return {
            "endee_status": "connected" if self.vector_store.health_check() else "disconnected",
            "collections": self.vector_store.list_collections(),
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.get_dimension(),
            "default_collection": Config.DEFAULT_COLLECTION
        }