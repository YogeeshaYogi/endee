"""Document processing and chunking utilities."""

import os
import hashlib
from typing import List, Dict, Any
import logging
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

import tiktoken
from config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, chunking, and preprocessing."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a document and extract text content."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > Config.MAX_DOCUMENT_SIZE_MB:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {Config.MAX_DOCUMENT_SIZE_MB}MB")
        
        # Extract text based on file type
        extension = file_path.suffix.lower()
        
        if extension == ".txt" or extension == ".md":
            text = self._load_text_file(file_path)
        elif extension == ".pdf":
            text = self._load_pdf_file(file_path)
        elif extension == ".docx":
            text = self._load_docx_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Generate document metadata
        doc_hash = hashlib.md5(text.encode()).hexdigest()
        
        return {
            "filename": file_path.name,
            "filepath": str(file_path),
            "extension": extension,
            "text": text,
            "hash": doc_hash,
            "size_chars": len(text),
            "size_tokens": len(self.tokenizer.encode(text))
        }
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file: {file_path}")
    
    def _load_pdf_file(self, file_path: Path) -> str:
        """Load PDF file and extract text."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
        
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")
        
        return text.strip()
    
    def _load_docx_file(self, file_path: Path) -> str:
        """Load DOCX file and extract text."""
        if docx is None:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            raise ValueError(f"Failed to read DOCX: {e}")
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap."""
        if metadata is None:
            metadata = {}
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "start_token": start_idx,
                "end_token": end_idx,
                "chunk_size": len(chunk_tokens),
                "chunk_chars": len(chunk_text)
            }
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            # Move to next chunk with overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - self.chunk_overlap
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Complete document processing pipeline."""
        # Load document
        doc_data = self.load_document(file_path)
        
        # Create base metadata
        base_metadata = {
            "filename": doc_data["filename"],
            "filepath": doc_data["filepath"],
            "extension": doc_data["extension"],
            "doc_hash": doc_data["hash"],
            "total_chars": doc_data["size_chars"],
            "total_tokens": doc_data["size_tokens"]
        }
        
        # Chunk the document
        chunks = self.chunk_text(doc_data["text"], base_metadata)
        
        logger.info(f"Processed document '{doc_data['filename']}' into {len(chunks)} chunks")
        return chunks