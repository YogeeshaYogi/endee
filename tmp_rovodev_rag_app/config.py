"""Configuration settings for the RAG application."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Endee Vector Database Settings
    ENDEE_HOST = os.getenv("ENDEE_HOST", "localhost")
    ENDEE_PORT = os.getenv("ENDEE_PORT", "8080")
    ENDEE_BASE_URL = f"http://{ENDEE_HOST}:{ENDEE_PORT}"
    ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
    
    # Embedding Model Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    
    # OpenAI Settings (for answer generation)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_DOCUMENT_SIZE_MB = 10
    
    # Retrieval Settings
    TOP_K_DOCUMENTS = 5
    SIMILARITY_THRESHOLD = 0.3
    
    # Collection/Index Names
    DEFAULT_COLLECTION = "documents"
    
    # File Upload Settings
    UPLOAD_DIR = "uploads"
    ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set. Answer generation will not work.")
            return False
        return True