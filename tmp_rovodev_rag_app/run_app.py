"""Main application runner with CLI interface."""

import argparse
import os
import sys
import logging
from pathlib import Path

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('rag_app.log')
        ]
    )

def run_streamlit():
    """Run the Streamlit web interface."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "web_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def run_api():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

def test_system():
    """Test the RAG system components."""
    from rag_pipeline import RAGPipeline
    from config import Config
    
    print("🔧 Testing RAG System Components...")
    
    try:
        # Test configuration
        print("✅ Configuration loaded")
        
        # Test RAG pipeline initialization
        rag = RAGPipeline()
        print("✅ RAG pipeline initialized")
        
        # Test system status
        status = rag.get_system_status()
        print(f"✅ System status: {status['endee_status']}")
        print(f"   Collections: {status['collections']}")
        print(f"   Embedding model: {status['embedding_model']}")
        
        # Test embedding service
        test_text = "This is a test document."
        embedding = rag.embedding_service.encode_text(test_text)
        print(f"✅ Embedding generation works (dimension: {len(embedding)})")
        
        print("\n🎉 All tests passed! RAG system is ready.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

def process_document(file_path: str, collection: str = None):
    """Process a single document."""
    from rag_pipeline import RAGPipeline
    
    try:
        rag = RAGPipeline()
        result = rag.ingest_document(file_path, collection)
        
        if result["status"] == "success":
            print(f"✅ Document processed successfully:")
            print(f"   File: {result['filename']}")
            print(f"   Chunks: {result['chunks_created']}")
            print(f"   Collection: {result['collection']}")
        else:
            print(f"❌ Document processing failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def query_documents(question: str, collection: str = None):
    """Query the document collection."""
    from rag_pipeline import RAGPipeline
    
    try:
        rag = RAGPipeline()
        result = rag.query(question, collection)
        
        print(f"❓ Question: {result['question']}")
        print(f"💡 Answer: {result['answer']}")
        
        if result['sources']:
            print(f"\n📚 Sources ({result['num_sources']}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source['filename']} (score: {source['score']:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="RAG Document Q&A System")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Web interface
    subparsers.add_parser("web", help="Run Streamlit web interface")
    
    # API server
    subparsers.add_parser("api", help="Run FastAPI server")
    
    # Test system
    subparsers.add_parser("test", help="Test system components")
    
    # Process document
    doc_parser = subparsers.add_parser("ingest", help="Process a document")
    doc_parser.add_argument("file", help="Path to document file")
    doc_parser.add_argument("--collection", help="Collection name")
    
    # Query documents
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--collection", help="Collection name")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.command == "web":
        print("🌐 Starting Streamlit web interface...")
        print("   URL: http://localhost:8501")
        run_streamlit()
        
    elif args.command == "api":
        print("🚀 Starting FastAPI server...")
        print("   URL: http://localhost:8000")
        print("   Docs: http://localhost:8000/docs")
        run_api()
        
    elif args.command == "test":
        test_system()
        
    elif args.command == "ingest":
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        process_document(args.file, args.collection)
        
    elif args.command == "query":
        query_documents(args.question, args.collection)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()