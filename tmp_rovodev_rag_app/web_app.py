"""Streamlit web interface for the RAG application."""

import streamlit as st
import os
import tempfile
from pathlib import Path
import logging
from typing import Dict, Any

from rag_pipeline import RAGPipeline
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached for performance)."""
    try:
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def main():
    st.title("🤖 RAG Document Q&A System")
    st.markdown("**Powered by Endee Vector Database**")
    
    # Initialize RAG system
    rag = initialize_rag_system()
    if rag is None:
        st.error("❌ Cannot connect to the RAG system. Please ensure Endee is running.")
        st.info("Run `docker-compose up` to start Endee vector database.")
        return
    
    # Sidebar for system status and configuration
    with st.sidebar:
        st.header("⚙️ System Status")
        
        if st.button("🔄 Refresh Status"):
            st.cache_resource.clear()
            st.experimental_rerun()
        
        try:
            status = rag.get_system_status()
            
            # Connection status
            if status["endee_status"] == "connected":
                st.success("✅ Endee Connected")
            else:
                st.error("❌ Endee Disconnected")
            
            # System info
            st.info(f"**Model:** {status['embedding_model']}")
            st.info(f"**Dimensions:** {status['embedding_dimension']}")
            st.info(f"**Collections:** {len(status['collections'])}")
            
            # Collections
            if status['collections']:
                st.subheader("📁 Collections")
                for collection in status['collections']:
                    st.text(f"• {collection}")
        
        except Exception as e:
            st.error(f"Status check failed: {e}")
        
        # Configuration
        st.header("🔧 Configuration")
        st.text(f"OpenAI API: {'✅' if Config.OPENAI_API_KEY else '❌'}")
        st.text(f"Collection: {Config.DEFAULT_COLLECTION}")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "❓ Ask Questions", "📊 System Info"])
    
    with tab1:
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, Markdown"
        )
        
        # Collection selector
        collection_name = st.text_input(
            "Collection Name", 
            value=Config.DEFAULT_COLLECTION,
            help="Name of the collection to store documents"
        )
        
        if uploaded_files and st.button("🚀 Process Documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")
                
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Process document
                    result = rag.ingest_document(tmp_path, collection_name)
                    results.append(result)
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    results.append({
                        "status": "error",
                        "filename": uploaded_file.name,
                        "error": str(e)
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Show results
            status_text.text("Processing complete!")
            
            for result in results:
                if result["status"] == "success":
                    st.success(f"✅ {result['filename']}: {result['chunks_created']} chunks created")
                else:
                    st.error(f"❌ {result['filename']}: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("❓ Ask Questions")
        
        # Question input
        question = st.text_area(
            "What would you like to know?",
            placeholder="Enter your question about the uploaded documents...",
            height=100
        )
        
        # Query settings
        col1, col2 = st.columns(2)
        with col1:
            collection = st.selectbox(
                "Collection",
                options=[Config.DEFAULT_COLLECTION],  # Could be expanded to show all collections
                index=0
            )
        with col2:
            top_k = st.slider("Number of sources", min_value=1, max_value=10, value=Config.TOP_K_DOCUMENTS)
        
        if question and st.button("🔍 Get Answer"):
            with st.spinner("Searching documents and generating answer..."):
                result = rag.query(question, collection, top_k)
            
            if result["status"] == "success":
                st.subheader("💡 Answer")
                st.write(result["answer"])
                
                if result["sources"]:
                    st.subheader(f"📚 Sources ({result['num_sources']})")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"Source {i}: {source['filename']} (Score: {source['score']:.3f})"):
                            st.text(f"Chunk: {source['chunk_index']}")
                            st.text(f"Relevance Score: {source['score']:.3f}")
                
            elif result["status"] == "no_results":
                st.warning("🔍 No relevant documents found for your question.")
                
            elif result["status"] == "low_relevance":
                st.warning("📄 Found documents but they don't seem relevant enough.")
                
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    with tab3:
        st.header("📊 System Information")
        
        try:
            status = rag.get_system_status()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔌 Connection Status")
                st.json({
                    "Endee Status": status["endee_status"],
                    "Collections": status["collections"]
                })
                
                st.subheader("🤖 AI Models")
                st.json({
                    "Embedding Model": status["embedding_model"],
                    "Embedding Dimension": status["embedding_dimension"],
                    "OpenAI Model": Config.OPENAI_MODEL if Config.OPENAI_API_KEY else "Not configured"
                })
            
            with col2:
                st.subheader("⚙️ Configuration")
                st.json({
                    "Default Collection": status["default_collection"],
                    "Chunk Size": Config.CHUNK_SIZE,
                    "Chunk Overlap": Config.CHUNK_OVERLAP,
                    "Top K Documents": Config.TOP_K_DOCUMENTS,
                    "Similarity Threshold": Config.SIMILARITY_THRESHOLD
                })
        
        except Exception as e:
            st.error(f"Failed to load system information: {e}")
        
        # Collection management
        st.subheader("🗂️ Collection Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Refresh Collections"):
                st.experimental_rerun()
        
        with col2:
            new_collection = st.text_input("Create New Collection")
            if new_collection and st.button("➕ Create"):
                try:
                    success = rag.vector_store.create_collection(
                        new_collection,
                        rag.embedding_service.get_dimension()
                    )
                    if success:
                        st.success(f"✅ Collection '{new_collection}' created!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Failed to create collection")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()