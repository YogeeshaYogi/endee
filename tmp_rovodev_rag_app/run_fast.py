"""Fast RAG query runner with speed options."""

import argparse
import sys
import time
from rag_pipeline import RAGPipeline
from simple_text_answer import SimpleTextAnswerer

def main():
    parser = argparse.ArgumentParser(description="Fast RAG Query Tool")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--mode", choices=["fast", "ai", "both"], default="both", 
                       help="Response mode: fast (instant), ai (ollama), or both")
    parser.add_argument("--collection", default="documents", help="Collection name")
    
    args = parser.parse_args()
    
    print(f"❓ Question: {args.question}")
    print(f"⚙️ Mode: {args.mode}")
    print("=" * 60)
    
    # Initialize systems
    rag = RAGPipeline()
    simple_answerer = SimpleTextAnswerer()
    
    # Get search results once
    query_embedding = rag.embedding_service.encode_text(args.question)
    search_results = rag.vector_store.search_vectors(args.collection, query_embedding, 5)
    
    # Extract contexts
    chunk_ids = []
    scores = {}
    for result in search_results:
        if isinstance(result, list) and len(result) >= 2:
            distance = result[0]
            chunk_id = result[1]
            chunk_ids.append(chunk_id)
            scores[chunk_id] = 1.0 - distance
    
    metadata_map = rag._get_metadata(args.collection, chunk_ids)
    contexts = []
    sources = []
    
    for chunk_id in chunk_ids:
        chunk_data = metadata_map.get(chunk_id, {})
        score = scores.get(chunk_id, 0)
        
        if score >= 0.3:  # Similarity threshold
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
        print("❌ No relevant documents found.")
        return
    
    # Fast mode
    if args.mode in ["fast", "both"]:
        print("⚡ FAST MODE (Instant Text Extraction)")
        start = time.time()
        fast_answer = simple_answerer.generate_answer(args.question, contexts)
        fast_time = time.time() - start
        
        print(f"⏱️ Time: {fast_time:.2f}s")
        print(f"💡 Answer: {fast_answer}")
        print(f"📚 Sources: {len(sources)} documents")
        
        if args.mode == "both":
            print("\n" + "=" * 60)
    
    # AI mode
    if args.mode in ["ai", "both"]:
        print("🤖 AI MODE (Ollama Generation)")
        start = time.time()
        ai_answer = rag.answer_generator.generate_answer(args.question, contexts)
        ai_time = time.time() - start
        
        print(f"⏱️ Time: {ai_time:.1f}s")
        print(f"💡 Answer: {ai_answer}")
        print(f"📚 Sources: {len(sources)} documents")
    
    # Show sources
    if sources:
        print(f"\n📋 Source Details:")
        for i, source in enumerate(sources, 1):
            print(f"   {i}. {source['filename']} (chunk {source['chunk_index']}, score: {source['score']:.3f})")

if __name__ == "__main__":
    main()