#!/usr/bin/env python3
"""Enhanced indexing script for medical emergency documents"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Missing dependencies: {e}")
    print("Install with: pip install sentence-transformers faiss-cpu")
    DEPENDENCIES_AVAILABLE = False

from src.utils import load_config
from src.rag.document_processor import process_medical_documents, DocumentChunk

def build_enhanced_index(config: Dict[str, Any]) -> None:
    """Build enhanced FAISS index with hybrid search support"""
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Cannot build index without required dependencies")
        return
    
    rag_config = config.get('rag', {})
    docs_dir = Path("data/manuals")
    index_dir = Path(rag_config.get('index_path', 'data/index'))
    
    # Create index directory
    index_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏗️  Building Enhanced RAG Index")
    print("=" * 50)
    
    # Step 1: Process documents
    print("\n📄 Processing Documents...")
    chunks = process_medical_documents(str(docs_dir), rag_config)
    
    if not chunks:
        print("❌ No documents found to index")
        return
    
    # Step 2: Load embedding model
    embedding_model_name = rag_config.get('embedding_model', 'BAAI/bge-small-en-v1.5')
    print(f"\n🔤 Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Step 3: Generate embeddings
    print("\n🧮 Generating embeddings...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.encode(
        texts, 
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"   Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Step 4: Build FAISS index
    print("\n🔍 Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for cosine similarity (normalized embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"   FAISS index built with {index.ntotal} vectors")
    
    # Step 5: Prepare documents and metadata for storage
    print("\n💾 Preparing document storage...")
    documents = []
    metadata = []
    
    for chunk in chunks:
        documents.append(chunk.text)
        
        # Create comprehensive metadata
        chunk_metadata = {
            'source': chunk.source,
            'chunk_id': chunk.chunk_id,
            'emergency_type': chunk.emergency_type,
            'severity_level': chunk.severity_level,
            'section': chunk.section,
            'keywords': chunk.keywords,
            **chunk.metadata
        }
        metadata.append(chunk_metadata)
    
    # Step 6: Build BM25 index for hybrid search
    print("\n🔎 Building BM25 index for hybrid search...")
    try:
        from rank_bm25 import BM25Okapi
        import re
        
        # Tokenize documents
        tokenized_docs = []
        for text in documents:
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokenized_docs.append(tokens)
        
        bm25_index = BM25Okapi(tokenized_docs)
        
        # Save BM25 index
        bm25_path = index_dir / 'bm25.pkl'
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'index': bm25_index,
                'processed_docs': tokenized_docs
            }, f)
        
        print(f"   BM25 index saved to {bm25_path}")
        
    except ImportError:
        print("   ⚠️  BM25 unavailable (install rank_bm25), skipping hybrid search")
    
    # Step 7: Save all indexes and data
    print("\n💾 Saving indexes...")
    
    # Save FAISS index
    faiss_path = index_dir / 'faiss.index'
    faiss.write_index(index, str(faiss_path))
    print(f"   FAISS index saved to {faiss_path}")
    
    # Save documents
    docs_path = index_dir / 'documents.pkl'
    with open(docs_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"   Documents saved to {docs_path}")
    
    # Save metadata
    meta_path = index_dir / 'metadata.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   Metadata saved to {meta_path}")
    
    # Step 8: Save index statistics
    stats = {
        'total_documents': len(documents),
        'total_chunks': len(chunks),
        'embedding_model': embedding_model_name,
        'embedding_dimension': dimension,
        'index_path': str(index_dir),
        'emergency_types': {},
        'severity_levels': {},
        'section_types': {}
    }
    
    # Calculate type distributions
    for chunk in chunks:
        etype = chunk.emergency_type
        stats['emergency_types'][etype] = stats['emergency_types'].get(etype, 0) + 1
        
        severity = chunk.severity_level
        stats['severity_levels'][severity] = stats['severity_levels'].get(severity, 0) + 1
        
        section = chunk.section
        stats['section_types'][section] = stats['section_types'].get(section, 0) + 1
    
    stats_path = index_dir / 'index_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    # Step 9: Display final statistics
    print("\n📊 Index Statistics:")
    print("-" * 30)
    print(f"Total chunks: {stats['total_documents']}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Index size: {index.ntotal} vectors")
    
    print("\nEmergency types:")
    for etype, count in sorted(stats['emergency_types'].items()):
        print(f"  {etype}: {count}")
    
    print("\nSeverity levels:")
    for severity, count in sorted(stats['severity_levels'].items()):
        print(f"  {severity}: {count}")
    
    print("\nSection types:")
    for section, count in sorted(stats['section_types'].items()):
        print(f"  {section}: {count}")
    
    print(f"\n✅ Enhanced index built successfully!")
    print(f"📁 Index location: {index_dir}")

def test_index(config: Dict[str, Any]) -> None:
    """Test the built index with sample queries"""
    
    if not DEPENDENCIES_AVAILABLE:
        return
    
    print("\n🧪 Testing Index...")
    print("-" * 30)
    
    try:
        from src.rag.engine import RAGEngine
        
        rag = RAGEngine(config.get('rag', {}))
        
        test_queries = [
            "severe bleeding from arm",
            "heart attack symptoms",
            "choking emergency",
            "unconscious person",
            "seizure first aid"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = rag.retrieve(query)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results[:2], 1):
                    source = result.get('metadata', {}).get('source', 'Unknown')
                    score = result.get('score', 0)
                    text_preview = result.get('text', '')[:100] + "..."
                    print(f"  {i}. {source} (score: {score:.3f})")
                    print(f"     {text_preview}")
            else:
                print("  No results found")
        
        print("\n✅ Index test completed!")
        
    except Exception as e:
        print(f"❌ Index test failed: {e}")

def main():
    """Main indexing function"""
    
    print("🚑 Crisis Assistant - Enhanced Index Builder")
    print("=" * 60)
    
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    # Build index
    build_enhanced_index(config)
    
    # Test index
    test_index(config)
    
    print("\n🎉 Indexing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())