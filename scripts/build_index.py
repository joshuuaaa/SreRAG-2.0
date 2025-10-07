"""Build FAISS index from medical documents"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

def load_documents(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load text documents from directory
    
    Args:
        input_dir: Directory containing .txt files
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    for file_path in Path(input_dir).glob('**/*.txt'):
        print(f"📄 Loading {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents.append({
            'text': content,
            'source': file_path.name,
            'path': str(file_path)
        })
    
    return documents

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if len(c) > 50]  # Filter tiny chunks

def build_index(input_dir: str, output_dir: str, embedding_model: str = 'BAAI/bge-small-en-v1.5'):
    """
    Build FAISS index from documents
    
    Args:
        input_dir: Directory with .txt files
        output_dir: Where to save index
        embedding_model: Sentence transformer model name
    """
    print("🚀 Starting RAG index build...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load documents
    print("\n📚 Loading documents...")
    docs = load_documents(input_dir)
    
    if not docs:
        print("❌ No .txt files found in input directory!")
        return
    
    print(f"✅ Loaded {len(docs)} documents")
    
    # Chunk documents
    print("\n✂️  Chunking documents...")
    chunks = []
    metadata = []
    
    for doc in docs:
        doc_chunks = chunk_text(doc['text'])
        chunks.extend(doc_chunks)
        
        for chunk in doc_chunks:
            metadata.append({
                'source': doc['source'],
                'path': doc['path']
            })
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Load embedding model
    print(f"\n🤖 Loading embedding model: {embedding_model}...")
    model = SentenceTransformer(embedding_model)
    print("✅ Model loaded!")
    
    # Generate embeddings
    print("\n🔢 Generating embeddings...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )
    
    # Build FAISS index
    print("\n🗂️  Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Save index
    index_file = os.path.join(output_dir, 'faiss.index')
    faiss.write_index(index, index_file)
    print(f"✅ Saved index to {index_file}")
    
    # Save documents
    docs_file = os.path.join(output_dir, 'documents.pkl')
    with open(docs_file, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✅ Saved documents to {docs_file}")
    
    # Save metadata
    meta_file = os.path.join(output_dir, 'metadata.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Saved metadata to {meta_file}")
    
    print(f"\n🎉 RAG index built successfully!")
    print(f"   - {len(chunks)} chunks indexed")
    print(f"   - {dimension} dimensions")
    print(f"   - Saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build FAISS RAG index')
    parser.add_argument('--input', default='data/manuals', help='Input directory with .txt files')
    parser.add_argument('--output', default='data/index', help='Output directory for index')
    parser.add_argument('--model', default='BAAI/bge-small-en-v1.5', help='Embedding model')
    
    args = parser.parse_args()
    
    build_index(args.input, args.output, args.model)