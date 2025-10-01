"""RAG retriever using FAISS for semantic search"""
import os
import pickle
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class RAGRetriever:
    def __init__(self, index_path: str, embedding_model: str, top_k: int = 3):
        self.index_path = index_path
        self.top_k = top_k
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load FAISS index and metadata
        self.index = None
        self.metadata = None
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if not os.path.exists(index_file):
            print(f"⚠️  No index found at {index_file}. Run 'make index' to build it.")
            return
        
        self.index = faiss.read_index(index_file)
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded RAG index: {len(self.metadata['chunks'])} chunks")
    
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search for relevant document chunks"""
        if self.index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, self.top_k * 2)
        
        # Filter and format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No more results
                break
            
            chunk = self.metadata['chunks'][idx]
            
            # Apply filters if provided
            if filters:
                if not self._matches_filters(chunk, filters):
                    continue
            
            results.append({
                'text': chunk['text'],
                'source': chunk['source'],
                'score': float(dist),
                'metadata': chunk.get('metadata', {})
            })
            
            if len(results) >= self.top_k:
                break
        
        return results
    
    def _matches_filters(self, chunk: Dict, filters: Dict) -> bool:
        """Check if chunk matches filter criteria"""
        metadata = chunk.get('metadata', {})
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True