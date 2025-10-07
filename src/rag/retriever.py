"""RAG retriever using FAISS and sentence transformers"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss

class RAGRetriever:
    """Retrieves relevant documents using semantic search"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG retriever
        
        Args:
            config: RAG configuration dictionary
        """
        self.index_path = config['index_path']
        self.embedding_model_name = config['embedding_model']
        self.top_k = config.get('top_k', 3)
        self.similarity_threshold = config.get('similarity_threshold', 0.5)
        
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._load_embedding_model()
        self._load_index()
    
    def _load_embedding_model(self):
        """Load sentence transformer model"""
        print(f"🔄 Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("✅ Embedding model loaded!")
    
    def _load_index(self):
        """Load FAISS index and documents"""
        index_file = os.path.join(self.index_path, 'faiss.index')
        docs_file = os.path.join(self.index_path, 'documents.pkl')
        meta_file = os.path.join(self.index_path, 'metadata.pkl')
        
        if not os.path.exists(index_file):
            print("⚠️  No index found. Run 'make index' to build RAG index.")
            # Create empty index
            dimension = 384  # BGE-small dimension
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            return
        
        print("🔄 Loading FAISS index...")
        self.index = faiss.read_index(index_file)
        
        with open(docs_file, 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded index with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query
        
        Args:
            query: Search query
            top_k: Number of results (override config)
            
        Returns:
            List of dicts with 'text', 'score', and 'metadata'
        """
        if self.index.ntotal == 0:
            print("⚠️  No documents in index")
            return []
        
        k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            if score < self.similarity_threshold:
                continue
            
            results.append({
                'text': self.documents[idx],
                'score': float(score),
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant medical information found in knowledge base."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            text = result['text']
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        return "\n\n".join(context_parts)