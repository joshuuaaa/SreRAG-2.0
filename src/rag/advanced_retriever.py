"""Advanced RAG retriever with hybrid search, reranking, and query expansion"""

import os
import pickle
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
import torch

@dataclass
class RetrievalResult:
    """Structured retrieval result"""
    text: str
    score: float
    metadata: Dict[str, Any]
    source: str
    chunk_id: str
    relevance_type: str  # 'semantic', 'keyword', 'hybrid'

class QueryProcessor:
    """Process and expand queries for better retrieval"""
    
    def __init__(self):
        self.medical_synonyms = {
            'bleeding': ['hemorrhage', 'blood loss', 'hemorrhaging'],
            'heart attack': ['myocardial infarction', 'MI', 'cardiac arrest'],
            'stroke': ['cerebrovascular accident', 'CVA', 'brain attack'],
            'seizure': ['convulsion', 'fit', 'epileptic episode'],
            'breathing': ['respiratory', 'ventilation', 'pulmonary'],
            'choking': ['airway obstruction', 'foreign body'],
            'unconscious': ['unresponsive', 'comatose', 'loss of consciousness'],
            'burn': ['thermal injury', 'scald', 'flame injury'],
            'fracture': ['broken bone', 'break', 'bone injury'],
            'poison': ['toxin', 'overdose', 'intoxication']
        }
        
        self.emergency_terms = {
            'severe', 'critical', 'emergency', 'urgent', 'life-threatening',
            'massive', 'heavy', 'profuse', 'uncontrolled', 'severe'
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with medical synonyms and related terms"""
        expanded_queries = [query.lower()]
        
        # Add synonym expansions
        for term, synonyms in self.medical_synonyms.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded_queries.append(query.lower().replace(term, synonym))
        
        # Extract key medical terms
        words = re.findall(r'\b\w+\b', query.lower())
        medical_words = [w for w in words if len(w) > 3]
        if medical_words:
            expanded_queries.append(' '.join(medical_words))
        
        return list(set(expanded_queries))
    
    def extract_emergency_indicators(self, query: str) -> Dict[str, Any]:
        """Extract emergency severity and context indicators"""
        query_lower = query.lower()
        
        # Severity indicators
        severity_terms = {
            'mild': ['mild', 'light', 'small', 'minor'],
            'moderate': ['moderate', 'medium'],
            'severe': ['severe', 'heavy', 'major', 'massive', 'profuse', 'uncontrolled']
        }
        
        severity = 'unknown'
        for level, terms in severity_terms.items():
            if any(term in query_lower for term in terms):
                severity = level
                break
        
        # Body system/location
        body_systems = {
            'cardiac': ['heart', 'cardiac', 'chest pain', 'palpitation'],
            'respiratory': ['breathing', 'lung', 'airway', 'cough', 'shortness'],
            'neurological': ['head', 'brain', 'seizure', 'stroke', 'consciousness'],
            'trauma': ['injury', 'wound', 'cut', 'break', 'fracture', 'burn'],
            'bleeding': ['bleeding', 'blood', 'hemorrhage'],
            'poisoning': ['poison', 'overdose', 'toxic', 'ingestion']
        }
        
        detected_systems = []
        for system, terms in body_systems.items():
            if any(term in query_lower for term in terms):
                detected_systems.append(system)
        
        return {
            'severity': severity,
            'body_systems': detected_systems,
            'has_emergency_terms': any(term in query_lower for term in self.emergency_terms)
        }

class HybridRetriever:
    """Hybrid retriever combining dense and sparse search with reranking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_path = config['index_path']
        self.embedding_model_name = config['embedding_model']
        self.top_k = config.get('top_k', 5)
        self.candidate_k = config.get('candidate_k', 20)
        self.similarity_threshold = config.get('similarity_threshold', 0.45)
        self.use_reranker = config.get('reranker', {}).get('enabled', True)
        self.reranker_model_name = config.get('reranker', {}).get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Models
        self.embedding_model = None
        self.reranker_model = None
        self.query_processor = QueryProcessor()
        
        # Indexes
        self.dense_index = None  # FAISS
        self.sparse_index = None  # BM25
        
        # Documents
        self.documents = []
        self.metadata = []
        self.processed_docs = []  # For BM25
        
        self._load_models()
        self._load_indexes()
    
    def _load_models(self):
        """Load embedding and reranking models"""
        print(f"🔄 Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("✅ Embedding model loaded!")
        
        if self.use_reranker:
            print(f"🔄 Loading reranker: {self.reranker_model_name}...")
            self.reranker_model = CrossEncoder(self.reranker_model_name)
            print("✅ Reranker loaded!")
    
    def _load_indexes(self):
        """Load FAISS and BM25 indexes"""
        # FAISS index
        faiss_path = os.path.join(self.index_path, 'faiss.index')
        docs_path = os.path.join(self.index_path, 'documents.pkl')
        meta_path = os.path.join(self.index_path, 'metadata.pkl')
        bm25_path = os.path.join(self.index_path, 'bm25.pkl')
        
        if not os.path.exists(faiss_path):
            print("⚠️  No FAISS index found. Building from documents...")
            self._build_indexes()
            return
        
        print("🔄 Loading indexes...")
        
        # Load FAISS
        self.dense_index = faiss.read_index(faiss_path)
        
        # Load documents and metadata
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load BM25
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
                self.sparse_index = bm25_data['index']
                self.processed_docs = bm25_data['processed_docs']
        else:
            print("Building BM25 index...")
            self._build_bm25_index()
        
        print(f"✅ Loaded {len(self.documents)} documents with hybrid search")
    
    def _build_indexes(self):
        """Build indexes from documents"""
        # This would normally build from the documents in data/manuals/
        # For now, create empty indexes
        dimension = 384  # BGE-small dimension
        self.dense_index = faiss.IndexFlatIP(dimension)
        self.documents = []
        self.metadata = []
        self.processed_docs = []
        self.sparse_index = None
        print("⚠️  Empty indexes created. Run indexing script to populate.")
    
    def _build_bm25_index(self):
        """Build BM25 index from documents"""
        if not self.documents:
            return
        
        # Tokenize documents for BM25
        self.processed_docs = []
        for doc in self.documents:
            # Simple tokenization (can be improved)
            tokens = re.findall(r'\b\w+\b', doc.lower())
            self.processed_docs.append(tokens)
        
        # Build BM25 index
        self.sparse_index = BM25Okapi(self.processed_docs)
        
        # Save BM25 index
        bm25_path = os.path.join(self.index_path, 'bm25.pkl')
        os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'index': self.sparse_index,
                'processed_docs': self.processed_docs
            }, f)
    
    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Semantic search using embeddings"""
        if self.dense_index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.dense_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= self.similarity_threshold:
                results.append((int(idx), float(score)))
        
        return results
    
    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Keyword search using BM25"""
        if not self.sparse_index:
            return []
        
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        if not query_tokens:
            return []
        
        scores = self.sparse_index.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Only positive scores
                results.append((int(idx), float(score)))
        
        return results
    
    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[int, float]], 
                               sparse_results: List[Tuple[int, float]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """Combine dense and sparse results using RRF"""
        
        # Create rank maps
        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_results)}
        
        # Calculate RRF scores
        all_indices = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        rrf_scores = {}
        
        for idx in all_indices:
            score = 0
            if idx in dense_ranks:
                score += 1 / (k + dense_ranks[idx])
            if idx in sparse_ranks:
                score += 1 / (k + sparse_ranks[idx])
            rrf_scores[idx] = score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_results]
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        if not self.use_reranker or not self.reranker_model or len(results) <= 1:
            return results
        
        # Prepare query-document pairs
        pairs = [(query, result.text) for result in results]
        
        # Get reranking scores
        scores = self.reranker_model.predict(pairs)
        
        # Update results with new scores
        for result, score in zip(results, scores):
            result.score = float(score)
            result.relevance_type = 'reranked'
        
        # Sort by reranking score
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Main retrieval method with hybrid search"""
        k = top_k or self.top_k
        candidate_k = max(self.candidate_k, k * 2)
        
        # Process query
        emergency_context = self.query_processor.extract_emergency_indicators(query)
        expanded_queries = self.query_processor.expand_query(query)
        
        # Search with main query and expansions
        all_dense_results = []
        all_sparse_results = []
        
        for q in expanded_queries[:3]:  # Limit to top 3 expansions
            dense_results = self._dense_search(q, candidate_k // len(expanded_queries))
            sparse_results = self._sparse_search(q, candidate_k // len(expanded_queries))
            
            all_dense_results.extend(dense_results)
            all_sparse_results.extend(sparse_results)
        
        # Remove duplicates while preserving best scores
        dense_dict = {}
        for idx, score in all_dense_results:
            if idx not in dense_dict or score > dense_dict[idx]:
                dense_dict[idx] = score
        
        sparse_dict = {}
        for idx, score in all_sparse_results:
            if idx not in sparse_dict or score > sparse_dict[idx]:
                sparse_dict[idx] = score
        
        dense_results = list(dense_dict.items())
        sparse_results = list(sparse_dict.items())
        
        # Fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Convert to RetrievalResult objects
        retrieval_results = []
        for idx, score in fused_results[:candidate_k]:
            if idx < len(self.documents):
                result = RetrievalResult(
                    text=self.documents[idx],
                    score=score,
                    metadata=self.metadata[idx],
                    source=self.metadata[idx].get('source', 'Unknown'),
                    chunk_id=f"chunk_{idx}",
                    relevance_type='hybrid'
                )
                retrieval_results.append(result)
        
        # Rerank if enabled
        if self.use_reranker and len(retrieval_results) > 1:
            retrieval_results = self._rerank_results(query, retrieval_results)
        
        # Apply emergency prioritization
        prioritized_results = self._prioritize_emergency_content(
            retrieval_results, emergency_context
        )
        
        return prioritized_results[:k]
    
    def _prioritize_emergency_content(self, results: List[RetrievalResult], 
                                    emergency_context: Dict[str, Any]) -> List[RetrievalResult]:
        """Prioritize results based on emergency context"""
        if not emergency_context.get('has_emergency_terms'):
            return results
        
        # Boost scores for emergency-related content
        priority_terms = {
            'severe': 1.3,
            'critical': 1.3,
            'emergency': 1.2,
            'life-threatening': 1.4,
            'immediate': 1.2,
            'urgent': 1.2
        }
        
        for result in results:
            text_lower = result.text.lower()
            boost_factor = 1.0
            
            for term, factor in priority_terms.items():
                if term in text_lower:
                    boost_factor = max(boost_factor, factor)
            
            # Boost based on detected body systems
            for system in emergency_context.get('body_systems', []):
                if system in text_lower:
                    boost_factor *= 1.1
            
            result.score *= boost_factor
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved results into context string"""
        if not results:
            return "No relevant medical information found in knowledge base."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            # Add relevance indicator
            relevance_indicator = "🔍" if result.relevance_type == "keyword" else "🧠"
            if result.relevance_type == "reranked":
                relevance_indicator = "⭐"
            
            context_parts.append(
                f"[Source {i}: {result.source}] {relevance_indicator}\n{result.text}"
            )
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': 384,  # BGE-small
            'model': self.embedding_model_name,
            'reranker_enabled': self.use_reranker,
            'hybrid_search': True,
            'index_path': self.index_path
        }

# Legacy alias for backward compatibility
RAGRetriever = HybridRetriever