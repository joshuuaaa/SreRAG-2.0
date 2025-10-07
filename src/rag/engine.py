#!/usr/bin/env python3
from typing import Any, Dict, List, Optional

try:
    # Use your existing retriever
    from .retriever import RAGRetriever
except Exception as e:
    raise RuntimeError(f"Failed to import RAGRetriever from src.rag.retriever: {e}")

class RAGEngine:
    """
    Wrapper around RAGRetriever so main.py can import src.rag.engine.RAGEngine.
    Provides:
      - retrieve(query) -> List[dict] with keys: text, metadata{source,...}, score
      - format_context(results) -> str
      - get_stats() -> dict
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.retriever = RAGRetriever(self.cfg)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Calls the underlying retriever and normalizes output so main.py
        can print Sources: ... with scores and source paths.
        """
        results = self.retriever.retrieve(query)

        # If your retriever already returns the desired schema, just return it.
        # Otherwise, adapt common shapes below.
        normalized: List[Dict[str, Any]] = []
        for r in results:
            # Common possibilities:
            # - {'text': ..., 'score': ..., 'source': ...}
            # - {'chunk': {'text': ...}, 'score': ..., 'metadata': {...}}
            # - {'text': ..., 'metadata': {...}}
            text = None
            score = r.get("score")
            source = None

            if "text" in r and isinstance(r["text"], str):
                text = r["text"]
            elif "chunk" in r and isinstance(r["chunk"], dict):
                text = r["chunk"].get("text")

            # metadata/source
            md = r.get("metadata") or {}
            if isinstance(md, dict):
                source = md.get("source") or md.get("path") or md.get("file")

            # Some retrievers place source at top level
            if not source:
                source = r.get("source") or "unknown"

            if text is None:
                # As a last resort, try to stringify
                text = str(r)

            normalized.append({
                "text": text,
                "metadata": {"source": source},
                "score": score if isinstance(score, (int, float)) else None,
            })

        return normalized

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Prefer the retriever’s own formatter if available; otherwise, build a simple citations block.
        """
        # Use RAGRetriever.format_context if it exists
        fmt = getattr(self.retriever, "format_context", None)
        if callable(fmt):
            try:
                return fmt(results)
            except Exception:
                pass

        # Fallback: build a simple citations block
        blocks = []
        for i, r in enumerate(results, 1):
            src = (r.get("metadata") or {}).get("source", f"Doc {i}")
            txt = (r.get("text") or "").strip()
            blocks.append(f"[{src}]\n{txt}")
        return "\n\n".join(blocks)

    def get_stats(self) -> Dict[str, Any]:
        """
        Best-effort stats for the startup banner.
        """
        stats: Dict[str, Any] = {}
        # Try to read common attributes from your retriever
        for attr, key in [
            ("embedding_model_name", "model"),
            ("embedding_model", "model"),
            ("index_path", "index_path"),
            ("dim", "embedding_dimension"),
        ]:
            if hasattr(self.retriever, attr):
                stats[key] = getattr(self.retriever, attr)

        # Count loaded docs if available
        total = None
        for attr in ("metadata", "documents", "docs", "chunks"):
            if hasattr(self.retriever, attr):
                obj = getattr(self.retriever, attr)
                try:
                    total = len(obj)
                    break
                except Exception:
                    pass
        if total is not None:
            stats["total_documents"] = total

        return stats