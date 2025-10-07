#!/usr/bin/env python3
"""
Build a FAISS (HNSW) index from medical documents for RAG.

- Sentence-aware chunking with token budget (default 300) and overlap.
- Normalized embeddings for cosine similarity via inner-product.
- Optional FAISS HNSW for faster CPU search (default on).
- Compatible outputs for existing retrievers:
    - faiss.index
    - documents.pkl
    - metadata.pkl
- Additional artifacts:
    - manifest.json (build settings, counts, model, dims)
    - bm25.jsonl (optional) for future hybrid retrieval

Usage:
  python scripts/build_index.py \
      --input data/manuals \
      --output data/index \
      --model BAAI/bge-small-en-v1.5 \
      --chunk_tokens 300 \
      --chunk_overlap 40 \
      --hnsw \
      --M 32 \
      --ef 128 \
      --bm25
"""
import os
import re
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

try:
    from rank_bm25 import BM25Okapi  # optional
except Exception:
    BM25Okapi = None


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _sent_split(text: str) -> List[str]:
    # Lightweight sentence splitter for Pi: split on . ! ? and paragraph breaks
    s = re.sub(r"[ \t]+", " ", text.strip())
    sents = re.split(r"(?<=[.!?])\s+|\n{2,}", s)
    return [x.strip() for x in sents if x.strip()]


def _est_tokens(s: str) -> int:
    # Rough token estimate to avoid heavy tokenizers on Pi
    return max(1, len(s.split()))


def chunk_sentences(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    sents = _sent_split(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            chunks.append(" ".join(cur).strip())
            if overlap_tokens > 0:
                tail = " ".join(" ".join(cur).split()[-overlap_tokens:])
                cur = [tail] if tail else []
                cur_tokens = _est_tokens(tail)
            else:
                cur, cur_tokens = [], 0

    for sent in sents:
        t = _est_tokens(sent)
        if cur_tokens + t > target_tokens and cur:
            flush()
        cur.append(sent)
        cur_tokens += t

    if cur:
        chunks.append(" ".join(cur).strip())

    # Filter tiny or duplicate chunks
    out, seen = [], set()
    for ch in chunks:
        ch_norm = " ".join(ch.split())
        if _est_tokens(ch_norm) < max(20, int(0.15 * target_tokens)):
            continue
        if ch_norm in seen:
            continue
        seen.add(ch_norm)
        out.append(ch_norm)
    return out


def load_documents(input_dir: str) -> List[Dict[str, Any]]:
    paths: List[Path] = []
    for ext in ("*.txt", "*.md"):
        paths.extend(Path(input_dir).glob(f"**/{ext}"))
    paths = sorted(set(paths))
    documents = []
    for p in paths:
        text = _read_text(p)
        documents.append({"text": text, "source": p.name, "path": str(p)})
    return documents


def build_index(
    input_dir: str,
    output_dir: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    chunk_tokens: int = 300,
    chunk_overlap: int = 40,
    normalize: bool = True,
    use_hnsw: bool = True,
    hnsw_M: int = 32,
    hnsw_ef: int = 128,
    build_bm25: bool = False,
):
    print("🚀 Starting RAG index build...")
    os.makedirs(output_dir, exist_ok=True)

    print("\n📚 Loading documents...")
    docs = load_documents(input_dir)
    if not docs:
        raise SystemExit(f"❌ No .txt/.md files found in {input_dir}")
    print(f"✅ Loaded {len(docs)} files")

    print("\n✂️  Chunking documents (sentence-aware)...")
    chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for doc in tqdm(docs, desc="Chunking"):
        doc_chunks = chunk_sentences(doc["text"], chunk_tokens, chunk_overlap)
        chunks.extend(doc_chunks)
        for _ in doc_chunks:
            metadata.append({"source": doc["source"], "path": doc["path"]})
    if not chunks:
        raise SystemExit("❌ No chunks created (check input documents).")
    print(f"✅ Created {len(chunks)} chunks")

    print(f"\n🤖 Loading embedding model: {embedding_model} ...")
    model = SentenceTransformer(embedding_model)
    dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded (dimension={dim})")

    print("\n🔢 Generating embeddings...")
    embs = model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=False,  # we normalize ourselves (optional)
    ).astype("float32")
    if normalize:
        # L2-normalize so inner product == cosine
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
    print("✅ Embeddings ready")

    print("\n🗂️  Building FAISS index...")
    if use_hnsw:
        index = faiss.IndexHNSWFlat(dim, hnsw_M, faiss.METRIC_INNER_PRODUCT)
        # efConstruction controls build accuracy/speed; efSearch is set at query time, but we store a default here
        index.hnsw.efConstruction = max(64, int(hnsw_ef))
        # Store a default efSearch; retriever can override
        index.hnsw.efSearch = max(64, int(hnsw_ef))
        print(f"ℹ️  Using HNSW: M={hnsw_M}, ef={hnsw_ef}")
    else:
        index = faiss.IndexFlatIP(dim)
        print("ℹ️  Using Flat (exact) index (slower on CPU but simplest)")

    index.add(embs)
    faiss_file = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, faiss_file)
    print(f"✅ Saved FAISS index -> {faiss_file}")

    # Save documents/chunks (compat with current retriever)
    docs_file = os.path.join(output_dir, "documents.pkl")
    with open(docs_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Saved chunks -> {docs_file}")

    meta_file = os.path.join(output_dir, "metadata.pkl")
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ Saved metadata -> {meta_file}")

    # Optional BM25 export for future hybrid retrieval
    bm25_file = None
    if build_bm25:
        if BM25Okapi is None:
            print("⚠️  rank_bm25 not installed; skipping BM25 export. pip install rank-bm25")
        else:
            print("\n📑 Preparing BM25 corpus (tokens only, for lightweight load later)...")
            tokenized = [c.lower().split() for c in chunks]
            # We don't need to persist inverted index; just persist tokens to rebuild BM25 quickly at load
            bm25_file = os.path.join(output_dir, "bm25.jsonl")
            with open(bm25_file, "w", encoding="utf-8") as f:
                for toks, md in zip(tokenized, metadata):
                    obj = {"tokens": toks, "source": md["source"], "path": md["path"]}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"✅ Saved BM25 corpus -> {bm25_file}")

    # Manifest with build settings for debugging
    manifest = {
        "embedding_model": embedding_model,
        "dimension": int(dim),
        "normalize_embeddings": bool(normalize),
        "index_type": "HNSWFlat" if use_hnsw else "FlatIP",
        "hnsw": {"M": int(hnsw_M), "ef_default": int(hnsw_ef)} if use_hnsw else None,
        "chunk_tokens": int(chunk_tokens),
        "chunk_overlap": int(chunk_overlap),
        "num_files": len(docs),
        "num_chunks": len(chunks),
        "artifacts": {
            "faiss_index": os.path.abspath(faiss_file),
            "documents": os.path.abspath(docs_file),
            "metadata": os.path.abspath(meta_file),
            "bm25_jsonl": os.path.abspath(bm25_file) if bm25_file else None,
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("🧾 Wrote manifest.json")

    print("\n🎉 RAG index built successfully!")
    print(f"   - {manifest['num_chunks']} chunks")
    print(f"   - {manifest['dimension']} dims | model={embedding_model}")
    print(f"   - index: {manifest['index_type']} -> {faiss_file}")


def main():
    ap = argparse.ArgumentParser(description="Build FAISS RAG index")
    ap.add_argument("--input", default="data/manuals", help="Input directory with .txt/.md files")
    ap.add_argument("--output", default="data/index", help="Output directory for index artifacts")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model")
    ap.add_argument("--chunk_tokens", type=int, default=300, help="Target tokens per chunk")
    ap.add_argument("--chunk_overlap", type=int, default=40, help="Token overlap between chunks")
    ap.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of embeddings")
    ap.add_argument("--flat", action="store_true", help="Use FlatIP instead of HNSW")
    ap.add_argument("--M", type=int, default=32, help="HNSW M (connections)")
    ap.add_argument("--ef", type=int, default=128, help="HNSW efSearch/efConstruction default")
    ap.add_argument("--bm25", action="store_true", help="Export BM25 token corpus jsonl")
    args = ap.parse_args()

    build_index(
        input_dir=args.input,
        output_dir=args.output,
        embedding_model=args.model,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        normalize=not args.no_normalize,
        use_hnsw=not args.flat,
        hnsw_M=args.M,
        hnsw_ef=args.ef,
        build_bm25=args.bm25,
    )


if __name__ == "__main__":
    main()