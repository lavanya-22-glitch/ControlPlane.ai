"""
rag/vectorstore.py
──────────────────
FAISS index management: build from corpus text, persist to disk, reload.
Chunking is handled here via LangChain's RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.documents import Document

from rag.config import (
    CORPUS_PATH,
    FAISS_INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from rag.embedder import MiniLMEmbedder


def _load_corpus() -> str:
    """Read raw corpus text from disk."""
    path = Path(CORPUS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}")
    return path.read_text(encoding="utf-8")


def _chunk_text(raw_text: str) -> List[Document]:
    """Split raw text into overlapping chunks as LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    return [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(chunks)]


def build_faiss_index(embedder: MiniLMEmbedder | None = None) -> FAISS:
    """
    Build a new FAISS index from the corpus and persist it to disk.
    Returns the loaded FAISS vector store.
    """
    if embedder is None:
        embedder = MiniLMEmbedder()

    print("[VectorStore] Loading and chunking corpus ...")
    raw = _load_corpus()
    docs = _chunk_text(raw)
    print(f"[VectorStore] Created {len(docs)} chunks from corpus.")

    print("[VectorStore] Embedding chunks and building FAISS index ...")
    store = FAISS.from_documents(docs, embedder)

    # Persist
    index_dir = Path(FAISS_INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    print(f"[VectorStore] Index saved to '{index_dir}'")
    return store


def load_faiss_index(embedder: MiniLMEmbedder | None = None) -> FAISS:
    """
    Load a persisted FAISS index from disk.
    Builds it first if it doesn't exist yet.
    """
    if embedder is None:
        embedder = MiniLMEmbedder()

    index_dir = Path(FAISS_INDEX_DIR)
    if not index_dir.exists() or not (index_dir / "index.faiss").exists():
        print("[VectorStore] No existing index found — building from scratch ...")
        return build_faiss_index(embedder)

    print(f"[VectorStore] Loading existing FAISS index from '{index_dir}' ...")
    store = FAISS.load_local(
        str(index_dir),
        embedder,
        allow_dangerous_deserialization=True,
    )
    print("[VectorStore] Index loaded.")
    return store


def get_all_docs(store: FAISS) -> List[Document]:
    """Return all documents stored in the FAISS index (for BM25 corpus)."""
    # FAISS docstore holds all docs by their internal IDs
    return list(store.docstore._dict.values())
