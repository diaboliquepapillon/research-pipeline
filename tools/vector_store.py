"""
ChromaDB helpers: persistent collection, ingest text files, and similarity search.

Designed so RAG can be exercised independently of agents (see ``ingest.py``).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import chromadb
from chromadb.api import Collection
from chromadb.config import Settings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

logger = logging.getLogger(__name__)

# Chroma's DefaultEmbeddingFunction delegates with ``ONNXMiniLM_L6_V2()(input)``, which
# constructs a *new* ONNX runtime session on every embed call. That dominated query
# latency in profiling; reuse one embedder for the process instead.
_shared_embedder: ONNXMiniLM_L6_V2 | None = None
_embedder_lock = threading.Lock()


def _get_shared_embedder() -> ONNXMiniLM_L6_V2:
    global _shared_embedder
    if _shared_embedder is None:
        with _embedder_lock:
            if _shared_embedder is None:
                _shared_embedder = ONNXMiniLM_L6_V2()
    return _shared_embedder

DEFAULT_PERSIST_DIR = ".chroma"
DEFAULT_COLLECTION = "research_refs"


def _persist_dir() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)


def _collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION)


class VectorStore:
    """Thin wrapper around a persistent Chroma collection with default embeddings."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_directory = persist_directory or _persist_dir()
        self.collection_name = collection_name or _collection_name()
        self._embedding_fn = _get_shared_embedder()
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
        )
        # Avoid repeated SQLite/metadata round-trips when planner/RAG call count() often.
        self._count_cache: int | None = None
        logger.debug(
            "VectorStore ready at %s collection=%s",
            self.persist_directory,
            self.collection_name,
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def count(self) -> int:
        if self._count_cache is None:
            self._count_cache = self._collection.count()
        return self._count_cache

    def _invalidate_count_cache(self) -> None:
        self._count_cache = None

    def add_text_chunks(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add or update documents (upsert by id)."""
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length")
        self._collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
        self._invalidate_count_cache()
        logger.info("Upserted %d chunks into %s", len(ids), self.collection_name)

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Run similarity search and return structured hits.

        Each hit includes ``document``, ``metadata``, and ``distance`` (if present).
        """
        n = max(1, min(n_results, 50))
        if self.count() == 0:
            logger.warning("Chroma collection is empty; RAG will return no hits")
            return []

        raw = self._collection.query(query_texts=[query_text], n_results=n)
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0] if raw.get("distances") else [None] * len(docs)

        results: list[dict] = []
        for i, doc in enumerate(docs):
            results.append(
                {
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return results

    def format_query_results(self, query_text: str, n_results: int = 5) -> str:
        """Human-readable bundle of retrieved passages for LLM context."""
        hits = self.query(query_text, n_results=n_results)
        if not hits:
            return "(No reference documents matched this query in the vector store.)"
        parts: list[str] = []
        for i, h in enumerate(hits, start=1):
            meta = h.get("metadata") or {}
            src = meta.get("source", "unknown")
            parts.append(f"--- Source {i} ({src}) ---\n{h['document']}")
        return "\n\n".join(parts)


def read_text_file(path: Path) -> str:
    """Load a UTF-8 text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Simple character-based chunking with overlap for long pages."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
