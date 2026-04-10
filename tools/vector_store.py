"""ChromaDB vector store helpers for the research pipeline."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

LOGGER = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start += step
    return chunks


@dataclass(slots=True)
class RetrievedChunk:
    """A normalized retrieval record from Chroma query output."""

    doc_id: str
    content: str
    source: str
    score: float


class VectorStore:
    """Small wrapper around ChromaDB with project-friendly defaults."""

    def __init__(self, persist_directory: str = ".chroma", collection_name: str = "research_docs") -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction(),
            metadata={"purpose": "research_pipeline_rag"},
        )
        self._count_cache: int | None = None

    @staticmethod
    def _build_chunk_id(source: str, chunk_index: int, content: str) -> str:
        fingerprint = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
        return f"{Path(source).name}-{chunk_index}-{fingerprint}"

    def add_text_chunks(
        self,
        chunks: Iterable[str],
        ids: Iterable[str],
        metadatas: Iterable[dict[str, str]] | None = None,
    ) -> None:
        """Add pre-computed chunks and ids to the collection."""
        chunk_list = list(chunks)
        id_list = list(ids)
        if not chunk_list:
            return
        if len(chunk_list) != len(id_list):
            raise ValueError("chunks and ids lengths must match")

        metadata_list = list(metadatas) if metadatas is not None else [{"source": "unknown"} for _ in chunk_list]
        if len(metadata_list) != len(chunk_list):
            raise ValueError("metadatas length must match chunks")

        self._collection.upsert(
            ids=id_list,
            documents=chunk_list,
            metadatas=metadata_list,
        )
        self._count_cache = None

    def add_documents(self, docs: list[tuple[str, str]], chunk_size: int = 1000, overlap: int = 150) -> int:
        """
        Ingest documents provided as (source_name, text) pairs.

        Returns number of chunks added.
        """
        chunk_texts: list[str] = []
        chunk_ids: list[str] = []
        chunk_metadata: list[dict[str, str]] = []

        for source, body in docs:
            chunks = chunk_text(body, chunk_size=chunk_size, overlap=overlap)
            for idx, ch in enumerate(chunks):
                chunk_texts.append(ch)
                chunk_ids.append(self._build_chunk_id(source, idx, ch))
                chunk_metadata.append({"source": source, "chunk_index": str(idx)})

        self.add_text_chunks(chunk_texts, chunk_ids, chunk_metadata)
        LOGGER.info("Ingested %s chunks into collection '%s'.", len(chunk_texts), self.collection_name)
        return len(chunk_texts)

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top-k chunks relevant to query_text."""
        if not query_text.strip():
            return []

        response = self._collection.query(query_texts=[query_text], n_results=top_k)
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for doc_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False):
            source = (meta or {}).get("source", "unknown")
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
            results.append(RetrievedChunk(doc_id=doc_id, content=doc, source=source, score=score))
        return results

    def format_query_results(self, query_text: str, top_k: int = 5) -> str:
        """Format retrieval output for prompt consumption."""
        results = self.query(query_text=query_text, top_k=top_k)
        if not results:
            return "No relevant RAG context found."
        lines: list[str] = []
        for i, item in enumerate(results, start=1):
            lines.append(
                f"[{i}] source={item.source} score={item.score:.3f}\n{item.content}"
            )
        return "\n\n".join(lines)

    def count(self) -> int:
        """Return number of vectors in the collection (cached)."""
        if self._count_cache is None:
            self._count_cache = int(self._collection.count())
        return self._count_cache
