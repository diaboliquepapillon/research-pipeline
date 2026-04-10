"""
Micro-benchmarks for vector store and text utilities.

Run::

    pytest benchmarks/ --benchmark-only -v

Profile a hot path::

    python scripts/profile_vector_store.py
"""

from __future__ import annotations

import tempfile

import chromadb
import pytest
from chromadb.config import Settings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from tools.vector_store import VectorStore, chunk_text

# Reuse ONNX embedder so this benchmark measures PersistentClient churn, not Chroma's
# DefaultEmbeddingFunction anti-pattern (new session per embed).
_BENCH_EMBEDDER = ONNXMiniLM_L6_V2()


def _make_store_with_docs(n_chunks: int) -> tuple[VectorStore, tempfile.TemporaryDirectory]:
    """Fresh temp Chroma with n_chunks short documents."""
    tmp = tempfile.TemporaryDirectory()
    vs = VectorStore(persist_directory=tmp.name, collection_name="bench_collection")
    texts = [f"Document {i} about retrieval augmented generation and vector databases." for i in range(n_chunks)]
    ids = [f"id-{i}" for i in range(n_chunks)]
    vs.add_text_chunks(texts, ids)
    return vs, tmp


def test_benchmark_chunk_text_huge(benchmark):
    """Worst-case-ish: very long single string (character chunking loop)."""
    body = ("paragraph about agents and chroma embeddings. " * 500)  # ~22k chars
    benchmark(chunk_text, body, 1200, 150)


def test_benchmark_vector_query_cold_client(benchmark):
    """Worst case: new PersistentClient + query each iteration (anti-pattern)."""

    def work():
        tmp = tempfile.TemporaryDirectory()
        try:
            c = chromadb.PersistentClient(path=tmp.name, settings=Settings(anonymized_telemetry=False))
            col = c.get_or_create_collection(
                name="benchcol",
                embedding_function=_BENCH_EMBEDDER,
            )
            col.add(
                ids=["a"],
                documents=["hello world chromadb benchmark"],
            )
            col.query(query_texts=["chroma benchmark"], n_results=1)
        finally:
            tmp.cleanup()

    benchmark(work)


def test_benchmark_vector_query_reused_store(benchmark):
    """Typical case: one VectorStore, repeated queries."""
    vs, tmp = _make_store_with_docs(50)
    try:
        benchmark(vs.format_query_results, "vector database retrieval", 8)
    finally:
        tmp.cleanup()


def test_benchmark_collection_count_repeated(benchmark):
    """Repeated count() used to hit SQLite each time (regression guard after caching)."""
    vs, tmp = _make_store_with_docs(20)
    try:

        def many_counts():
            for _ in range(30):
                vs.count()

        benchmark(many_counts)
    finally:
        tmp.cleanup()
