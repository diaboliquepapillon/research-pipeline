"""RAG agent: retrieve local context from ChromaDB for each question."""

from __future__ import annotations

import logging

from tools.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)


class RAGAgent:
    """Fetch local vector-store evidence for planned sub-questions."""

    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()

    def run(self, sub_questions: list[str], top_k: int = 4) -> dict[str, list[dict[str, str]]]:
        """Retrieve top-k local chunks for each sub-question."""
        rag_results: dict[str, list[dict[str, str]]] = {}

        for question in sub_questions:
            try:
                chunks = self.store.query(query_text=question, top_k=top_k)
                rag_results[question] = [
                    {
                        'doc_id': item.doc_id,
                        'source': item.source,
                        'score': f"{item.score:.4f}",
                        'content': item.content,
                    }
                    for item in chunks
                ]
            except Exception as exc:
                LOGGER.exception('RAG retrieval failed for question %r: %s', question, exc)
                rag_results[question] = []

        return rag_results
