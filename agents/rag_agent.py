"""
RAG Agent: queries the local ChromaDB collection via an MAF tool for grounded snippets.
"""

from __future__ import annotations

import logging
from typing import Annotated, TYPE_CHECKING

from agent_framework import tool

from tools.vector_store import VectorStore

if TYPE_CHECKING:
    from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)


def _build_reference_tool(vs: VectorStore):
    @tool(
        name="search_reference_documents",
        description=(
            "Search ingested reference documents (vector store) for passages relevant "
            "to the query. Use for facts that may appear in internal or uploaded docs."
        ),
    )
    def search_reference_documents(
        query: Annotated[str, "Natural language query aligned with the sub-question"],
        n_results: Annotated[int, "How many chunks to retrieve (1-10)"] = 5,
    ) -> str:
        return vs.format_query_results(query, n_results=n_results)

    return search_reference_documents


class RagAgent:
    """Agentic RAG: the model may call the retrieval tool and then explain results."""

    def __init__(self, chat_client: OpenAIChatClient, vector_store: VectorStore) -> None:
        ref_tool = _build_reference_tool(vector_store)
        self._vector_store = vector_store
        self._agent = chat_client.as_agent(
            name="RagAgent",
            instructions=(
                "You answer using the search_reference_documents tool against a local knowledge "
                "base. Call the tool with a focused query. Then summarize the retrieved "
                "passages in 3-6 bullets plus a 2-sentence overview. "
                "If the tool returns no matches, state that clearly."
            ),
            tools=[ref_tool],
        )

    async def retrieve(self, sub_question: str, parent_topic: str = "") -> str:
        """Run retrieval-oriented reasoning for *sub_question*."""
        logger.info("RAG: sub_question=%r", sub_question[:120])
        ctx = ""
        if parent_topic:
            ctx = f"Overall topic: {parent_topic}\n\n"
        prompt = (
            f"{ctx}Sub-question:\n{sub_question}\n\n"
            "Use search_reference_documents at least once, then summarize."
        )
        try:
            if self._vector_store.count() == 0:
                return (
                    "(Vector store is empty — run `python ingest.py` on reference_docs "
                    "before expecting RAG hits.)"
                )
            resp = await self._agent.run(prompt)
            return resp.text.strip() or "(RAG agent returned empty text.)"
        except Exception as exc:
            logger.exception("RAG agent failed: %s", exc)
            return f"(RAG agent error: {exc})"
