"""
Web search and mock retrieval for the Search Agent.

Live search uses ``duckduckgo-search`` when ``USE_MOCK_SEARCH`` is not set.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated

from agent_framework import tool

logger = logging.getLogger(__name__)

# Tiny in-repo knowledge base for demos without network
MOCK_SNIPPETS: dict[str, str] = {
    "default": (
        "Mock search mode: no live results. Ingest local reference_docs and enable "
        "USE_MOCK_SEARCH=0 with network access for real web search, or expand MOCK_SNIPPETS."
    ),
    "agent": (
        "Microsoft Agent Framework (MAF) is an open-source SDK for building agents with "
        "tool use, workflows, and multiple model providers (OpenAI, Azure OpenAI, etc.)."
    ),
    "rag": (
        "Retrieval-Augmented Generation combines a retriever (vector DB, search index) with "
        "an LLM so answers are grounded in private or fresh documents."
    ),
}


def _mock_answer(query: str) -> str:
    q = query.lower()
    body = MOCK_SNIPPETS["default"]
    for key, snippet in MOCK_SNIPPETS.items():
        if key != "default" and key in q:
            body = snippet
            break
    payload = {
        "query": query,
        "results": [
            {"title": "Mock result", "body": body, "url": "https://example.invalid/mock"},
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def run_web_search(query: str, max_results: int = 5) -> str:
    """
    Execute a web search and return a compact JSON string of results.

    This is the core implementation used by the ``web_search`` tool and by
    :class:`~agents.searcher.SearchAgent` for direct calls.
    """
    if os.getenv("USE_MOCK_SEARCH", "0").strip() in {"1", "true", "True", "yes"}:
        logger.info("Mock search for: %s", query)
        return _mock_answer(query)

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed; using mock search")
        return _mock_answer(query)

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max(1, min(max_results, 10))))
    except Exception as exc:
        logger.exception("Web search failed; falling back to mock: %s", exc)
        return _mock_answer(query)

    simplified = []
    for h in hits:
        simplified.append(
            {
                "title": h.get("title", ""),
                "body": h.get("body", ""),
                "href": h.get("href", ""),
            }
        )
    return json.dumps({"query": query, "results": simplified}, ensure_ascii=False)


@tool
def web_search(
    query: Annotated[str, "Focused search query; prefer one clear question or keyword phrase"],
    max_results: Annotated[int, "Number of results to return (1-10)"] = 5,
) -> str:
    """Search the public web for recent information relevant to the query."""
    return run_web_search(query, max_results=max_results)
