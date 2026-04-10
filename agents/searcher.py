"""Search agent module for per-question evidence retrieval."""

from __future__ import annotations

import logging

from tools.search_tool import SearchTool

LOGGER = logging.getLogger(__name__)


class SearchAgent:
    """Retrieve external evidence for each research sub-question."""

    def __init__(self, search_tool: SearchTool | None = None) -> None:
        self.search_tool = search_tool or SearchTool()

    def run(self, sub_questions: list[str], top_k: int = 4) -> dict[str, list[dict[str, str]]]:
        results: dict[str, list[dict[str, str]]] = {}
        for question in sub_questions:
            try:
                results[question] = self.search_tool.search(query=question, top_k=top_k)
            except Exception as exc:
                LOGGER.exception("Search failed for question '%s': %s", question, exc)
                results[question] = []
        return results
