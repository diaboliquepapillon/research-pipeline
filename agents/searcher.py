"""
Search Agent: uses tool-calling (web search) to gather external evidence per sub-question.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tools.search_tool import web_search

if TYPE_CHECKING:
    from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)


class SearchAgent:
    """Runs MAF agent with ``web_search`` tool for one sub-question at a time."""

    def __init__(self, chat_client: OpenAIChatClient) -> None:
        self._agent = chat_client.as_agent(
            name="SearchAgent",
            instructions=(
                "You are a research assistant. For each user sub-question, call the web_search "
                "tool one or more times with focused queries. Then synthesize: "
                "(1) 3-6 bullet points of factual claims, "
                "(2) a 2-3 sentence summary. "
                "If search returns little data, say so honestly. Do not fabricate sources."
            ),
            tools=[web_search],
        )

    async def research(self, sub_question: str, parent_topic: str = "") -> str:
        """Retrieve and summarize web evidence for *sub_question*."""
        logger.info("Searcher: sub_question=%r", sub_question[:120])
        ctx = ""
        if parent_topic:
            ctx = f"Overall topic: {parent_topic}\n\n"
        prompt = (
            f"{ctx}Sub-question to investigate:\n{sub_question}\n\n"
            "Use web_search as needed, then produce bullets + short summary."
        )
        try:
            resp = await self._agent.run(prompt)
            return resp.text.strip() or "(Search agent returned empty text.)"
        except Exception as exc:
            logger.exception("Search agent failed: %s", exc)
            return f"(Search agent error: {exc})"
