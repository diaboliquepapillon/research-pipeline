"""
Orchestrates Planner → Search → RAG → Writer with explicit context passing.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from agents.planner import PlannerAgent
from agents.rag_agent import RagAgent
from agents.searcher import SearchAgent
from agents.writer import WriterAgent
from llm_client import build_chat_client
from tools.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """
    End-to-end research run: each stage receives structured context from the previous one.
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._chat = build_chat_client()
        self._vs = vector_store or VectorStore()
        self.planner = PlannerAgent(self._chat)
        self.searcher = SearchAgent(self._chat)
        self.rag = RagAgent(self._chat, self._vs)
        self.writer = WriterAgent(self._chat)

    async def _search_and_rag_for_subquestion(self, sq: str, topic: str) -> dict[str, Any]:
        web_text, rag_text = await asyncio.gather(
            self.searcher.research(sq, parent_topic=topic),
            self.rag.retrieve(sq, parent_topic=topic),
        )
        return {
            "sub_question": sq,
            "web": web_text,
            "rag": rag_text,
        }

    async def run(self, topic: str) -> tuple[Path, dict[str, Any]]:
        """
        Execute the full pipeline for *topic*.

        Returns:
            Tuple of (report_path, debug_context dict for UI or logging).
        """
        logger.info("Pipeline start topic=%r", topic[:120])
        ctx: dict[str, Any] = {"topic": topic, "sub_questions": [], "findings": []}

        plan = await self.planner.plan(topic)
        ctx["sub_questions"] = plan.sub_questions
        ctx["planner_rationale"] = plan.rationale

        # Search and RAG do not depend on each other for the same sub-question; overlap them
        # to save roughly one LLM round-trip per sub-question vs strict sequential awaits.
        findings: list[dict[str, Any]] = []
        for sq in plan.sub_questions:
            findings.append(await self._search_and_rag_for_subquestion(sq, topic))

        ctx["findings"] = findings
        out_path = await self.writer.write_report(topic, findings)
        ctx["report_path"] = str(out_path)
        logger.info("Pipeline complete -> %s", out_path)
        return out_path, ctx
