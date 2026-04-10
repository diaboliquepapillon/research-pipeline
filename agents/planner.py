"""
Planner Agent: decomposes a research topic into focused sub-questions.

Uses Microsoft Agent Framework structured outputs (Pydantic) for reliable parsing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)


class PlanResult(BaseModel):
    """Structured plan returned by the planner LLM."""

    sub_questions: list[str] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Ordered sub-questions that together cover the topic",
    )
    rationale: str = Field(default="", description="Brief note on how the plan maps to the topic")


class PlannerAgent:
    """Breaks a user topic into sub-questions for downstream retrieval."""

    def __init__(self, chat_client: OpenAIChatClient) -> None:
        self._agent = chat_client.as_agent(
            name="PlannerAgent",
            instructions=(
                "You are a research planner. Given a topic, output 3-7 concrete sub-questions "
                "that would need to be answered to produce an excellent survey. "
                "Questions should be specific, non-overlapping, and suitable for web search "
                "and document retrieval. Do not answer them."
            ),
            default_options={"response_format": PlanResult},
        )

    async def plan(self, topic: str) -> PlanResult:
        """Generate sub-questions for *topic*."""
        logger.info("Planner: topic=%r", topic[:120])
        try:
            resp = await self._agent.run(
                f"Research topic:\n{topic}\n\nProduce sub_questions and a short rationale."
            )
            parsed = resp.value
            if parsed is None:
                raise ValueError("Planner returned no structured value")
            return parsed
        except Exception:
            logger.exception("Planner agent failed; using single fallback question")
            return PlanResult(sub_questions=[topic], rationale="Fallback: planner error")
