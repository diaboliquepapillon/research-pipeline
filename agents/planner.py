"""Planner agent: decompose a research topic into sub-questions."""

from __future__ import annotations

import logging

from agents.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)


class PlannerAgent:
    """Create a focused plan that downstream agents can execute."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, topic: str, max_questions: int = 5) -> list[str]:
        """Return sub-questions for the research topic."""
        topic = topic.strip()
        if not topic:
            return []

        system_prompt = (
            "You are a research planning assistant. "
            "Return STRICT JSON with key 'sub_questions' as an array of concise questions."
        )
        user_prompt = (
            f"Topic: {topic}\n"
            f"Create {max_questions} diverse sub-questions covering fundamentals, current state, "
            "trade-offs, and practical recommendations."
        )

        try:
            payload = self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            questions = payload.get('sub_questions', [])
            cleaned = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
            if cleaned:
                return cleaned[:max_questions]
        except Exception as exc:
            LOGGER.warning('LLM planning failed, using fallback planner: %s', exc)

        return [
            f"What is the current state of {topic}?",
            f"What are the key methods, tools, or frameworks for {topic}?",
            f"What are major trade-offs and risks in {topic}?",
            f"What real-world case studies exist for {topic}?",
            f"What best-practice recommendations apply to {topic}?",
        ][:max_questions]
