"""
Writer Agent: synthesizes planner + search + RAG context into a markdown report file.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")


def _slugify(topic: str) -> str:
    s = topic.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return (s[:60] or "report").strip("-")


class WriterAgent:
    """Produces a polished markdown report and persists it under ``outputs/``."""

    def __init__(self, chat_client: OpenAIChatClient) -> None:
        self._agent = chat_client.as_agent(
            name="WriterAgent",
            instructions=(
                "You write professional technical markdown reports. "
                "Use clear headings (##, ###), bullet lists where helpful, and a short "
                "executive summary. Distinguish **Web findings** vs **Reference library (RAG)** "
                "sections per sub-question when sources differ. "
                "End with **Open questions** and **Suggested next steps**. "
                "Do not invent citations; reflect only the provided context."
            ),
        )

    def _build_prompt(self, topic: str, findings: list[dict[str, Any]]) -> str:
        lines = [
            f"# Report task",
            f"Topic: {topic}",
            "",
            "## Assembled findings (from upstream agents)",
            "",
        ]
        for i, block in enumerate(findings, start=1):
            sq = block.get("sub_question", "")
            lines.append(f"### Block {i}: {sq}")
            lines.append("#### Web search synthesis")
            lines.append(block.get("web", "").strip() or "_None_")
            lines.append("")
            lines.append("#### RAG / reference library synthesis")
            lines.append(block.get("rag", "").strip() or "_None_")
            lines.append("")
        lines.append("Write the final markdown report now.")
        return "\n".join(lines)

    async def write_report(self, topic: str, findings: list[dict[str, Any]]) -> Path:
        """
        Generate markdown from structured *findings* and save to ``outputs/``.

        Returns:
            Path to the written ``.md`` file.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"{_slugify(topic)}-{ts}.md"
        out_path = OUTPUT_DIR / filename

        logger.info("Writer: topic=%r -> %s", topic[:80], out_path)
        prompt = self._build_prompt(topic, findings)
        try:
            resp = await self._agent.run(prompt)
            body = resp.text.strip()
            if not body:
                body = "_Report generation returned empty content._\n"
        except Exception as exc:
            logger.exception("Writer agent failed: %s", exc)
            body = f"# Report\n\n_(Writer error: {exc})_\n"

        header = (
            f"<!-- generated: {datetime.now(timezone.utc).isoformat()} topic: {topic[:200]} -->\n"
        )
        out_path.write_text(header + body + "\n", encoding="utf-8")
        logger.info("Wrote report (%d chars) to %s", len(body), out_path)
        return out_path
