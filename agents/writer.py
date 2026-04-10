"""Writer agent: synthesize search and RAG evidence into markdown output."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from agents.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)


class WriterAgent:
    """Generate and persist a structured markdown research report."""

    def __init__(self, llm: LLMClient | None = None, output_dir: str = 'outputs') -> None:
        self.llm = llm or LLMClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_prompt(
        self,
        topic: str,
        sub_questions: list[str],
        search_context: dict[str, list[dict[str, str]]],
        rag_context: dict[str, list[dict[str, str]]],
    ) -> str:
        lines: list[str] = [f"Topic: {topic}", '', 'Sub-questions and evidence:']

        for idx, q in enumerate(sub_questions, start=1):
            lines.append(f"\n{idx}. {q}")
            lines.append('Web/Tool Evidence:')
            for item in search_context.get(q, []):
                lines.append(
                    f"- {item.get('title', '')} | {item.get('url', '')} | "
                    f"{item.get('source', '')}\n  {item.get('snippet', '')}"
                )

            lines.append('Local RAG Evidence:')
            for item in rag_context.get(q, []):
                lines.append(
                    f"- source={item.get('source', '')} score={item.get('score', '')}\n"
                    f"  {item.get('content', '')[:500]}"
                )

        return '\n'.join(lines)

    def run(
        self,
        topic: str,
        sub_questions: list[str],
        search_context: dict[str, list[dict[str, str]]],
        rag_context: dict[str, list[dict[str, str]]],
    ) -> Path:
        """Create and save the final markdown report to /outputs."""
        system_prompt = (
            "You are a technical research writer. "
            "Produce a clear markdown report with these sections exactly: "
            "# Title, ## Executive Summary, ## Findings, ## Recommendations, ## Sources. "
            "In Findings, structure by sub-question and cite evidence snippets."
        )
        user_prompt = self._build_prompt(topic, sub_questions, search_context, rag_context)

        try:
            report_md = self.llm.complete(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
            if not report_md.strip().startswith('#'):
                report_md = f"# Research Report: {topic}\n\n" + report_md
        except Exception as exc:
            LOGGER.exception('Writer LLM failed, creating fallback report: %s', exc)
            report_md = self._fallback_report(topic, sub_questions, search_context, rag_context)

        ts = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
        safe_topic = ''.join(ch for ch in topic.lower() if ch.isalnum() or ch in {'-', '_'})[:40] or 'topic'
        out_path = self.output_dir / f"report-{safe_topic}-{ts}.md"
        out_path.write_text(report_md, encoding='utf-8')
        return out_path

    @staticmethod
    def _fallback_report(
        topic: str,
        sub_questions: list[str],
        search_context: dict[str, list[dict[str, str]]],
        rag_context: dict[str, list[dict[str, str]]],
    ) -> str:
        lines = [
            f"# Research Report: {topic}",
            '',
            '## Executive Summary',
            'LLM generation was unavailable; this report summarizes collected evidence directly.',
            '',
            '## Findings',
        ]
        for i, q in enumerate(sub_questions, start=1):
            lines.append(f"### {i}. {q}")
            lines.append('**Search Evidence**')
            for item in search_context.get(q, []):
                lines.append(f"- {item.get('title','')} ({item.get('url','')})")
            lines.append('**RAG Evidence**')
            for item in rag_context.get(q, []):
                lines.append(f"- {item.get('source','')} (score={item.get('score','')})")
            lines.append('')

        lines += [
            '## Recommendations',
            '- Validate key claims with primary sources before decision-making.',
            '- Expand `reference_docs/` and rerun ingestion to improve local-context coverage.',
            '',
            '## Sources',
            '- Tool search results listed under each finding.',
            '- Local ChromaDB chunks from ingested docs.',
        ]
        return '\n'.join(lines)
