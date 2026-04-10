"""
Streamlit UI for the multi-agent research pipeline.

Run from the project root::

    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("app")


def main() -> None:
    st.set_page_config(page_title="Research Pipeline", layout="wide")
    st.title("Multi-agent research pipeline")
    st.markdown(
        "Planner → Search (web) → RAG (ChromaDB) → Writer. "
        "Requires API keys in `.env` (see `.env.example`)."
    )

    topic = st.text_area(
        "Research topic",
        placeholder="e.g. Compare retrieval options for agentic RAG in 2026",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Run pipeline", type="primary")
    with col2:
        show_debug = st.checkbox("Show intermediate context", value=False)

    if run_btn:
        if not topic.strip():
            st.warning("Enter a topic first.")
            return

        with st.spinner("Running agents (this may take a minute)…"):
            try:
                from pipeline import ResearchPipeline

                async def _go() -> tuple[Path, dict]:
                    pipe = ResearchPipeline()
                    return await pipe.run(topic.strip())

                report_path, ctx = asyncio.run(_go())
            except ValueError as exc:
                st.error(str(exc))
                logger.warning("Configuration error: %s", exc)
                return
            except Exception as exc:
                st.exception(exc)
                logger.exception("Pipeline failed")
                return

        st.success(f"Report saved to `{report_path}`")
        try:
            data = report_path.read_bytes()
            st.download_button(
                label="Download markdown report",
                data=data,
                file_name=report_path.name,
                mime="text/markdown",
            )
        except OSError as exc:
            st.warning(f"Could not read report for download: {exc}")

        with st.expander("Final report preview", expanded=True):
            st.markdown(report_path.read_text(encoding="utf-8"))

        if show_debug:
            with st.expander("Debug: planner + per-step context"):
                st.json(
                    {
                        "sub_questions": ctx.get("sub_questions"),
                        "planner_rationale": ctx.get("planner_rationale"),
                        "findings": ctx.get("findings"),
                    }
                )


if __name__ == "__main__":
    main()
