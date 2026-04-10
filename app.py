"""Streamlit app for the multi-agent research pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents.planner import PlannerAgent
from agents.rag_agent import RAGAgent
from agents.searcher import SearchAgent
from agents.writer import WriterAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
)
LOGGER = logging.getLogger('app')


def run_pipeline(topic: str) -> tuple[Path, dict[str, object]]:
    """Run planner -> searcher -> rag -> writer and return report path plus context."""
    planner = PlannerAgent()
    searcher = SearchAgent()
    rag = RAGAgent()
    writer = WriterAgent()

    sub_questions = planner.run(topic)
    search_context = searcher.run(sub_questions)
    rag_context = rag.run(sub_questions)
    report_path = writer.run(
        topic=topic,
        sub_questions=sub_questions,
        search_context=search_context,
        rag_context=rag_context,
    )

    context = {
        'sub_questions': sub_questions,
        'search_context': search_context,
        'rag_context': rag_context,
    }
    return report_path, context


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title='Multi-Agent Research Pipeline', layout='wide')

    st.title('Multi-Agent Research Pipeline')
    st.caption('Planner -> Search -> RAG -> Writer')

    topic = st.text_input('Research topic', placeholder='e.g., retrieval-augmented generation for enterprise search')

    if st.button('Generate Report', type='primary'):
        if not topic.strip():
            st.warning('Please provide a research topic.')
            return

        with st.spinner('Running agents...'):
            try:
                report_path, context = run_pipeline(topic)
            except Exception as exc:
                LOGGER.exception('Pipeline failed: %s', exc)
                st.error(f'Pipeline failed: {exc}')
                return

        st.success(f'Report generated: {report_path.name}')

        st.subheader('Planned sub-questions')
        for i, q in enumerate(context['sub_questions'], start=1):
            st.markdown(f"{i}. {q}")

        report_text = report_path.read_text(encoding='utf-8')
        st.subheader('Report Preview')
        st.markdown(report_text)

        st.download_button(
            label='Download report (.md)',
            data=report_text,
            file_name=report_path.name,
            mime='text/markdown',
        )


if __name__ == '__main__':
    main()
