#!/usr/bin/env python3
"""cProfile hot paths for vector store count + query (run from repo root)."""

from __future__ import annotations

import cProfile
import pstats
import sys
import tempfile
from io import StringIO
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import chromadb
from chromadb.config import Settings

from tools.vector_store import VectorStore


def main() -> None:
    tmp = tempfile.TemporaryDirectory()
    vs = VectorStore(persist_directory=tmp.name, collection_name="profile_col")
    vs.add_text_chunks(
        [f"chunk {i} about agents and rag" for i in range(30)],
        [f"c{i}" for i in range(30)],
    )

    def hot_path():
        for _ in range(40):
            vs.count()
        for _ in range(15):
            vs.format_query_results("agent framework retrieval", 5)

    pr = cProfile.Profile()
    pr.enable()
    hot_path()
    pr.disable()

    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(25)
    print(s.getvalue())
    tmp.cleanup()


if __name__ == "__main__":
    main()
