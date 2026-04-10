#!/usr/bin/env python3
"""
Ingest reference documents (.txt, .md) from a directory into ChromaDB.

Usage:
    python ingest.py --path ./reference_docs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from tools.vector_store import VectorStore, chunk_text, read_text_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")


def ingest_directory(vs: VectorStore, root: Path) -> int:
    """Ingest all supported files under ``root``. Returns number of chunks added."""
    patterns = ("*.md", "*.txt")
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(root.rglob(pat)))
    if not files:
        logger.warning("No .md or .txt files found under %s", root)
        return 0

    texts: list[str] = []
    ids: list[str] = []
    metas: list[dict] = []
    for fp in files:
        try:
            content = read_text_file(fp)
        except OSError as exc:
            logger.error("Failed to read %s: %s", fp, exc)
            continue
        rel = str(fp.relative_to(root))
        for i, chunk in enumerate(chunk_text(content)):
            cid = f"{rel}::chunk-{i}"
            texts.append(chunk)
            ids.append(cid)
            metas.append({"source": rel, "chunk_index": i})

    if not texts:
        return 0

    vs.add_text_chunks(texts, ids, metas)
    return len(texts)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest reference docs into ChromaDB")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("reference_docs"),
        help="Directory containing .md and .txt files (default: ./reference_docs)",
    )
    args = parser.parse_args()
    root: Path = args.path.resolve()
    if not root.is_dir():
        logger.error("Not a directory: %s", root)
        return 1

    vs = VectorStore()
    before = vs.count()
    n = ingest_directory(vs, root)
    after = vs.count()
    logger.info("Ingest complete: %d new chunks written (collection size %d -> %d)", n, before, after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
