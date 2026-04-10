"""Ingest reference documents from `reference_docs/` into ChromaDB."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from tools.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("ingest")


def load_reference_documents(reference_dir: Path) -> list[tuple[str, str]]:
    """Load text/markdown documents from a directory recursively."""
    docs: list[tuple[str, str]] = []
    for path in reference_dir.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if content:
            docs.append((str(path.relative_to(reference_dir)), content))
    return docs


def run_ingest(
    reference_dir: Path,
    persist_directory: str,
    collection_name: str,
    chunk_size: int,
    overlap: int,
) -> int:
    """Read source docs and push chunks to Chroma."""
    docs = load_reference_documents(reference_dir=reference_dir)
    if not docs:
        LOGGER.warning("No documents found in %s", reference_dir)
        return 0

    store = VectorStore(persist_directory=persist_directory, collection_name=collection_name)
    added = store.add_documents(docs=docs, chunk_size=chunk_size, overlap=overlap)
    LOGGER.info("Done. Added %s chunks from %s documents.", added, len(docs))
    return added


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest docs into ChromaDB")
    parser.add_argument("--reference-dir", default="reference_docs")
    parser.add_argument("--persist-dir", default=".chroma")
    parser.add_argument("--collection-name", default="research_docs")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    try:
        run_ingest(
            reference_dir=Path(args.reference_dir),
            persist_directory=args.persist_dir,
            collection_name=args.collection_name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc
