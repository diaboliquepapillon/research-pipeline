"""Search tool with web and local corpus fallback."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)


class SearchTool:
    """Simple retrieval tool used by SearchAgent."""

    def __init__(self, reference_dir: str = "reference_docs") -> None:
        self.reference_dir = Path(reference_dir)
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """Try web search first (if configured), otherwise local docs."""
        if self.serper_api_key:
            try:
                results = self._search_serper(query=query, top_k=top_k)
                if results:
                    return results
            except Exception as exc:
                LOGGER.warning("Web search failed, falling back to local search: %s", exc)
        return self._search_local_docs(query=query, top_k=top_k)

    def _search_serper(self, query: str, top_k: int) -> list[dict[str, str]]:
        url = "https://google.serper.dev/search"
        response = requests.post(
            url,
            headers={"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"},
            json={"q": query, "num": top_k},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        organic = payload.get("organic", [])
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "web",
            }
            for item in organic[:top_k]
        ]

    def _search_local_docs(self, query: str, top_k: int) -> list[dict[str, str]]:
        if not self.reference_dir.exists():
            return []
        query_terms = self._tokenize(query)
        scored: list[tuple[float, str, str]] = []
        for path in self.reference_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            score = self._score_text(query_terms, text)
            if score <= 0:
                continue
            scored.append((score, str(path.name), text))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, str]] = []
        for score, title, text in scored[:top_k]:
            snippet = re.sub(r"\s+", " ", text).strip()[:400]
            results.append(
                {
                    "title": title,
                    "url": f"local://{title}",
                    "snippet": snippet,
                    "source": f"local(score={score:.2f})",
                }
            )
        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {tok for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()) if tok}

    @staticmethod
    def _score_text(query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        terms = SearchTool._tokenize(text)
        overlap = len(query_terms.intersection(terms))
        return overlap / max(len(query_terms), 1)
