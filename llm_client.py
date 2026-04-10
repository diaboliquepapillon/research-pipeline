"""
Shared chat client factory for Microsoft Agent Framework.

Prefers Azure OpenAI when endpoint + key are configured; otherwise uses
OpenAI or any OpenAI-compatible endpoint via OPENAI_API_KEY.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)


def build_chat_client() -> OpenAIChatClient:
    """
    Construct an :class:`OpenAIChatClient` from environment variables.

    Returns:
        Configured client for ``as_agent(...)``.

    Raises:
        ValueError: If no usable credentials are present.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if azure_endpoint and azure_key:
        logger.info("Using Azure OpenAI client (endpoint=%s, deployment=%s)", azure_endpoint, deployment)
        return OpenAIChatClient(
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_key=azure_key,
            model=deployment,
            api_version=api_version,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "Set OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY in your environment."
        raise ValueError(msg)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    logger.info("Using OpenAI-compatible client (model=%s)", model)
    kwargs: dict[str, Any] = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIChatClient(**kwargs)
