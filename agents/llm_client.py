"""LLM backend abstraction for Azure Foundry or OpenAI fallback."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    """Runtime config for model and provider selection."""

    model: str
    provider: str = "openai"


class LLMClient:
    """Minimal chat-completions wrapper with JSON helper."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.config = config or LLMConfig(model=model, provider=provider)
        self._client = self._build_client()

    def _build_client(self) -> OpenAI:
        provider = self.config.provider.lower()
        if provider == "azure":
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
            if not endpoint or not api_key:
                raise ValueError("Azure provider selected but endpoint/key missing in environment")
            return OpenAI(
                api_key=api_key,
                base_url=f"{endpoint}/openai/deployments/{self.config.model}",
                default_query={"api-version": api_version},
                default_headers={"api-key": api_key},
            )
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Get a plain text completion."""
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            LOGGER.exception("LLM completion failed: %s", exc)
            raise

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Get model output and parse as JSON object."""
        raw = self.complete(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Guard against markdown fenced JSON.
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
