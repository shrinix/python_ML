"""LLM adapter with optional OpenAI backend.

Keeps the project offline by default. When ENABLE_LLM is true and required
environment vars are present, uses the selected provider to generate a
final answer from retrieved context.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    from chess_tutor.config import ENABLE_LLM, LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY_ENV
except Exception:
    from config import ENABLE_LLM, LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY_ENV  # type: ignore


class LLMClient:
    def __init__(self):
        # Allow runtime env toggle to override imported config value
        env_flag = os.environ.get("ENABLE_LLM", "").lower() in {"1", "true", "yes"}
        self.enabled = bool(ENABLE_LLM or env_flag)
        self.provider = (LLM_PROVIDER or "").lower()
        self.model = LLM_MODEL
        self._client = None
        # Do not hard-disable here; allow runtime toggle to enable later.
        # Client will be lazily initialized during generate().

    def _try_init_openai(self) -> bool:
        if self.provider != "openai":
            return False
        key = os.environ.get(OPENAI_API_KEY_ENV)
        if not key:
            return False
        try:
            from openai import OpenAI  # type: ignore
            # Newer SDKs auto-read env; keep explicit for clarity
            self._client = OpenAI(api_key=key)
            return True
        except Exception:
            return False

    def generate(self, system: str, user: str, temperature: float = 0.2, max_tokens: Optional[int] = None) -> Optional[str]:
        if not self.enabled:
            return None
        # Lazy init client on first use if enabled via runtime toggle
        if self.provider == "openai" and self._client is None:
            self._try_init_openai()
        if self.provider == "openai" and self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg = resp.choices[0].message.content if resp and resp.choices else None
                if msg:
                    return msg.strip()
            except Exception:
                return None
        return None
