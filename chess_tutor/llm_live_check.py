#!/usr/bin/env python3
"""Simple executable script to verify live LLM connectivity (no pytest).

Usage:
  python llm_live_check.py            # uses current working interpreter
  ENABLE_LLM=true OPENAI_API_KEY=sk-... python llm_live_check.py

Behavior:
  * Verifies required environment variables and openai package.
  * Instantiates LLMClient and performs a short generation.
  * Prints a concise JSON-style summary plus the model response snippet.
  * Exit codes:
      0 success (response received)
      2 skipped (preconditions not met)
      1 any failure / exception

Suppressing FAISS SWIG warnings: they may still appear if FAISS is installed; they are harmless.
"""
from __future__ import annotations
import os
import sys
import json
import traceback
from typing import Dict, Any

# Ensure we can import backend even if run from repo root or this directory
THIS = os.path.abspath(os.path.dirname(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

# Import config first (may set defaults) then consider runtime overrides
try:
    from chess_tutor import config as _cfg  # type: ignore
except Exception:
    import config as _cfg  # type: ignore

RESULT: Dict[str, Any] = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "enable_llm_env": os.environ.get("ENABLE_LLM"),
    "enable_llm_cfg": getattr(_cfg, "ENABLE_LLM", None),
    "provider_cfg": getattr(_cfg, "LLM_PROVIDER", None),
    "model_cfg": getattr(_cfg, "LLM_MODEL", None),
}

# Preconditions
key_env_name = getattr(_cfg, "OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
api_key_present = bool(os.environ.get(key_env_name))
RESULT["api_key_env_var"] = key_env_name
RESULT["api_key_present"] = api_key_present

# Determine if we should attempt live call
enabled_env = str(os.environ.get("ENABLE_LLM", "")).lower() in {"1", "true", "yes"}
should_run = enabled_env or bool(getattr(_cfg, "ENABLE_LLM", False))
if not should_run:
    RESULT["status"] = "skip"
    RESULT["reason"] = "ENABLE_LLM not true"
    print(json.dumps(RESULT, indent=2))
    sys.exit(2)
if (getattr(_cfg, "LLM_PROVIDER", "openai") or "").lower() != "openai":
    RESULT["status"] = "skip"
    RESULT["reason"] = f"Unsupported provider {getattr(_cfg, 'LLM_PROVIDER', None)}"
    print(json.dumps(RESULT, indent=2))
    sys.exit(2)
try:
    import openai  # noqa: F401
    RESULT["openai_import"] = True
except Exception as e:
    RESULT["openai_import"] = False
    RESULT["status"] = "skip"
    RESULT["reason"] = f"openai package not available: {e}".strip()
    print(json.dumps(RESULT, indent=2))
    sys.exit(2)
if not api_key_present:
    RESULT["status"] = "skip"
    RESULT["reason"] = f"Environment variable {key_env_name} not set"
    print(json.dumps(RESULT, indent=2))
    sys.exit(2)

# Attempt generation
try:
    # Ensure runtime ENABLE_LLM visible to adapter
    os.environ.setdefault("ENABLE_LLM", "true")
    try:
        from backend.llm_adapter import LLMClient  # type: ignore
    except Exception:
        from chess_tutor.backend.llm_adapter import LLMClient  # type: ignore
    client = LLMClient()
    RESULT["client_enabled"] = bool(getattr(client, "enabled", False))
    if not RESULT["client_enabled"]:
        RESULT["status"] = "error"
        RESULT["reason"] = "LLMClient disabled despite ENABLE_LLM"
        print(json.dumps(RESULT, indent=2))
        sys.exit(1)
    resp = client.generate(
        system="You are a concise assistant.",
        user="Reply with exactly the word OK.",
        temperature=0.0,
        max_tokens=5,
    )
    RESULT["raw_response"] = resp
    if not (isinstance(resp, str) and resp.strip()):
        RESULT["status"] = "error"
        RESULT["reason"] = "Empty or None response from LLM"
        print(json.dumps(RESULT, indent=2))
        sys.exit(1)
    RESULT["status"] = "success"
    RESULT["response_preview"] = resp.strip()[:120]
    print(json.dumps(RESULT, indent=2))
    # Success exit code
    sys.exit(0)
except SystemExit:
    raise
except Exception:
    RESULT["status"] = "exception"
    RESULT["traceback"] = traceback.format_exc(limit=6)
    print(json.dumps(RESULT, indent=2))
    sys.exit(1)
