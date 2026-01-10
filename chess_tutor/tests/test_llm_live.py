"""Live LLM responsiveness test.

This test makes a real call via backend.llm_adapter.LLMClient when:
- ENABLE_LLM=true (env)
- LLM_PROVIDER=openai (default) and 'openai' package is installed
- OPENAI_API_KEY is set in environment

Otherwise the test is skipped with a clear reason, so it won't fail CI by default.
"""
from __future__ import annotations
import os
import sys
import pathlib
import pytest

# Ensure we can import the backend package regardless of pytest CWD
HERE = pathlib.Path(__file__).resolve()
PKG_DIR = HERE.parents[1]  # .../chess_tutor
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# Read config after sys.path adjustment
try:
    from chess_tutor import config as _cfg
except Exception:
    import config as _cfg  # type: ignore


def _can_run_live_test() -> tuple[bool, str]:
    """Evaluate preconditions; return (can_run, reason_if_not)."""
    enabled_env = os.environ.get("ENABLE_LLM", "").lower() in {"1", "true", "yes"}
    enabled_cfg = bool(getattr(_cfg, "ENABLE_LLM", False))
    if not (enabled_env or enabled_cfg):
        return False, "ENABLE_LLM not true (export ENABLE_LLM=true)"
    provider = (getattr(_cfg, "LLM_PROVIDER", "openai") or "openai").lower()
    if provider != "openai":
        return False, f"Unsupported provider {provider} (only openai supported)"
    try:
        import openai  # noqa: F401
    except Exception:
        return False, "Missing package 'openai' (pip install openai)"
    key_env_name = getattr(_cfg, "OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
    if not os.environ.get(key_env_name):
        return False, f"Environment variable {key_env_name} not set"
    return True, ""

def test_llm_live_responsive():
    """Calls the real LLM via LLMClient if env configured; otherwise skips with explicit reason.

    Run with: pytest -s tests/test_llm_live.py  (use -s to see print output)
    Required:
      export ENABLE_LLM=true
      export OPENAI_API_KEY=sk-... (or matching OPENAI_API_KEY_ENV)
      pip install openai
    """
    can_run, reason = _can_run_live_test()
    if not can_run:
        pytest.skip(reason)

    # Force-enable via env (won't hurt if already true)
    os.environ.setdefault("ENABLE_LLM", "true")

    # Import adapter after env set
    try:
        from backend.llm_adapter import LLMClient
    except Exception:
        from chess_tutor.backend.llm_adapter import LLMClient  # type: ignore

    print("[llm-live] Preconditions satisfied; invoking OpenAI model")
    print("[llm-live] Python:", sys.executable)
    print("[llm-live] ENABLE_LLM:", os.environ.get("ENABLE_LLM"))
    key_env_name = getattr(_cfg, "OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
    print(f"[llm-live] API key present in {key_env_name}:", bool(os.environ.get(key_env_name)))

    client = LLMClient()
    assert getattr(client, "enabled", False), "LLMClient not enabled; set ENABLE_LLM=true"

    system = "You are a helpful assistant. Reply briefly."
    user = "Say 'ok' if you can read this."
    out = client.generate(system, user, temperature=0.0, max_tokens=10)
    assert isinstance(out, str) and len(out.strip()) > 0, f"Empty/None response from live LLM: {out}"
    print("[llm-live] Live LLM response:", out[:120])
