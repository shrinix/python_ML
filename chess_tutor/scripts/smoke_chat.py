#!/usr/bin/env python3
"""Quick smoke test for /chat endpoint in chess_tutor backend.

Usage:
    python scripts/smoke_chat.py --api http://localhost:8000 \
        --question "How do I handle an isolated pawn?" \
        --include-games --include-principles

It will print the answer, whether LLM is enabled, and the number of sources.
"""
from __future__ import annotations
import argparse
import json
import sys
from urllib.parse import urljoin

try:
    import requests  # type: ignore
except Exception:
    print("Please install requests: pip install requests")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--question", default="What is an isolated pawn?")
    p.add_argument("--include-games", action="store_true")
    p.add_argument("--include-principles", action="store_true")
    p.add_argument("--game-id", default=None)
    p.add_argument("--ply", type=int, default=None)
    args = p.parse_args()

    # Fetch LLM status
    status_url = urljoin(args.api, "/admin/llm_status")
    try:
        st = requests.get(status_url, timeout=10)
        st.raise_for_status()
        status = st.json()
    except Exception as e:
        print(f"[error] failed to get LLM status: {e}")
        status = {}

    chat_url = urljoin(args.api, "/chat")
    payload = {
        "question": args.question,
        "include_games": bool(args.include_games),
        "include_principles": bool(args.include_principles),
    }
    if args.game_id:
        payload["game_id"] = args.game_id
    if args.ply is not None:
        payload["ply"] = int(args.ply)
    try:
        r = requests.post(chat_url, json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        print(f"[error] chat request failed: {e}")
        return 2

    print("=== LLM Status ===")
    print(json.dumps(status, indent=2))
    print("=== Chat Response ===")
    print(resp.get("answer", "<no answer>"))
    sources = resp.get("sources") or []
    print(f"\nSources: {len(sources)}")
    for i, s in enumerate(sources[:5], 1):
        print(f"  {i}. [{s.get('type')}] {str(s.get('meta') or '')[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
