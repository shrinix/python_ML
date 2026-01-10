"""Recompute and fix final_fen in games.json based on sanitized SAN moves.

Usage:
  python /Users/shriniwasiyengar/git/python_ML/chess_tutor/scripts/fix_final_fen.py
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import chess  # type: ignore
except Exception as e:  # pragma: no cover - environment constraint
    raise SystemExit(f"python-chess is required to fix final_fen: {e}")

try:
    from chess_tutor.config import GAMES_PATH
except Exception:
    GAMES_PATH = str(Path(__file__).resolve().parents[1] / "index_store" / "games.json")


def sanitize_moves(moves: list[str]) -> list[str]:
    b = chess.Board()
    clean: list[str] = []
    for san in moves or []:
        try:
            mv = b.parse_san(san)
        except Exception:
            break
        b.push(mv)
        clean.append(san)
    return clean


def main() -> int:
    p = Path(GAMES_PATH)
    if not p.exists():
        print(f"games.json not found at {GAMES_PATH}")
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    changed = 0
    for g in data:
        gid = g.get("id") or "<unknown>"
        moves = g.get("moves") or []
        clean = sanitize_moves(moves)
        b = chess.Board()
        for san in clean:
            b.push(b.parse_san(san))
        computed = b.fen()
        current = (g.get("final_fen") or "").strip()
        if current != computed:
            print(f"Fixing {gid}: final_fen {current!r} -> {computed!r}")
            g["final_fen"] = computed
            changed += 1
    if changed:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Updated {changed} games in {GAMES_PATH}")
    else:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
