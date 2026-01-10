"""Validate extracted games for move legality and final FEN consistency.

Run:
  python -m chess_tutor.scripts.validate_games
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import chess  # type: ignore
except Exception:
    chess = None  # type: ignore

try:
    from chess_tutor.config import GAMES_PATH
except Exception:
    GAMES_PATH = str(Path(__file__).resolve().parents[1] / "index_store" / "games.json")


def sanitize_moves(moves: list[str]) -> list[str]:
    if chess is None:
        return moves
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
    total = len(data)
    trunc = 0
    fen_mismatch = 0
    examples_trunc: list[str] = []
    examples_fen: list[str] = []
    for g in data:
        gid = g.get("id") or "<unknown>"
        moves = g.get("moves") or []
        clean = sanitize_moves(moves)
        if len(clean) < len(moves):
            trunc += 1
            if len(examples_trunc) < 5:
                examples_trunc.append(gid)
        fin = (g.get("final_fen") or "").strip()
        if fin and chess is not None:
            b = chess.Board()
            for san in clean:
                b.push(b.parse_san(san))
            got = b.fen()
            if got != fin:
                fen_mismatch += 1
                if len(examples_fen) < 5:
                    examples_fen.append(f"{gid} -> expected {fin}, got {got}")
    print(f"Loaded {total} games from {GAMES_PATH}")
    print(f"- Truncated move lists (due to illegal SAN tail): {trunc}")
    if examples_trunc:
        print("  e.g.", ", ".join(examples_trunc))
    print(f"- final_fen mismatches: {fen_mismatch}")
    if examples_fen:
        print("  e.g.")
        for ex in examples_fen:
            print("   ", ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
