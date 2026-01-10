"""CLI utilities to load chess content (PDF books or PGN files), extract games,
and build a principle index using the existing PrinciplesEngine.

Usage examples:

  # Parse a PDF book, extract games, detect principles, persist games + index
  python cli_loader.py --pdf path/to/book.pdf --source "MyBook"

  # Parse a PGN file containing one or more games
  python cli_loader.py --pgn path/to/game.pgn --source "MyPGN"

Artifacts written under INDEX_PATH (see config.py):
  games.json              -> merged list of extracted games
  principle_index.json    -> mapping principle_id -> list of occurrences

Each principle occurrence record contains:
  { "game_id": str, "ply": int, "fen": str, "san": str }

If a principle detector exposes richer info via detect_info() returning
fields like side / squares / captured, those are included as optional keys:
  side, squares, captured

The script intentionally does not touch the SQLite DB used by the FastAPI
backend (author endpoints) to keep it lightweight; it operates purely on
JSON artifacts for offline indexing workflows.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

from config import INDEX_PATH, GAMES_PATH  # type: ignore
from pdf_ingest import extract_text_from_pdf  # type: ignore
from game_extraction import extract_games_from_pdf, save_games  # type: ignore
from backend.principles.engine import PrinciplesEngine  # type: ignore

try:  # python-chess for PGN parsing & SAN application
    import chess
    import chess.pgn as _pgn
except Exception:  # pragma: no cover - python-chess optional
    chess = None  # type: ignore
    _pgn = None  # type: ignore

REGISTRY_PATH = Path(__file__).resolve().parent / "backend" / "principles" / "registry.json"


def load_pgn_games(path: str, source_name: str) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    if _pgn is None:
        raise RuntimeError("python-chess not installed; PGN parsing unavailable")
    with open(path, "r", encoding="utf-8") as f:
        while True:
            game = _pgn.read_game(f)
            if game is None:
                break
            moves_san: List[str] = []
            board = game.board()
            for mv in game.mainline_moves():
                san = board.san(mv)
                moves_san.append(san)
                board.push(mv)
            gid = f"{source_name}-PGN{len(games)+1}"
            games.append({
                "id": gid,
                "source": source_name,
                "moves": moves_san,
                "final_fen": board.fen(),
                "text": str(game)[:5000]  # trimmed raw PGN text for reference
            })
    return games


def build_principle_index(games: List[Dict[str, Any]], engine: PrinciplesEngine) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {pid: [] for pid in engine.list_ids()}
    if not chess:
        return index
    detectors_info = getattr(engine, 'detectors_info', {}) or {}
    for g in games:
        moves = g.get('moves') or []
        if not moves:
            continue
        board = chess.Board()
        for ply, san in enumerate(moves, start=1):
            try:
                mv = board.parse_san(san)
            except Exception:
                # Skip illegal token silently (already filtered earlier in extraction)
                continue
            board.push(mv)
            tags = engine.analyze(board)
            if not tags:
                continue
            fen = board.fen()
            for pid in tags:
                rec = {"game_id": g['id'], "ply": ply, "fen": fen, "san": san}
                # Optional richer info
                info_fn = detectors_info.get(pid)
                if info_fn:
                    try:
                        info = info_fn(board) or {}
                        # Expect keys: squares, side, captured (safe subset)
                        if 'side' in info and isinstance(info['side'], str):
                            rec['side'] = info['side']
                        if 'squares' in info and isinstance(info['squares'], list):
                            # ensure list of strings
                            rec['squares'] = [str(s) for s in info['squares'][:12]]
                        if 'captured' in info and isinstance(info['captured'], list):
                            rec['captured'] = [str(c) for c in info['captured'][:12]]
                    except Exception:
                        pass
                index.setdefault(pid, []).append(rec)
    # Optional cap to keep JSON manageable
    for pid, lst in index.items():
        if len(lst) > 2000:
            index[pid] = lst[:2000]
    return index


def persist_principle_index(index: Dict[str, List[Dict[str, Any]]]):
    os.makedirs(INDEX_PATH, exist_ok=True)
    out_path = Path(INDEX_PATH) / "principle_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"🧠 Stored principle index with {sum(len(v) for v in index.values())} occurrences at {out_path}")


def parse_pdf(path: str, source_name: str, min_moves: int) -> List[Dict[str, Any]]:
    pages = extract_text_from_pdf(path)
    games = extract_games_from_pdf(pages, source_name, min_moves=min_moves)
    return games


def main():
    ap = argparse.ArgumentParser(description="Load chess sources (PDF/PGN) and build principle index")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Path to PDF book to parse for games")
    g.add_argument("--pgn", help="Path to PGN file (one or more games)")
    ap.add_argument("--source", help="Source name label", default="Source")
    ap.add_argument("--min-moves", type=int, default=8, help="Minimum half-moves threshold to keep a game")
    ap.add_argument("--no-save-games", action="store_true", help="Do not persist games.json (principle index only)")
    ap.add_argument("--registry", help="Override principles registry path", default=str(REGISTRY_PATH))
    args = ap.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.is_file():
        raise SystemExit(f"Registry not found: {registry_path}")
    engine = PrinciplesEngine(registry_path)
    print(f"Loaded {len(engine.list_ids())} principles from {registry_path}")

    games: List[Dict[str, Any]] = []
    if args.pdf:
        print(f"📄 Parsing PDF: {args.pdf}")
        games = parse_pdf(args.pdf, args.source, args.min_moves)
    elif args.pgn:
        print(f"♟ Parsing PGN: {args.pgn}")
        games = load_pgn_games(args.pgn, args.source)
    else:
        raise SystemExit("Either --pdf or --pgn must be provided")

    if not games:
        print("No games extracted; nothing to index")
        return
    print(f"Extracted {len(games)} games; building principle index ...")
    index = build_principle_index(games, engine)
    persist_principle_index(index)
    if not args.no_save_games:
        save_games(games)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
