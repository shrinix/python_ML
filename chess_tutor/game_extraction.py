"""Game extraction logic: parse SAN moves, build FEN, save/load games."""
import os
import re
import json
from typing import List, Dict
from .config import INDEX_PATH, GAMES_PATH, EXTRACT_GAMES_VERBOSE
from .pdf_ingest import extract_text_from_pdf

try:
    import chess
    import chess.pgn as _pgn
    import io as _io
except ImportError:
    chess = None
    _pgn = None
    _io = None

SAN_TOKEN_REGEX = re.compile(r"(O-O-O|O-O|[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x[a-h][1-8])")


def _normalize_game_text(txt: str) -> str:
    txt = re.sub(r"\{[^}]*\}", " ", txt)
    txt = re.sub(r"\([^)]*\)", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def parse_algebraic_moves(text: str):
    cleaned = re.sub(r"\d+\.\.\.", "", re.sub(r"\d+.", "", text))
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
    return SAN_TOKEN_REGEX.findall(cleaned)


def moves_to_final_fen(moves_san, start_fen=None):
    if chess is None:
        return None
    board = chess.Board(fen=start_fen) if start_fen else chess.Board()
    for san in moves_san:
        try:
            move = board.parse_san(san)
            board.push(move)
        except Exception:
            break
    return board.fen()


def extract_games_from_pdf(pages: List[Dict], source_name: str, min_moves=8):
    games = []
    if not pages:
        return games
    combined = "\n".join(p['content'] for p in pages)
    raw_segments = re.split(r"(?=\b1\.(?:\s|\.\.))", combined)
    gid_counter = 0
    for seg in raw_segments:
        s = seg.strip()
        if not s.startswith("1."):
            continue
        norm = _normalize_game_text(s)
        san_tokens = SAN_TOKEN_REGEX.findall(norm)
        if len(san_tokens) < min_moves * 2:
            continue
        final_fen = None
        if _pgn and _io and '1.' in norm:
            try:
                game_obj = _pgn.read_game(_io.StringIO(norm))
                if game_obj:
                    board_tmp = game_obj.end().board()
                    final_fen = board_tmp.fen()
            except Exception:
                final_fen = None
        if final_fen is None:
            final_fen = moves_to_final_fen(san_tokens)
        gid_counter += 1
        games.append({
            "id": f"{source_name}-G{gid_counter}",
            "source": source_name,
            "start_page": None,
            "end_page": None,
            "moves": san_tokens,
            "final_fen": final_fen
        })
    if EXTRACT_GAMES_VERBOSE:
        print(f"🔍 Game extraction: {len(games)} candidates from {source_name}")
    return games


def save_games(games):
    if not games:
        return
    os.makedirs(INDEX_PATH, exist_ok=True)
    existing = []
    if os.path.isfile(GAMES_PATH):
        try:
            with open(GAMES_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing_ids = {g.get('id') for g in existing}
    merged = existing + [g for g in games if g.get('id') not in existing_ids]
    with open(GAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"💾 Stored {len(games)} extracted games (total {len(merged)}).")


def load_games():
    if os.path.isfile(GAMES_PATH):
        try:
            with open(GAMES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load games.json: {e}")
    return []

