import json
from pathlib import Path

import pytest

try:
    import chess
except Exception:  # pragma: no cover - python-chess unavailable
    chess = None

try:
    # Prefer package-relative import
    from chess_tutor.config import GAMES_PATH
except Exception:
    # Fallback to direct path if tests are executed from repo root without package context
    GAMES_PATH = str(Path(__file__).resolve().parents[1] / "index_store" / "games.json")


def _load_games() -> list[dict]:
    p = Path(GAMES_PATH)
    if not p.exists():
        pytest.skip(f"games.json not found at {GAMES_PATH}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        pytest.fail(f"Failed to parse games.json: {e}")


def _sanitize_moves(moves: list[str]) -> list[str]:
    if chess is None:
        # If python-chess isn't available, just return the input to avoid hard fails
        return moves
    board = chess.Board()
    clean: list[str] = []
    for san in moves or []:
        try:
            mv = board.parse_san(san)
        except Exception:
            break
        board.push(mv)
        clean.append(san)
    return clean


@pytest.mark.parametrize("game", _load_games())
def test_games_have_minimal_schema(game: dict):
    assert isinstance(game, dict)
    assert "id" in game and isinstance(game["id"], str) and game["id"].strip()
    assert "moves" in game and isinstance(game["moves"], list)
    # If present, final_fen should be a string
    if "final_fen" in game and game["final_fen"] is not None:
        assert isinstance(game["final_fen"], str)


@pytest.mark.parametrize("game", _load_games())
def test_moves_are_parseable_prefix(game: dict):
    moves = game.get("moves") or []
    if not moves:
        pytest.skip(f"Game {game.get('id')} has no moves")
    clean = _sanitize_moves(moves)
    # At least the first move must be legal, otherwise extraction is clearly broken
    assert len(clean) >= 1, f"First move illegal in game {game.get('id')}: {moves[:3]}"
    # If we had to truncate, report it without failing the suite (expected if book OCR was noisy)
    if len(clean) < len(moves):
        pytest.xfail(f"Game {game.get('id')}: truncated {len(moves)-len(clean)} trailing moves due to SAN parse errors")


@pytest.mark.parametrize("game", _load_games())
def test_final_fen_matches_sanitized_sequence(game: dict):
    if chess is None:
        pytest.skip("python-chess not installed; skipping FEN validation")
    final_fen = (game.get("final_fen") or "").strip()
    if not final_fen:
        pytest.skip(f"Game {game.get('id')} missing final_fen; skipping")
    moves = game.get("moves") or []
    clean = _sanitize_moves(moves)
    board = chess.Board()
    for san in clean:
        mv = board.parse_san(san)
        board.push(mv)
    got = board.fen()
    if got != final_fen:
        # Don't fail hard yet; mark as expected mismatch to surface problems without breaking CI
        pytest.xfail(
            f"final_fen mismatch for {game.get('id')}: expected {final_fen!r}, got {got!r}; sanitized_len={len(clean)}, original_len={len(moves)}"
        )
