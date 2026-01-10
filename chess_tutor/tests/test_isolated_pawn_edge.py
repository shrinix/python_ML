import os
import chess
from backend.principles.isolated_pawn import detect, IGNORE_EDGE

EDGE_FEN = "rnbqk2r/pp3ppp/4p3/8/1b1PP3/5N2/P4PPP/R1BQKB1R w KQkq - 1 9"

def test_isolated_pawn_edge_fen_default_ignores_a_file():
    # Ensure environment default ignores edge pawns
    assert IGNORE_EDGE, "Expected IGNORE_EDGE default True"
    board = chess.Board(EDGE_FEN)
    assert not detect(board), "Edge pawn (a2) should be ignored; detector returned True"

def test_isolated_pawn_detection_includes_edge_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("ISOLATED_IGNORE_EDGE_FILES", "0")
    # Force re-import to pick up new env flag
    import importlib, backend.principles.isolated_pawn as iso
    importlib.reload(iso)
    board = chess.Board(EDGE_FEN)
    assert iso.detect(board), "With edge ignore disabled, a2 is isolated and should be detected"
