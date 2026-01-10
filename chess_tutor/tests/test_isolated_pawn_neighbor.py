import sys
import pathlib
import os
import chess

# Ensure package root on sys.path for direct import during pytest
HERE = pathlib.Path(__file__).resolve()
PKG = HERE.parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from backend.principles.isolated_pawn import _is_isolated  # type: ignore


def test_black_d4_not_isolated_with_e6_neighbor():
    fen = 'rnbqkb1r/pp3ppp/4p3/8/3pP3/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8'
    b = chess.Board(fen)
    d4 = chess.parse_square('d4')
    assert not _is_isolated(b, chess.BLACK, d4)


def test_black_d4_isolated_when_e6_removed():
    fen = 'rnbqkb1r/pp3ppp/8/8/3pP3/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8'
    b = chess.Board(fen)
    d4 = chess.parse_square('d4')
    assert _is_isolated(b, chess.BLACK, d4)


def test_black_e6_not_isolated_with_f7_neighbor():
    fen = 'rnbqkb1r/pp3ppp/4p3/8/3PP3/5N2/P4PPP/R1BQKB1R b KQkq - 0 8'
    b = chess.Board(fen)
    e6 = chess.parse_square('e6')
    assert not _is_isolated(b, chess.BLACK, e6)
