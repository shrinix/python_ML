"""Detector for the Isolated Pawn principle.

Standard definition: A pawn with no friendly pawn on adjacent files (left or right)
anywhere on the board. Edge pawns (a- or h-file) are *technically* isolated if
their only adjacent file lacks a friendly pawn. However, some pedagogical sources
exclude edge pawns from instructional tagging (they are often less critical).

We support an optional environment override:
  ISOLATED_IGNORE_EDGE_FILES=1 (default) -> do not count a/h pawns as isolated.

This addresses false-positive concerns when users expect only central / semi-central
isolated pawns to trigger the principle.
"""
import os
try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

IGNORE_EDGE = os.environ.get("ISOLATED_IGNORE_EDGE_FILES", "1").lower() in {"1", "true", "yes"}

def _is_isolated(board, color, sq) -> bool:
    """Return True if pawn on sq is isolated.

    A pawn is isolated if there is no friendly pawn on either adjacent file
    (regardless of rank distance). Edge pawns may be excluded by IGNORE_EDGE.
    """
    file_idx = chess.square_file(sq)
    if IGNORE_EDGE and file_idx in (0, 7):  # a or h file exclusion
        return False
    pawns_same_color = board.pieces(chess.PAWN, color)
    adj_files = {file_idx - 1, file_idx + 1}
    for p in pawns_same_color:
        pf = chess.square_file(p)
        if pf in adj_files:
            return False
    return True

def detect(board) -> bool:
    if chess is None or board is None:
        return False
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.pieces(chess.PAWN, color):
            if _is_isolated(board, color, sq):
                return True
    return False

def detect_info(board):
    """Return structured info for isolated pawn detection.

    Schema:
    {
      'principle': 'IsolatedPawn',
      'detected': bool,
      'impacted_side': 'white' | 'black' | None,
      'impact_squares': [algebraic...],
      'meta': { 'count': int }
    }
    """
    if chess is None or board is None:
        return {
            'principle': 'IsolatedPawn',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    isolated_white = []
    isolated_black = []
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.pieces(chess.PAWN, color):
            if _is_isolated(board, color, sq):
                (isolated_white if color == chess.WHITE else isolated_black).append(chess.square_name(sq))
    all_iso = isolated_white + isolated_black
    impacted_side = None
    if isolated_white:
        impacted_side = 'white'
    elif isolated_black:
        impacted_side = 'black'
    return {
        'principle': 'IsolatedPawn',
        'detected': bool(all_iso),
        'impacted_side': impacted_side,
        'impact_squares': all_iso,
        'meta': {
            'white_isolated': len(isolated_white),
            'black_isolated': len(isolated_black)
        }
    }

def visualize(board):
    if chess is None or board is None:
        return None
    highlights = []
    for color in (chess.WHITE, chess.BLACK):
        pawns = list(board.pieces(chess.PAWN, color))
        files = {chess.square_file(sq) for sq in pawns}
        for sq in pawns:
            f = chess.square_file(sq)
            if IGNORE_EDGE and f in (0,7):
                continue
            if (f - 1 not in files) and (f + 1 not in files):
                highlights.append({
                    "square": chess.square_name(sq),
                    "color": "#60a5faaa" if color == chess.WHITE else "#3b82f6aa"
                })
    return {"highlights": highlights}
