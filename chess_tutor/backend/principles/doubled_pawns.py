"""Detector for Doubled Pawns.

Definition: Two or more pawns of the same color on the same file (column).
This includes tripled pawns. The detector returns True if any color has
at least one file with a count of 2 or more pawns.
"""

try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore


def _doubled_groups(board):
    """Return a mapping of doubled-pawn files and squares per color.

    Returns: {
      'white': {file_index: [squares...]},
      'black': {file_index: [squares...]}
    }
    Only includes entries where len(squares) >= 2.
    """
    groups = {'white': {}, 'black': {}}
    if chess is None or board is None:
        return groups
    for color, key in ((chess.WHITE, 'white'), (chess.BLACK, 'black')):
        file_to_sqs = {}
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            file_to_sqs.setdefault(f, []).append(sq)
        for f, sqs in file_to_sqs.items():
            if len(sqs) >= 2:
                groups[key][f] = sqs
    return groups


def detect(board) -> bool:
    """True if any color has >=2 pawns on the same file.

    The position reported by the user (rnbqkb1r/pp3ppp/4p3/2pn4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6)
    has NO doubled pawns; this function correctly returns False for it.
    """
    if chess is None:
        return False
    groups = _doubled_groups(board)
    return bool(groups['white'] or groups['black'])

def detect_info(board):
    """Structured info for doubled pawns.

    Impact squares: all pawns that are part of a doubled (>=2) group.
    impacted_side: color (white/black) if that side has at least one doubled file.
    If both sides have doubled pawns we pick 'white' then include all squares.
    """
    groups = _doubled_groups(board)
    white_files = groups['white']
    black_files = groups['black']
    impact = []
    for f, sqs in white_files.items():
        impact.extend(chess.square_name(sq) for sq in sqs)
    for f, sqs in black_files.items():
        impact.extend(chess.square_name(sq) for sq in sqs)
    impacted_side = None
    if white_files:
        impacted_side = 'white'
    elif black_files:
        impacted_side = 'black'
    return {
        'principle': 'DoubledPawns',
        'detected': bool(white_files or black_files),
        'impacted_side': impacted_side,
        'impact_squares': impact,
        'meta': {
            'white_doubled_files': list(white_files.keys()),
            'black_doubled_files': list(black_files.keys())
        }
    }


def visualize(board):
    """Return overlay instructions for doubled pawns: highlight files and pawn squares."""
    if chess is None:
        return None
    groups = _doubled_groups(board)
    highlights = []
    for color, key in ((chess.WHITE, 'white'), (chess.BLACK, 'black')):
        for f, sqs in groups[key].items():
            # highlight the full file and the specific pawn squares
            for r in range(8):
                highlights.append({
                    "square": chess.square_name(chess.square(f, r)),
                    "color": "#f59e0b33" if color == chess.WHITE else "#f59e0b55"
                })
            for sq in sqs:
                highlights.append({
                    "square": chess.square_name(sq),
                    "color": "#fbbf24aa" if color == chess.WHITE else "#f59e0baa"
                })
    return {"highlights": highlights}
