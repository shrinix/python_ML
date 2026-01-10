try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

def _open_files(board):
    open_files = set()
    for f in range(8):
        if not any(board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN for r in range(8)):
            open_files.add(f)
    return open_files

def detect(board) -> bool:
    if chess is None:
        return False
    of = _open_files(board)
    for color in (chess.WHITE, chess.BLACK):
        for r in board.pieces(chess.ROOK, color):
            if chess.square_file(r) in of:
                return True
    return False

def detect_info(board):
    """Structured info for open files (rook presence on open file)."""
    if chess is None:
        return {
            'principle': 'OpenFiles',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    of = _open_files(board)
    rooks_white = []
    rooks_black = []
    for color in (chess.WHITE, chess.BLACK):
        for r in board.pieces(chess.ROOK, color):
            if chess.square_file(r) in of:
                if color == chess.WHITE:
                    rooks_white.append(chess.square_name(r))
                else:
                    rooks_black.append(chess.square_name(r))
    impact = rooks_white + rooks_black
    impacted_side = None
    if rooks_white:
        impacted_side = 'white'
    elif rooks_black:
        impacted_side = 'black'
    return {
        'principle': 'OpenFiles',
        'detected': bool(impact),
        'impacted_side': impacted_side,
        'impact_squares': impact,
        'meta': {
            'open_files': list(of),
            'white_rooks_on_open': len(rooks_white),
            'black_rooks_on_open': len(rooks_black)
        }
    }
