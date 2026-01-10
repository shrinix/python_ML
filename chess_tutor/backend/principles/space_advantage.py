try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

def _opponent_half_squares(color):
    """Return list of squares indices in the opponent half relative to color.

    For White, opponent (Black) half is ranks 4..7 (0-indexed >=4). For Black, opponent half is ranks 0..3.
    """
    if color == chess.WHITE:
        return [sq for sq in chess.SQUARES if chess.square_rank(sq) >= 4]
    else:
        return [sq for sq in chess.SQUARES if chess.square_rank(sq) <= 3]

def _controlled_opponent_half(board, color):
    squares = _opponent_half_squares(color)
    controlled = []
    for sq in squares:
        try:
            if board.is_attacked_by(color, sq):
                controlled.append(sq)
        except Exception:
            continue
    return controlled

def detect(board) -> bool:
    """Return True if one side controls more squares in the opponent half.

    Enumerate all squares in opponent territory attacked by each side. If counts differ, space advantage exists.
    """
    if chess is None or board is None:
        return False
    try:
        white_ctrl = _controlled_opponent_half(board, chess.WHITE)
        black_ctrl = _controlled_opponent_half(board, chess.BLACK)
        diff = abs(len(white_ctrl) - len(black_ctrl))
        # Only report space advantage if the difference is at least 2
        return diff >= 2
    except Exception:
        return False

def detect_info(board):
    """Structured info: enumerate controlled squares in opponent half for each side.

    impacted_side: side with higher count ('white'/'black') or None if equal.
    impact_squares: list of controlled squares (opponent territory) for the advantaged side.
    meta: counts and diff.
    """
    if chess is None or board is None:
        return {
            'principle': 'SpaceAdvantage',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    try:
        white_ctrl = _controlled_opponent_half(board, chess.WHITE)
        black_ctrl = _controlled_opponent_half(board, chess.BLACK)
        w_count = len(white_ctrl)
        b_count = len(black_ctrl)
        diff = w_count - b_count
        abs_diff = abs(diff)
        if abs_diff < 2:
            return {
                'principle': 'SpaceAdvantage',
                'detected': False,
                'impacted_side': None,
                'impact_squares': [],
                'meta': {'white_controlled': w_count, 'black_controlled': b_count, 'diff': diff}
            }
        impacted_side = 'white' if diff > 0 else 'black'
        impact_raw = white_ctrl if diff > 0 else black_ctrl
        # Convert to square names, cap list for brevity
        impact_squares = [chess.square_name(sq) for sq in impact_raw][:16]
        return {
            'principle': 'SpaceAdvantage',
            'detected': True,
            'impacted_side': impacted_side,
            'impact_squares': impact_squares,
            'meta': {
                'white_controlled': w_count,
                'black_controlled': b_count,
                'diff': diff
            }
        }
    except Exception:
        return {
            'principle': 'SpaceAdvantage',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
