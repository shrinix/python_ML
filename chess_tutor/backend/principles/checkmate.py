try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

RED = "#ff1a1a"


def detect(board) -> bool:
    if chess is None or board is None:
        return False
    try:
        return board.is_checkmate()
    except Exception:
        return False


def detect_info(board):
    if chess is None or board is None:
        return {
            'principle': 'Checkmate',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    try:
        detected = board.is_checkmate()
        side = None
        squares = []
        meta = {'checkmate': bool(detected)}
        if detected:
            stm_white = board.turn == chess.WHITE
            side = 'white' if stm_white else 'black'
            king_sq = board.king(board.turn)
            if king_sq is not None:
                squares = [chess.square_name(king_sq)]
                atk = [chess.square_name(s) for s in board.attackers(not board.turn, king_sq)]
                meta['attackers'] = atk
        return {
            'principle': 'Checkmate',
            'detected': detected,
            'impacted_side': side,
            'impact_squares': squares,
            'meta': meta
        }
    except Exception:
        return {
            'principle': 'Checkmate',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }


def visualize(board):  # pragma: no cover
    if chess is None or board is None:
        return {'arrows': [], 'highlights': []}
    out = {'arrows': [], 'highlights': []}
    try:
        if not board.is_checkmate():
            return out
        king_sq = board.king(board.turn)
        if king_sq is None:
            return out
        king_name = chess.square_name(king_sq)
        # Stronger highlight for checkmated king
        out['highlights'].append({'square': king_name, 'principle': 'Checkmate', 'color': '#b91c1cbb'})
        # Draw red arrows from attackers to the king (final threats)
        for s in board.attackers(not board.turn, king_sq):
            out['arrows'].append({'from': chess.square_name(s), 'to': king_name, 'color': RED})
        return out
    except Exception:
        return {'arrows': [], 'highlights': []}
