try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

def detect(board) -> bool:
    if chess is None:
        return False
    for color in (chess.WHITE, chess.BLACK):
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        for sq in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(sq)
            # is passed?
            passed = True
            for f in (file_idx-1, file_idx, file_idx+1):
                if not (0 <= f <= 7):
                    continue
                for p in board.pieces(chess.PAWN, enemy):
                    if chess.square_file(p) != f:
                        continue
                    if color == chess.WHITE and chess.square_rank(p) > chess.square_rank(sq):
                        passed = False; break
                    if color == chess.BLACK and chess.square_rank(p) < chess.square_rank(sq):
                        passed = False; break
                if not passed:
                    break
            if not passed:
                continue
            # rook behind pawn
            for r in board.pieces(chess.ROOK, color):
                if chess.square_file(r) != file_idx:
                    continue
                if color == chess.WHITE and chess.square_rank(r) < chess.square_rank(sq):
                    return True
                if color == chess.BLACK and chess.square_rank(r) > chess.square_rank(sq):
                    return True
    return False

def detect_info(board):
    """Structured info for rook behind passed pawn principle.

    Impact squares: passed pawn(s) plus supporting rook(s) behind them.
    impacted_side: color of first qualifying passed pawn + rook pair.
    """
    if chess is None:
        return {
            'principle': 'RookBehindPassedPawn',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    impact_white = []
    impact_black = []
    for color in (chess.WHITE, chess.BLACK):
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        for sq in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(sq)
            passed = True
            for f in (file_idx-1, file_idx, file_idx+1):
                if not (0 <= f <= 7):
                    continue
                for p in board.pieces(chess.PAWN, enemy):
                    if chess.square_file(p) != f:
                        continue
                    if color == chess.WHITE and chess.square_rank(p) > chess.square_rank(sq):
                        passed = False; break
                    if color == chess.BLACK and chess.square_rank(p) < chess.square_rank(sq):
                        passed = False; break
                if not passed:
                    break
            if not passed:
                continue
            for r in board.pieces(chess.ROOK, color):
                if chess.square_file(r) != file_idx:
                    continue
                if color == chess.WHITE and chess.square_rank(r) < chess.square_rank(sq):
                    sqs = [chess.square_name(sq), chess.square_name(r)]
                    impact_white.extend(sqs)
                if color == chess.BLACK and chess.square_rank(r) > chess.square_rank(sq):
                    sqs = [chess.square_name(sq), chess.square_name(r)]
                    impact_black.extend(sqs)
    impact = impact_white + impact_black
    impacted_side = None
    if impact_white:
        impacted_side = 'white'
    elif impact_black:
        impacted_side = 'black'
    return {
        'principle': 'RookBehindPassedPawn',
        'detected': bool(impact),
        'impacted_side': impacted_side,
        'impact_squares': impact,
        'meta': {
            'white_pairs': len(impact_white) // 2,
            'black_pairs': len(impact_black) // 2
        }
    }
