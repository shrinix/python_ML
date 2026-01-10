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
            if passed:
                return True
    return False

def detect_info(board):
    """Structured info for passed pawns.

    Impact squares: squares of passed pawns (all, could be multiple).
    impacted_side: color of first passed pawn (white preferred if both).
    meta lists counts.
    """
    if chess is None:
        return {
            'principle': 'PassedPawn',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    passed_white = []
    passed_black = []
    for color in (chess.WHITE, chess.BLACK):
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        for sq in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(sq)
            is_passed = True
            for f in (file_idx-1, file_idx, file_idx+1):
                if not (0 <= f <= 7):
                    continue
                for p in board.pieces(chess.PAWN, enemy):
                    if chess.square_file(p) != f:
                        continue
                    if color == chess.WHITE and chess.square_rank(p) > chess.square_rank(sq):
                        is_passed = False; break
                    if color == chess.BLACK and chess.square_rank(p) < chess.square_rank(sq):
                        is_passed = False; break
                if not is_passed:
                    break
            if is_passed:
                if color == chess.WHITE:
                    passed_white.append(chess.square_name(sq))
                else:
                    passed_black.append(chess.square_name(sq))
    impact = passed_white + passed_black
    impacted_side = None
    if passed_white:
        impacted_side = 'white'
    elif passed_black:
        impacted_side = 'black'
    return {
        'principle': 'PassedPawn',
        'detected': bool(impact),
        'impacted_side': impacted_side,
        'impact_squares': impact,
        'meta': {
            'white_passed': len(passed_white),
            'black_passed': len(passed_black)
        }
    }
