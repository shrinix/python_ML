try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

def _pawn_attacks(board, color):
    squares = set()
    for sq in board.pieces(chess.PAWN, color):
        for t in board.attacks(sq):
            squares.add(t)
    return squares

def detect(board) -> bool:
    if chess is None:
        return False
    # A backward pawn (simplified):
    # - has no friendly pawn on adjacent files that can advance to support it (same rank or behind)
    # - its forward square (one step ahead) is controlled by an enemy pawn
    for color in (chess.WHITE, chess.BLACK):
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        enemy_pawn_ctrl = _pawn_attacks(board, enemy)
        for sq in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(sq)
            rank = chess.square_rank(sq)
            # forward square
            fwd = chess.square(file_idx, rank + (1 if color == chess.WHITE else -1)) if (0 <= rank + (1 if color == chess.WHITE else -1) <= 7) else None
            if fwd is None:
                continue
            # no friendly support from adjacent files behind or same rank
            supported = False
            for f in (file_idx-1, file_idx+1):
                if 0 <= f <= 7:
                    for p in board.pieces(chess.PAWN, color):
                        if chess.square_file(p) != f:
                            continue
                        pr = chess.square_rank(p)
                        # supporting if can advance to same rank or ahead to defend
                        if color == chess.WHITE and pr <= rank:
                            supported = True; break
                        if color == chess.BLACK and pr >= rank:
                            supported = True; break
                if supported:
                    break
            if supported:
                continue
            # enemy pawn attacks the pawn (simplified backward condition)
            if sq in enemy_pawn_ctrl:
                return True
    return False

def detect_info(board):
    """Structured info for backward pawns (simplified heuristic).

    Impact squares: pawns satisfying backward conditions.
    impacted_side: color owning first backward pawn if any.
    meta includes counts per side.
    """
    if chess is None:
        return {
            'principle': 'BackwardPawn',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    backward_white = []
    backward_black = []
    for color in (chess.WHITE, chess.BLACK):
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        enemy_pawn_ctrl = _pawn_attacks(board, enemy)
        for sq in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(sq)
            rank = chess.square_rank(sq)
            step = 1 if color == chess.WHITE else -1
            nxt_rank = rank + step
            if not (0 <= nxt_rank <= 7):
                continue
            fwd = chess.square(file_idx, nxt_rank)
            supported = False
            for f in (file_idx-1, file_idx+1):
                if 0 <= f <= 7:
                    for p in board.pieces(chess.PAWN, color):
                        if chess.square_file(p) != f:
                            continue
                        pr = chess.square_rank(p)
                        if color == chess.WHITE and pr <= rank:
                            supported = True; break
                        if color == chess.BLACK and pr >= rank:
                            supported = True; break
                if supported:
                    break
            if supported:
                continue
            # enemy pawn attacks the pawn (simplified backward condition)
            if sq in enemy_pawn_ctrl:
                if color == chess.WHITE:
                    backward_white.append(chess.square_name(sq))
                else:
                    backward_black.append(chess.square_name(sq))
    impact = backward_white + backward_black
    impacted_side = None
    if backward_white:
        impacted_side = 'white'
    elif backward_black:
        impacted_side = 'black'
    return {
        'principle': 'BackwardPawn',
        'detected': bool(impact),
        'impacted_side': impacted_side,
        'impact_squares': impact,
        'meta': {
            'white_backward': len(backward_white),
            'black_backward': len(backward_black)
        }
    }
