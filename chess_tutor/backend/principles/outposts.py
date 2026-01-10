try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

def detect(board) -> bool:
    if chess is None:
        return False
    def pawn_attacks(color):
        squares = set()
        for sq in board.pieces(chess.PAWN, color):
            for t in board.attacks(sq):
                squares.add(t)
        return squares
    for color in (chess.WHITE, chess.BLACK):
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        enemy_pawn_att = pawn_attacks(enemy)
        for n in board.pieces(chess.KNIGHT, color):
            rank = chess.square_rank(n)
            if color == chess.WHITE and rank < 3:
                continue
            if color == chess.BLACK and rank > 4:
                continue
            if n not in enemy_pawn_att:
                return True
    return False

def detect_info(board):
    """Structured info for outposts (knight occupying advanced square not attacked by enemy pawn).

    Impact squares: qualifying knight squares.
    impacted_side: color of first qualifying knight (white preferred).
    meta: counts per side.
    """
    if chess is None:
        return {
            'principle': 'Outposts',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    def pawn_attacks(color):
        squares = set()
        for sq in board.pieces(chess.PAWN, color):
            for t in board.attacks(sq):
                squares.add(t)
        return squares
    outpost_white = []
    outpost_black = []
    for color in (chess.WHITE, chess.BLACK):
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        enemy_pawn_att = pawn_attacks(enemy)
        for n in board.pieces(chess.KNIGHT, color):
            rank = chess.square_rank(n)
            if color == chess.WHITE and rank < 3:
                continue
            if color == chess.BLACK and rank > 4:
                continue
            if n not in enemy_pawn_att:
                if color == chess.WHITE:
                    outpost_white.append(chess.square_name(n))
                else:
                    outpost_black.append(chess.square_name(n))
    impact = outpost_white + outpost_black
    impacted_side = None
    if outpost_white:
        impacted_side = 'white'
    elif outpost_black:
        impacted_side = 'black'
    return {
        'principle': 'Outposts',
        'detected': bool(impact),
        'impacted_side': impacted_side,
        'impact_squares': impact,
        'meta': {
            'white_outposts': len(outpost_white),
            'black_outposts': len(outpost_black)
        }
    }
