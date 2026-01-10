try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

def detect(board) -> bool:
    if chess is None:
        return False
    # Bishop pair advantage: side has two bishops on opposite colors, opponent exactly one bishop
    def square_color(sq: int) -> int:
        # compute square color (0 or 1) without relying on optional API
        return (chess.square_file(sq) + chess.square_rank(sq)) & 1

    for color in (chess.WHITE, chess.BLACK):
        own = list(board.pieces(chess.BISHOP, color))
        if len(own) < 2:
            continue
        if len({square_color(sq) for sq in own}) < 2:
            continue
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        opp = list(board.pieces(chess.BISHOP, enemy))
        # advantage if opponent has fewer than two bishops
        if len(opp) < 2:
            return True
    return False

def detect_info(board):
    """Structured info for bishop pair advantage.

    Impact squares: bishops of side with the pair (opponent has exactly one bishop).
    impacted_side: side holding the bishop pair.
    meta: counts of bishops per side.
    """
    if chess is None:
        return {
            'principle': 'BishopPair',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    adv_side = None
    impact: list[str] = []
    white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))
    def square_color(sq: int) -> int:
        return (chess.square_file(sq) + chess.square_rank(sq)) & 1

    for color in (chess.WHITE, chess.BLACK):
        own = list(board.pieces(chess.BISHOP, color))
        if len(own) < 2:
            continue
        if len({square_color(sq) for sq in own}) < 2:
            continue
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        opp = list(board.pieces(chess.BISHOP, enemy))
        # advantage if opponent has fewer than two bishops
        if len(opp) < 2:
            adv_side = 'white' if color == chess.WHITE else 'black'
            impact = [chess.square_name(sq) for sq in own]
            break
    return {
        'principle': 'BishopPair',
        'detected': adv_side is not None,
        'impacted_side': adv_side,
        'impact_squares': impact,
        'meta': {
            'white_bishops': len(white_bishops),
            'black_bishops': len(black_bishops)
        }
    }
