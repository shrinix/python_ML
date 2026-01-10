try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Human-friendly piece names matching chess library piece_type
PIECE_NAMES = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
}

def _material_score(board, color):
    score = 0
    pieces_by_type = {}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.color != color:
            continue
        val = PIECE_VALUES.get(p.piece_type, 0)
        score += val
        pieces_by_type.setdefault(p.piece_type, []).append(sq)
    return score, pieces_by_type

def detect(board):
    """Return True if one side has a material advantage (difference >=1 pawn unit)."""
    if chess is None or board is None:
        return False
    w_score, _ = _material_score(board, chess.WHITE)
    b_score, _ = _material_score(board, chess.BLACK)
    diff = w_score - b_score
    return abs(diff) >= 1  # any positive difference counts

def detect_info(board):
    """Structured info:
    impacted_side: 'white' or 'black'
    impact_squares: squares of extra material pieces for that side (limit 8)
    captured_pieces: list of piece symbols (e.g. ['Q','P','P']) representing pieces of the *opposite* color that account for the advantage.
    material_diff: signed difference in pawn units (white minus black)
    """
    if chess is None or board is None:
        return {}
    w_score, w_map = _material_score(board, chess.WHITE)
    b_score, b_map = _material_score(board, chess.BLACK)
    diff = w_score - b_score
    if diff == 0:
        return {}
    if diff > 0:
        side = 'white'
        advantaged_map = w_map
        disadvantaged_map = b_map
    else:
        side = 'black'
        advantaged_map = b_map
        disadvantaged_map = w_map
    # Build list of extra material squares: pieces of advantaged side that exceed count of opponent's same type
    extra_squares = []
    captured_list = []
    for ptype, squares in advantaged_map.items():
        opp_count = len(disadvantaged_map.get(ptype, []))
        adv_count = len(squares)
        if adv_count > opp_count:
            # squares of the surplus pieces (difference many from the tail of list)
            surplus = adv_count - opp_count
            extra_squares.extend([chess.square_name(sq) for sq in squares[-surplus:]])
            captured_list.extend([PIECE_NAMES.get(ptype, '?')] * surplus)
    # If still empty (possible when value diff caused by different piece mix), fall back to all pieces contributing to diff up to limit
    if not extra_squares:
        for ptype, squares in advantaged_map.items():
            extra_squares.extend([chess.square_name(sq) for sq in squares])
            if len(extra_squares) >= 8:
                break
    # Cap
    if len(extra_squares) > 8:
        extra_squares = extra_squares[:8]
    return {
        'detected': True,
        'impacted_side': side,
        'impact_squares': extra_squares,
        'captured_pieces': captured_list,
        'material_diff': diff,
    }

def visualize(board):  # pragma: no cover - simple overlay helper
    info = detect_info(board)
    if not info:
        return {}
    highlights = []
    for sq in info.get('impact_squares', []):
        highlights.append({'square': sq, 'color': 'yellow'})
    return {'highlights': highlights}
