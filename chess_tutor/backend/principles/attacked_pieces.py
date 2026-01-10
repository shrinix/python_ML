try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

RED = "#ff1a1a"


def _previous_side(board):
    # board.turn is side to move; previous mover is opposite
    return chess.BLACK if board.turn == chess.WHITE else chess.WHITE


def detect(board) -> bool:
    if chess is None or board is None:
        return False
    try:
        prev = _previous_side(board)
        # Enemy (victim) side is the opposite of prev; pieces of this side are under attack
        enemy = chess.BLACK if prev == chess.WHITE else chess.WHITE
        # Any enemy piece attacked by any prev piece?
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p or p.color != prev:
                continue
            # attacked squares by this piece
            for tsq in board.attacks(sq):
                tp = board.piece_at(tsq)
                if tp and tp.color != prev:
                    return True
        return False
    except Exception:
        return False


def detect_info(board):
    import sys
    try:
        print(f"[DEBUG] FEN: {board.fen()}", file=sys.stderr)
    except Exception:
        pass
    # Log attacks detected for this FEN
    try:
        attacked_white = []
        attacked_black = []
        attack_map = []
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p:
                continue
            for tsq in board.attacks(sq):
                tp = board.piece_at(tsq)
                if tp and tp.color != p.color:
                    attacker_name = chess.square_name(sq)
                    victim_name = chess.square_name(tsq)
                    attack_map.append((attacker_name, victim_name))
                    if tp.color == chess.WHITE:
                        attacked_white.append(victim_name)
                    else:
                        attacked_black.append(victim_name)
        print(f"[DEBUG] Attacks for FEN: {board.fen()} => {attack_map}", file=sys.stderr)
    except Exception:
        attacked_white = []
        attacked_black = []
        attack_map = []
    if chess is None or board is None:
        return {
            'principle': 'AttackedPieces',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }
    try:
        # Always return all attacked squares for both sides
        attacked_white = sorted(set(attacked_white))
        attacked_black = sorted(set(attacked_black))
        impact_squares = attacked_white + attacked_black
        impacted_side = None
        if attacked_white and attacked_black:
            impacted_side = 'both'
        elif attacked_white:
            impacted_side = 'white'
        elif attacked_black:
            impacted_side = 'black'
        return {
            'principle': 'AttackedPieces',
            'detected': bool(attack_map),
            'impacted_side': impacted_side,
            'impact_squares': impact_squares[:16],
            'meta': {
                'attacked_white': attacked_white,
                'attacked_black': attacked_black,
                'attacked_count': len(impact_squares),
                'attacks': attack_map[:32],
            }
        }
    except Exception:
        return {
            'principle': 'AttackedPieces',
            'detected': False,
            'impacted_side': None,
            'impact_squares': [],
            'meta': {}
        }


def visualize(board):  # pragma: no cover
    import sys
    print(f"[DEBUG] AttackedPieces.visualize called. FEN: {board.fen()}", file=sys.stderr)
    prev = _previous_side(board)
    pawn_sqs = [chess.square_name(sq) for sq in board.pieces(chess.PAWN, prev)]
    print(f"[DEBUG] Previous mover: {prev} ({'white' if prev == chess.WHITE else 'black'}), Pawn squares: {pawn_sqs}", file=sys.stderr)
    # Print all pieces for previous mover
    piece_sqs = [chess.square_name(sq) for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == prev]
    print(f"[DEBUG] All pieces for previous mover: {piece_sqs}", file=sys.stderr)
    info = detect_info(board)
    if chess is None or board is None:
        return {'arrows': [], 'highlights': []}
    arrows = []
    highlights = []
    debug_arrows = []
    # Always show attack arrows for previous mover if any attacks exist
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.color != prev:
            continue
        print(f"[DEBUG] Checking piece {p.symbol()} at {chess.square_name(sq)} for attacks", file=sys.stderr)
        for tsq in board.attacks(sq):
            tp = board.piece_at(tsq)
            print(f"[DEBUG] {p.symbol()} at {chess.square_name(sq)} attacks {chess.square_name(tsq)}; target piece: {tp.symbol() if tp else None}", file=sys.stderr)
            # Only generate arrow if target is enemy piece
            if tp and tp.color != p.color:
                arrows.append({'from': chess.square_name(sq), 'to': chess.square_name(tsq), 'color': RED})
                debug_arrows.append((chess.square_name(sq), chess.square_name(tsq)))
    print(f"[DEBUG] AttackedPieces arrows for prev mover {prev}: {debug_arrows}", file=sys.stderr)
    # Add highlights for all attacked/victim squares, color-coded by side
    for sq in info.get('meta', {}).get('attacked_white', []):
        highlights.append({'square': sq, 'principle': 'AttackedPieces', 'color': '#60a5faaa'})
    for sq in info.get('meta', {}).get('attacked_black', []):
        highlights.append({'square': sq, 'principle': 'AttackedPieces', 'color': '#3b82f6aa'})
    print(f"[DEBUG] AttackedPieces highlights: {highlights}", file=sys.stderr)
    return {'arrows': arrows, 'highlights': highlights}
    # Always show attack arrows for previous mover if any attacks exist
    import sys
    debug_arrows = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.color != prev:
            continue
        # For pawns, check diagonal attacks
        if p.piece_type == chess.PAWN:
            for tsq in board.attacks(sq):
                tp = board.piece_at(tsq)
                if tp and tp.color != prev:
                    arrows.append({'from': chess.square_name(sq), 'to': chess.square_name(tsq), 'color': RED})
                    debug_arrows.append((chess.square_name(sq), chess.square_name(tsq)))
        else:
            for tsq in board.attacks(sq):
                tp = board.piece_at(tsq)
                if tp and tp.color != prev:
                    arrows.append({'from': chess.square_name(sq), 'to': chess.square_name(tsq), 'color': RED})
                    debug_arrows.append((chess.square_name(sq), chess.square_name(tsq)))
    print(f"[DEBUG] AttackedPieces arrows for prev mover {prev}: {debug_arrows}", file=sys.stderr)
    # Add highlights for all attacked/victim squares, color-coded by side
    for sq in info.get('meta', {}).get('attacked_white', []):
        highlights.append({'square': sq, 'principle': 'AttackedPieces', 'color': '#60a5faaa'})
    for sq in info.get('meta', {}).get('attacked_black', []):
        highlights.append({'square': sq, 'principle': 'AttackedPieces', 'color': '#3b82f6aa'})
    return {'arrows': arrows, 'highlights': highlights}
