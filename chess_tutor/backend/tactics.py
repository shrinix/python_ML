try:
    import chess  # type: ignore
    import chess.engine  # type: ignore
except Exception:
    chess = None  # type: ignore

RED = "#ff1a1a"
YELLOW = "#eab308"
BLUE = "#3b82f6"

# Simple, fast motifs for current position:
# - Immediate checks (legal moves delivering check)
# - Mate-in-1 (legal moves delivering checkmate)
# - Hanging captures (capture undefended enemy piece)

def _move_to_from_to(board, mv):
    try:
        return chess.square_name(mv.from_square), chess.square_name(mv.to_square)
    except Exception:
        return None, None


def _is_hanging(board, target_sq):
    # target piece is undefended (no friendly defenders), but attacked by side to move
    p = board.piece_at(target_sq)
    if not p:
        return False
    color = p.color
    defenders = board.attackers(color, target_sq)
    return len(defenders) == 0


def analyze_tactics(board):
    """Return a dict with tactics list and overlay arrows/highlights.
    Each tactic: {kind, move, from, to, score?, note}
    Overlays: {arrows:[], highlights:[]}
    """
    if chess is None or board is None:
        return {"tactics": [], "overlays": {"arrows": [], "highlights": []}}
    out = {"tactics": [], "overlays": {"arrows": [], "highlights": []}}
    stm = board.turn
    # Generate legal moves once
    legal = list(board.legal_moves)
    # Mate-in-1 and checks
    for mv in legal:
        try:
            board.push(mv)
            if board.is_checkmate():
                fr, to = _move_to_from_to(board, mv)
                out["tactics"].append({
                    "kind": "MateIn1",
                    "move": board.san(board.peek()),
                    "from": fr,
                    "to": to,
                    "note": "Delivers checkmate."
                })
                if fr and to:
                    out["overlays"]["arrows"].append({"from": fr, "to": to, "color": RED})
                    out["overlays"]["highlights"].append({"square": to, "principle": "Checkmate", "color": "#b91c1cbb"})
            elif board.is_check():
                fr, to = _move_to_from_to(board, mv)
                out["tactics"].append({
                    "kind": "Check",
                    "move": board.san(board.peek()),
                    "from": fr,
                    "to": to,
                    "note": "Delivers check."
                })
                if fr and to:
                    out["overlays"]["arrows"].append({"from": fr, "to": to, "color": YELLOW})
                    out["overlays"]["highlights"].append({"square": to, "principle": "KingCheck", "color": "#ef4444aa"})
        except Exception:
            pass
        finally:
            try:
                board.pop()
            except Exception:
                pass
    # Hanging captures: capture an undefended enemy piece
    # Evaluate capture moves and mark those where destination contains undefended enemy
    for mv in legal:
        to_sq = mv.to_square
        target = board.piece_at(to_sq)
        if target and target.color != stm:
            if _is_hanging(board, to_sq):
                fr, to = _move_to_from_to(board, mv)
                try:
                    san = board.san(mv)
                except Exception:
                    san = None
                out["tactics"].append({
                    "kind": "HangingCapture",
                    "move": san,
                    "from": fr,
                    "to": to,
                    "note": "Capture undefended enemy piece."
                })
                if fr and to:
                    out["overlays"]["arrows"].append({"from": fr, "to": to, "color": BLUE})
                    out["overlays"]["highlights"].append({"square": to, "principle": "Hanging", "color": "#60a5faaa"})
    return out


def analyze_engine_lines(board, max_lines: int = 3, depth: int = 12, movetime_ms: int | None = None):
    """Use a UCI engine (Stockfish) to get multi-move tactical lines from current position.
    Returns dict with keys:
      - lines: [ { line: [SAN...], first_san, first_from, first_to, score_cp, mate } ]
      - overlays: {arrows:[...], highlights:[...]}
    If engine is unavailable, returns empty result gracefully.
    """
    if chess is None or board is None:
        return {"lines": [], "overlays": {"arrows": [], "highlights": []}}
    import os
    engine_path = os.environ.get("STOCKFISH_PATH", "stockfish")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception:
        return {"lines": [], "overlays": {"arrows": [], "highlights": []}}
    try:
        # Try to enable MultiPV
        try:
            engine.configure({"MultiPV": max(1, int(max_lines))})
            multipv = max(1, int(max_lines))
        except Exception:
            multipv = 1
        limit = chess.engine.Limit(depth=depth) if movetime_ms is None else chess.engine.Limit(time=movetime_ms/1000.0)
        infos = engine.analyse(board, limit, multipv=multipv)
        if not isinstance(infos, list):
            infos = [infos]
        # Sort by pv rank if available
        try:
            infos.sort(key=lambda i: i.get('multipv', 1))
        except Exception:
            pass
        lines = []
        overlays = {"arrows": [], "highlights": []}
        for info in infos[:max_lines]:
            pv = info.get('pv') or []
            if not pv:
                continue
            # Convert PV to SAN sequence on a copy board
            tmp = board.copy()
            san_line = []
            fens = []
            first_from = None
            first_to = None
            try:
                for idx, mv in enumerate(pv):
                    if idx == 0:
                        try:
                            first_from = chess.square_name(mv.from_square)
                            first_to = chess.square_name(mv.to_square)
                        except Exception:
                            first_from = first_from or None
                            first_to = first_to or None
                    san_line.append(tmp.san(mv))
                    tmp.push(mv)
                    # Record FEN after each move
                    try:
                        fens.append(tmp.fen())
                    except Exception:
                        fens.append('')
                score_cp = None
                mate = None
                sc = info.get('score')
                if sc is not None:
                    try:
                        # Prefer pov for side-to-move
                        pov = sc.pov(board.turn)
                        if pov.is_mate():
                            mate = pov.mate()
                        else:
                            score_cp = pov.score(mate_score=100000)
                    except Exception:
                        pass
                first_san = san_line[0] if san_line else None
                LinesItem = {
                    "line": san_line,
                    "first_san": first_san,
                    "first_from": first_from,
                    "first_to": first_to,
                    "score_cp": score_cp,
                    "mate": mate,
                    "fens": fens,
                }
                lines.append(LinesItem)
                # Add an arrow for the first move
                if first_from and first_to:
                    overlays["arrows"].append({"from": first_from, "to": first_to, "color": YELLOW})
            except Exception:
                continue
        return {"lines": lines, "overlays": overlays}
    finally:
        try:
            engine.quit()
        except Exception:
            pass
