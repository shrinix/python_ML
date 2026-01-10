try:
    import chess  # type: ignore
except Exception:  # pragma: no cover
    chess = None  # type: ignore

# We'll reuse the central tactics analyzer to avoid duplicating logic.
try:
    # Prefer absolute import when package is available
    from backend.tactics import analyze_tactics  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Fallback to relative import when running within the principles package
        from ..tactics import analyze_tactics  # type: ignore
    except Exception:
        analyze_tactics = None  # type: ignore


def detect(board) -> bool:
    """Detect if there are immediate tactics available for the side to move.
    Returns True if any tactic is found by the analyzer.
    """
    if chess is None or board is None or analyze_tactics is None:
        return False
    try:
        data = analyze_tactics(board) or {}
        t = data.get("tactics") or []
        return len(t) > 0
    except Exception:
        return False


def detect_info(board):
    """Structured info for tactics detection.
    Provides a summary: count, kinds, and first few SANs and moves.
    """
    if chess is None or board is None or analyze_tactics is None:
        return {
            "principle": "Tactics",
            "detected": False,
            "impacted_side": None,
            "impact_squares": [],
            "meta": {}
        }
    try:
        data = analyze_tactics(board) or {}
        items = list(data.get("tactics") or [])
        detected = len(items) > 0
        stm_white = board.turn == chess.WHITE
        side = "white" if stm_white else "black"
        # Collect target squares (to) for quick UI hinting
        squares = []
        kinds = []
        moves = []
        for it in items[:8]:  # cap to avoid verbosity
            to_sq = (it.get("to") or "").strip()
            if to_sq:
                squares.append(to_sq)
            k = (it.get("kind") or "").strip()
            if k:
                kinds.append(k)
            mv = (it.get("move") or "").strip()
            if mv:
                moves.append(mv)
        meta = {
            "count": len(items),
            "kinds": kinds,
            "moves": moves
        }
        return {
            "principle": "Tactics",
            "detected": detected,
            "impacted_side": side if detected else None,
            "impact_squares": squares,
            "meta": meta
        }
    except Exception:
        return {
            "principle": "Tactics",
            "detected": False,
            "impacted_side": None,
            "impact_squares": [],
            "meta": {}
        }


def visualize(board):  # pragma: no cover
    """Return overlays for tactics using the central analyzer.
    Merges arrows/highlights for MateIn1, Check, HangingCapture, etc.
    """
    if chess is None or board is None or analyze_tactics is None:
        return {"arrows": [], "highlights": []}
    try:
        data = analyze_tactics(board) or {}
        overlays = data.get("overlays") or {}
        arrows = overlays.get("arrows") if isinstance(overlays, dict) else None
        highlights = overlays.get("highlights") if isinstance(overlays, dict) else None
        return {
            "arrows": arrows if isinstance(arrows, list) else [],
            "highlights": highlights if isinstance(highlights, list) else []
        }
    except Exception:
        return {"arrows": [], "highlights": []}
