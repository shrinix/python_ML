"""Game extraction logic: parse SAN moves, build FEN, save/load games."""
import os
import re
import json
from typing import List, Dict, Tuple
from pathlib import Path
try:  # Support package-relative and standalone execution
    from .config import INDEX_PATH, GAMES_PATH, EXTRACT_GAMES_VERBOSE
except Exception:  # pragma: no cover - standalone fallback
    from config import INDEX_PATH, GAMES_PATH, EXTRACT_GAMES_VERBOSE  # type: ignore
try:
    from .pdf_ingest import extract_text_from_pdf
except Exception:  # pragma: no cover - standalone fallback
    from pdf_ingest import extract_text_from_pdf  # type: ignore

try:
    import chess
    import chess.pgn as _pgn
    import io as _io
except ImportError:
    chess = None
    _pgn = None
    _io = None

# Improved SAN token regex.
# Previous pattern allowed optional piece designator, then optional file, producing
# duplicates like 'Nff3' or 'ee4' that later failed legality parsing and truncated games.
# New pattern splits alternatives explicitly:
#  - Castling: O-O / O-O-O
#  - Piece moves (with optional disambiguation file/rank, optional capture, optional promotion for underpromotions):
#    [NBRQK][a-h]?[1-8]?x?[a-h][1-8]
#  - Pawn capture: [a-h]x[a-h][1-8](=Promotion)?
#  - Pawn push: [a-h][1-8](=Promotion)?
#  - Suffix: optional check/mate + optional annotation (!, ?, !!, !?, ?!)
SAN_TOKEN_REGEX = re.compile(
    r"("  # capture each full SAN token
    r"O-O-O|O-O"  # castling (letter O only after normalization)
    r"|[NBRQK][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?"  # piece move (incl capture, promotion)
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?"  # pawn capture (incl promotion)
    r"|[a-h][1-8](?:=[QRBN])?"  # pawn push (incl promotion)
    r")"  # end capture group
    r"(?:[+#])?"  # optional check/mate
    r"(?:[!?]{1,2})?"  # optional annotation marks
)

def _normalize_castling(token: str) -> str:
    """Convert zero-based castling '0-0(-0)?' variants to 'O-O(-O)?'. Preserve suffixes.
    Many PDFs use glyph '0' instead of letter 'O'."""
    if not token:
        return token
    # Extract annotation/check suffix separately
    core = token
    # pattern: 0-0 or 0-0-0 optionally followed by + # ! ? combinations already handled later
    m = re.match(r"(0-0(-0)?)([+#]?)([!?]{1,2})?$", core)
    if not m:
        return token
    castle_part, long_flag, check_part, annot_part = m.group(1), m.group(2), m.group(3), m.group(4)
    o_form = castle_part.replace('0', 'O')
    return f"{o_form}{check_part or ''}{annot_part or ''}"
SENT_END_REGEX = re.compile(r"[.!?]\s*$")


def _normalize_game_text(txt: str) -> str:
    # remove PGN comments and parentheses
    txt = re.sub(r"\{[^}]*\}", " ", txt)
    txt = re.sub(r"\([^)]*\)", " ", txt)
    # drop diagram/caption lines like "Position after: 8... Bb4+"
    txt = re.sub(r"(?is)position\s+after\s*:.*?(?:[.!?]|$)", " ", txt)
    txt = re.sub(r"(?is)diagram\s*:.*?(?:[.!?]|$)", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def parse_algebraic_moves(text: str):
    cleaned = re.sub(r"\d+\.\.\.", "", re.sub(r"\d+.", "", text))
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
    return SAN_TOKEN_REGEX.findall(cleaned)


def extract_mainline_from_text(text: str) -> List[str]:
    """Parse mainline SAN from raw text using move numbers as anchors.
    Previous version only captured the *first* move after each numeric marker
    (white's move for '13.' and black's move for '13...') then cleared both
    expectations. This produced half-games (e.g., stopping effectively at
    move 12 when only white moves recorded). New logic:
      - After 'n.' set expect_white=True; once a white SAN captured, set expect_black=True.
      - After 'n...' set expect_black=True immediately.
      - Narrative suppression retained ("position after", "after 10...").
      - If multiple SAN tokens appear sequentially (rare), only first fulfilling expectation is taken.
    """
    tokens: List[str] = []
    for part in re.split(r"(\b\d+\...|\b\d+\.|\s+)", text):
        if not part or part.isspace():
            continue
        tokens.append(part)
    main: List[str] = []
    expect_white = False
    expect_black = False
    in_narrative = False
    for tok in tokens:
        low = tok.lower()
        if ("position after" in low) or (low.strip().startswith("after ")):
            in_narrative = True
        if SENT_END_REGEX.search(tok):
            in_narrative = False
        # Move number for white
        if re.match(r"^\d+\.$", tok):
            if in_narrative:
                expect_white = False
                expect_black = False
                continue
            expect_white = True
            expect_black = False
            continue
        # Ellipsis indicates black to move
        if re.match(r"^\d+\.\.\.$", tok):
            if in_narrative:
                expect_white = False
                expect_black = False
                continue
            expect_black = True
            expect_white = False
            continue
        tok_norm = _normalize_castling(tok)
        m = SAN_TOKEN_REGEX.fullmatch(tok_norm)
        if m:
            if expect_white:
                main.append(m.group(1))
                expect_white = False
                expect_black = True  # Prepare for black reply
                continue
            if expect_black:
                main.append(m.group(1))
                expect_black = False
                continue
    return main


def moves_to_final_fen(moves_san, start_fen=None):
    if chess is None:
        return None
    board = chess.Board(fen=start_fen) if start_fen else chess.Board()
    for san in moves_san:
        try:
            move = board.parse_san(san)
            board.push(move)
        except Exception:
            break
    return board.fen()


def _validate_mainline_legality(moves: List[str]) -> List[str]:
    """Validate sequential legality of SAN tokens.
    Revised: Do *not* truncate after streaks of illegal tokens; simply skip
    any token that cannot be parsed in the current position. This avoids
    premature game termination when narrative text or variation fragments
    leak into the mainline token list.

    Debugging aid: if env DEBUG_SAN is set, we record skipped tokens.
    """
    if chess is None or not moves:
        return moves
    import os
    debug = bool(os.environ.get('DEBUG_SAN'))
    board = chess.Board()
    out: List[str] = []
    skipped: List[str] = []
    for san in moves:
        try:
            mv = board.parse_san(san)
        except Exception:
            if debug:
                skipped.append(san)
            continue
        board.push(mv)
        out.append(san)
    if debug:
        print(f"[san-debug] accepted={len(out)} skipped={len(skipped)}")
        if skipped:
            print(f"[san-debug] skipped tokens: {' '.join(skipped[:40])}{' ...' if len(skipped)>40 else ''}")
    return out


def _normalize_castling_in_list(moves: List[str]) -> List[str]:
    """Post-process a list of SAN tokens replacing any zero-based castling variants
    that slipped through span parsing (e.g. '0-0', '0-0-0', possibly with suffixes)
    with letter 'O' forms. This is defensive: the main token regex only matches
    letter-based forms, but previously stored games or span merging can retain
    zero glyphs. We do not attempt deep SAN validation here, only a character
    substitution for the canonical tokens.
    """
    out: List[str] = []
    for san in moves:
        # Match beginning to end allowing optional check / annotation suffix
        m = re.match(r"^(0-0(?:-0)?)([+#]?)([!?]{1,2})?$", san)
        if m:
            castle, check_part, annot_part = m.group(1), m.group(2), m.group(3)
            out.append(castle.replace('0', 'O') + (check_part or '') + (annot_part or ''))
        else:
            out.append(san)
    return out


def _extract_lines_from_spans(spans: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Use span boldness to distinguish mainline vs variations.
    Returns (main_moves, variations).
    variations: list of { at_ply:int, label:str?, line:[SAN,...] }
    """
    main_tokens: List[str] = []
    variations: List[Dict] = []
    if not spans:
        return main_tokens, variations

    # 1) Merge figurine glyph spans (e.g., KNSB#20figurine "N" + next span "f3") into a single SAN token "Nf3"
    merged_spans: List[Dict] = []
    i = 0
    while i < len(spans):
        sp = spans[i] or {}
        txt = (sp.get("text") or "")
        font = (sp.get("font") or "").lower()
        is_figurine = ("figurine" in font) and (txt.strip() in {"K","Q","R","B","N"})
        if is_figurine and (i + 1) < len(spans):
            nxt = spans[i + 1] or {}
            nxt_txt = (nxt.get("text") or "")
            # Accept plausible SAN tail like "f3", "xb4+", "a5 10.", etc.
            if re.match(r"^[a-hO0-9x=+#\.\!\?\s]+$", nxt_txt, flags=re.IGNORECASE):
                combined = txt + nxt_txt
                merged_spans.append({
                    "text": combined,
                    "bold": bool(nxt.get("bold") or sp.get("bold")),
                    "font": nxt.get("font") or sp.get("font"),
                    "size": nxt.get("size") or sp.get("size"),
                })
                i += 2
                continue
        # default: keep as-is
        merged_spans.append(sp)
        i += 1

    # 2) Reconstruct text with bold flags preserved per token (split by move numbers/whitespace)
    tokens: List[Tuple[str, bool]] = []
    for sp in merged_spans:
        txt = sp.get("text") or ""
        bold = bool(sp.get("bold"))
        # Split into move-number markers and other text
        parts = re.split(r"(\b\d+\.\.\.|\b\d+\.|\s+)", txt)
        for part in parts:
            if not part or part.isspace():
                continue
            tokens.append((part, bold))
    # Build lines: sequences of SAN moves
    # Heuristic: consecutive bold SAN make mainline; non-bold SAN around move numbers build variation chunks with labels
    cur_line: List[str] = []
    ply_counter = 0
    suppress_caption = False  # ignore SAN within diagram/"Position after" captions
    last_nonspace_chunk = ""
    def flush_line():
        nonlocal cur_line, main_tokens
        if cur_line:
            main_tokens.extend(cur_line)
            cur_line = []
    # Buffer for building a variation when encountering non-bold SAN starting at a move number
    var_buf: List[str] = []
    var_label: List[str] = []
    var_at_ply: int | None = None

    def finalize_variation():
        nonlocal var_buf, var_label, var_at_ply, variations
        if var_buf:
            label = " ".join(var_label).strip() or None
            variations.append({
                "at_ply": max(0, var_at_ply or 0),
                "label": label,
                "line": list(var_buf),
            })
        var_buf = []
        var_label = []
        var_at_ply = None
    for i, (tok, is_bold) in enumerate(tokens):
        # Is this a move number marker?
        lower_tok = tok.lower()
        # Detect diagram/caption cues
        if "position after" in lower_tok or "diagram" in lower_tok:
            suppress_caption = True
        # Sentence termination may end a caption suppression
        if SENT_END_REGEX.search(tok):
            suppress_caption = False

        mnum = re.match(r"^(\d+)\.\.\.$", tok) or re.match(r"^(\d+)\.$", tok)
        if mnum:
            # Determine ply index for this move number
            move_no = int(mnum.group(1))
            # White ply index is (move_no*2 - 2), black is (move_no*2 - 1)
            is_black = tok.endswith("...")
            # Starting a new variation block if we were buffering one
            if var_buf:
                finalize_variation()
            var_at_ply = (move_no * 2 - 1) if is_black else (move_no * 2 - 2)
            # If we are in mainline accumulation and hit a non-bold section after a bold run, consider flushing
            continue
        # Extract SAN tokens from this chunk
        tok_norm = _normalize_castling(tok)
        sans = SAN_TOKEN_REGEX.findall(tok_norm)
        if not sans:
            # narrative label text
            if var_at_ply is not None:
                var_label.append(tok)
            continue
        # We have SAN(s) in this token
        if is_bold and not suppress_caption:
            # Mainline move(s)
            flush_line()
            for s in sans:
                cur_line.append(s)
                ply_counter += 1
        else:
            # Variation move(s)
            # Ignore caption-derived SANs
            if suppress_caption:
                continue
            # Only record variation if we have an explicit at-ply anchor (move number seen)
            if var_at_ply is None:
                # Heuristic: accept variations only when explicitly anchored by a move number
                # This avoids pulling SAN from narrative sentences and diagram captions
                continue
            var_buf.extend(sans)
            # If next token is a bold move number or end, finalize this variation
            # We'll finalize variations after loop
        # Reset at_ply only when we hit a new move number marker
    # Finalize mainline
    flush_line()
    # Defensive normalization of any zero-based castling that evaded earlier normalization
    main_tokens = _normalize_castling_in_list(main_tokens)
    # Legality guard: trim any accidental narrative/variation SAN that slipped in
    main_tokens = _validate_mainline_legality(main_tokens)
    # Finalize a pending variation buffer
    if var_buf:
        finalize_variation()
    return main_tokens, variations


def extract_games_from_pdf(pages: List[Dict], source_name: str, min_moves=8):
    games = []
    if not pages:
        return games
    # try to stitch per-game spans using bold mainline heuristic
    combined = "\n".join(p['content'] for p in pages)
    raw_segments = re.split(r"(?=\b1\.(?:\s|\.\.))", combined)
    gid_counter = 0
    for seg in raw_segments:
        s = seg.strip()
        if not s.startswith("1."):
            continue
        # Combine spans across pages for this segment
        seg_spans: List[Dict] = []
        # Use small anchors from the segment to find contributing pages
        head_anchor = s[:200]
        tail_anchor = s[-200:]
        for entry in pages:
            if not entry.get("spans"):
                continue
            page_txt = entry.get("content") or ""
            if not page_txt:
                continue
            found = False
            if head_anchor and head_anchor in page_txt:
                found = True
            elif tail_anchor and tail_anchor in page_txt:
                found = True
            else:
                # fallback: check a handful of initial words overlap
                words = s.split()
                probe = " ".join(words[:20]) if words else ""
                if probe and probe in page_txt:
                    found = True
            if found:
                seg_spans.extend(entry["spans"])  # associate this page's spans to the segment
        # Extract using spans if available
        main_moves: List[str] = []
        variations: List[Dict] = []
        if seg_spans:
            main_moves, variations = _extract_lines_from_spans(seg_spans)
        if not main_moves:
            # Fallback to stricter mainline parser from plain text
            norm = _normalize_game_text(s)
            main_moves = extract_mainline_from_text(norm)
        else:
            # Detect truncation: if span-derived sequence appears to be a strict prefix
            # of text-derived sequence, promote the longer text-derived version.
            norm_full = _normalize_game_text(s)
            text_moves_full = extract_mainline_from_text(norm_full)
            if main_moves and text_moves_full and len(text_moves_full) > len(main_moves):
                # Check prefix relationship
                prefix_ok = all(a == b for a, b in zip(main_moves, text_moves_full))
                if prefix_ok:
                    # Use longer sequence; spans likely missed non-bold moves.
                    main_moves = text_moves_full
                    if os.environ.get('DEBUG_SAN'):
                        print(f"[san-debug] span prefix length={len(main_moves)} upgraded to text length={len(text_moves_full)}")
        # Defensive normalization + legality guard on extracted mainline
        main_moves = _normalize_castling_in_list(main_moves)
        main_moves = _validate_mainline_legality(main_moves)
        if len(main_moves) < min_moves * 2:
            continue
        final_fen = None
        if _pgn and _io and '1.' in s:
            try:
                # Normalize zero-based castling inside raw segment for PGN reader
                s_pgn = re.sub(r'0-0(-0)?', lambda m: m.group(0).replace('0','O'), s)
                game_obj = _pgn.read_game(_io.StringIO(s_pgn))
                if game_obj:
                    board_tmp = game_obj.end().board()
                    final_fen = board_tmp.fen()
            except Exception:
                final_fen = None
        if final_fen is None:
            final_fen = moves_to_final_fen(main_moves)
        gid_counter += 1
        game_rec = {
            "id": f"{source_name}-G{gid_counter}",
            "source": source_name,
            "start_page": None,
            "end_page": None,
            "moves": main_moves,
            "final_fen": final_fen,
            "text": s
        }
        if variations:
            game_rec["variations"] = variations
        games.append(game_rec)
    if EXTRACT_GAMES_VERBOSE:
        print(f"🔍 Game extraction: {len(games)} candidates from {source_name}")
    return games


def save_games(games):
    if not games:
        return
    os.makedirs(INDEX_PATH, exist_ok=True)
    existing = []
    if os.path.isfile(GAMES_PATH):
        try:
            with open(GAMES_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    # Dedup by id
    seen = set()
    merged = []
    # Prefer newly extracted games over existing duplicates by iterating new first
    for g in games + existing:
        gid = g.get('id')
        if gid in seen:
            continue
        seen.add(gid)
        merged.append(g)
    with open(GAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"💾 Stored {len(games)} extracted games (total {len(merged)}).")


def load_games():
    results = []
    tried = []
    # Primary path under package
    tried.append(GAMES_PATH)
    if os.path.isfile(GAMES_PATH):
        try:
            with open(GAMES_PATH, "r", encoding="utf-8") as f:
                results = json.load(f) or []
        except Exception as e:
            print(f"Failed to load games.json: {e}")
    # Legacy path at repository root: ../../index_store/games.json
    if not results:
        repo_root = Path(__file__).resolve().parents[1]  # python_ML root
        legacy_path = str(repo_root / "index_store" / "games.json")
        tried.append(legacy_path)
        if os.path.isfile(legacy_path):
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    results = json.load(f) or []
                print(f"Loaded legacy games from {legacy_path}")
            except Exception as e:
                print(f"Failed to load legacy games.json: {e}")
    if not results:
        print(f"No games found. Tried: {tried}")
    return results

