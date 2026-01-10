#!/usr/bin/env python3
"""Unified principle testing CLI.

Key capabilities:
  • List registry principles (--list)
  • Run detectors on provided FENs (--fen / --fen-file / --auto-fens)
  • Exercise all principles (--all) or a subset (--principle ... repeatable)
  • Run a JSON truthy/falsy suite (--run-json <path>)
  • Invoke a raw module/func bypassing registry (--module <mod> [--func detect])
  • Built‑in regression mini‑suite (--builtin) validating structured outputs
  • Visual overlays / highlighted squares (--highlights, --style, --ansi, --unicode)

Structured detector contract (detect_info):
  { "principle": <id>, "detected": bool, "impacted_side": "white"|"black"|None,
     "impact_squares": ["e4", ...], "meta": {...} }

Legacy detectors returning bool are auto‑wrapped to this structure when displayed.
Exit status is non‑zero if JSON or built‑in suites record failures.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
import pytest
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set, Any

# Make sure we can import the backend principles package when running this script directly
HERE = Path(__file__).resolve()
# tests directory is one level under project root; climb two levels if layout changes
PROJECT_ROOT = HERE.parents[1]  # .../chess_tutor
BACKEND_DIR = PROJECT_ROOT / "backend"
PRINCIPLES_DIR = BACKEND_DIR / "principles"
for p in (PROJECT_ROOT, BACKEND_DIR, PRINCIPLES_DIR):
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

try:
    import chess  # type: ignore
except Exception as e:  # pragma: no cover
    print("python-chess not installed. pip install python-chess", file=sys.stderr)
    raise

# Import the engine (registry-based loader) from backend.principles
try:
    from backend.principles.engine import PrinciplesEngine  # type: ignore
except Exception as e:
    print(f"Failed to import PrinciplesEngine: {e}\n  sys.path: {sys.path[:5]}...\n  PROJECT_ROOT: {PROJECT_ROOT}\n  BACKEND_DIR exists? {BACKEND_DIR.exists()}\n  PRINCIPLES_DIR exists? {PRINCIPLES_DIR.exists()}", file=sys.stderr)
    raise

# Registry lives under backend/principles/registry.json
REGISTRY_PATH = PRINCIPLES_DIR / "registry.json"

# Built-in structured test cases: each entry contains a principle ID, a FEN and the expected boolean result.
# This lightweight suite supplements pytest regression tests and can be used ad-hoc from the CLI and by pytest.
TEST_CASES = [
    {"principle": "DoubledPawns", "fen": "8/8/8/8/8/2P5/2P5/8 w - - 0 1", "expected": True,  "impact_contains": ["c2","c3"], "impacted_side": "white"},
    {"principle": "DoubledPawns", "fen": "8/8/8/8/8/8/2P5/8 w - - 0 1", "expected": False, "impact_contains": [], "impacted_side": None},
    {"principle": "IsolatedPawn", "fen": "8/8/8/8/3P4/8/8/8 w - - 0 1", "expected": True,  "impact_contains": ["d4"], "impacted_side": "white"},
    {"principle": "IsolatedPawn", "fen": "8/8/8/8/3P4/4P3/8/8 w - - 0 1", "expected": False, "impact_contains": [], "impacted_side": None},
    {"principle": "BackwardPawn", "fen": "8/8/8/4p3/3P4/8/8/8 w - - 0 1", "expected": True,  "impact_contains": ["d4"], "impacted_side": "white"},
    {"principle": "BackwardPawn", "fen": "8/8/8/8/3P4/8/8/8 w - - 0 1", "expected": False, "impact_contains": [], "impacted_side": None},
    {"principle": "PassedPawn", "fen": "8/8/4P3/8/8/8/8/8 w - - 0 1", "expected": True,  "impact_contains": ["e6"], "impacted_side": "white"},
    {"principle": "PassedPawn", "fen": "8/4p3/4P3/8/8/8/8/8 w - - 0 1", "expected": False, "impact_contains": ["e6"], "impacted_side": None},
    {"principle": "RookBehindPassedPawn", "fen": "8/8/4P3/8/8/8/8/4R3 w - - 0 1", "expected": True,  "impact_contains": ["e6","e1"], "impacted_side": "white"},
    {"principle": "RookBehindPassedPawn", "fen": "8/8/4P3/8/8/8/8/8 w - - 0 1", "expected": False, "impact_contains": ["e6"], "impacted_side": None},
    {"principle": "OpenFiles", "fen": "8/8/8/8/8/8/8/R7 w - - 0 1", "expected": True,  "impact_contains": ["a1"], "impacted_side": "white"},
    {"principle": "OpenFiles", "fen": "8/8/8/8/8/8/P7/R7 w - - 0 1", "expected": False, "impact_contains": ["a1"], "impacted_side": None},
    {"principle": "Outposts", "fen": "8/8/8/3N4/8/8/8/8 w - - 0 1", "expected": True,  "impact_contains": ["d5"], "impacted_side": "white"},
    {"principle": "Outposts", "fen": "8/8/2p5/3N4/8/8/8/8 w - - 0 1", "expected": False, "impact_contains": ["d5"], "impacted_side": None},
    {"principle": "SpaceAdvantage", "fen": "r1bq1rk1/pp1nppbp/3p1np1/2pP4/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1", "expected": True,  "space_advantage": "white", "impacted_side": "white"},
    {"principle": "SpaceAdvantage", "fen": "8/8/8/8/8/8/8/8 w - - 0 1", "expected": False, "space_advantage": None, "impacted_side": None},
    {"principle": "BishopPair", "fen": "8/8/8/8/8/8/8/3BB2k w - - 0 1", "expected": True,  "impact_contains": ["d1","e1"], "impacted_side": "white"},
    {"principle": "BishopPair", "fen": "8/8/8/8/8/8/8/3B1B1k w - - 0 1", "expected": False, "impact_contains": ["d1","f1"], "impacted_side": None},
]

@dataclass
class Detector:
    id: str
    run: Callable[[chess.Board], Any]  # bool or structured dict
    info: Optional[Callable[[chess.Board], Dict[str, Any]]] = None  # detect_info if available


def load_detectors_from_registry(ids: Optional[List[str]] = None) -> Tuple[List[Detector], PrinciplesEngine]:
    """Return detectors plus the instantiated engine (to reuse for visualize)."""
    eng = PrinciplesEngine(REGISTRY_PATH)
    specs = [s for s in eng.list_specs() if (not ids or s.id in ids)]
    out: List[Detector] = []
    for s in specs:
        fn = eng.detectors.get(s.id)
        if not fn:
            continue
        info_fn = eng.detectors_info.get(s.id)
        out.append(Detector(id=s.id, run=fn, info=info_fn))
    return out, eng


def load_detector_from_module(module: str, func: str = "detect", id_hint: Optional[str] = None) -> Detector:
    mod = import_module(f"backend.principles.{module}")
    fn = getattr(mod, func)
    info_fn = getattr(mod, 'detect_info', None)
    return Detector(id=id_hint or module, run=fn, info=info_fn if callable(info_fn) else None)


def iter_fens_from_file(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield s


def evaluate(detectors: List[Detector], fens: List[str]) -> List[Tuple[str, Dict[str, Dict[str, Any]]]]:
    """Return per FEN a mapping principle->structured result.
    Ensures every value is a dict with keys: detected, impacted_side, impact_squares, meta.
    """
    out: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    for fen in fens:
        try:
            board = chess.Board(fen)
        except Exception:
            out.append((fen, {"__error__": {"detected": False, "error": True}}))
            continue
        row: Dict[str, Dict[str, Any]] = {}
        for d in detectors:
            structured: Dict[str, Any]
            try:
                if d.info:
                    structured = d.info(board)
                else:
                    raw = d.run(board)
                    if isinstance(raw, dict):  # detector upgraded detect()
                        structured = raw
                    else:
                        structured = {
                            "principle": d.id,
                            "detected": bool(raw),
                            "impacted_side": None,
                            "impact_squares": [],
                            "meta": {},
                        }
                # Normalize / fill missing keys
                structured.setdefault("principle", d.id)
                structured.setdefault("detected", False)
                structured.setdefault("impacted_side", None)
                structured.setdefault("impact_squares", [])
                structured.setdefault("meta", {})
            except Exception as e:
                structured = {"principle": d.id, "detected": False, "error": str(e), "impacted_side": None, "impact_squares": [], "meta": {}}
            row[d.id] = structured
        out.append((fen, row))
    return out


def board_ascii(
    board: chess.Board,
    use_unicode: bool = False,
    with_coords: bool = True,
    highlight_squares: Optional[Set[int]] = None,
    style: str = "bold",
    ansi: bool = False,
) -> str:
    """Render a simple ASCII (or Unicode) board with optional coordinates.

    ASCII uses letters KQRBNP (white) / kqrbnp (black). Empty squares are dots.
    Unicode uses chess unicode characters via board.unicode().
    """
    # For now, we don't inject highlights into unicode rendering.
    if use_unicode:
        return board.unicode(borders=True, invert_color=False)
    piece_to_char = {
        chess.KING:   ('K', 'k'),
        chess.QUEEN:  ('Q', 'q'),
        chess.ROOK:   ('R', 'r'),
        chess.BISHOP: ('B', 'b'),
        chess.KNIGHT: ('N', 'n'),
        chess.PAWN:   ('P', 'p'),
    }
    highlight_squares = highlight_squares or set()
    use_bracket = (style == "bracket")
    cell_w = 3 if use_bracket else 1  # [X] vs X
    def fmt_cell(ch: str, sq: int) -> str:
        if sq in highlight_squares:
            if use_bracket:
                return f"[{ch}]"
            if ansi:
                # bold blue
                return f"\033[1;34m{ch}\033[0m"
        return ch
    rows: List[str] = []
    for rank in range(7, -1, -1):  # 8..1
        line: List[str] = []
        for file in range(8):
            sq = chess.square(file, rank)
            p = board.piece_at(sq)
            if not p:
                ch = '.'
            else:
                up, low = piece_to_char[p.piece_type]
                ch = up if p.color == chess.WHITE else low
            line.append(fmt_cell(ch, sq))
        if with_coords:
            rows.append(f"{rank+1} " + ' '.join(line))
        else:
            rows.append(' '.join(line))
    footer = '  a b c d e f g h' if with_coords else ''
    return '\n'.join(rows + ([footer] if with_coords else []))


def _visualize_squares_for(eng: 'PrinciplesEngine', board: 'chess.Board', pid: str) -> List[str]:
    try:
        overlays = eng.visualize(board, [pid])
    except Exception:
        return []
    highs = overlays.get("highlights") or []
    out: List[str] = []
    for h in highs:
        if isinstance(h, dict) and h.get("square"):
            out.append(h["square"])
    return out


def run_builtin_suite(engine: PrinciplesEngine, cases: List[Dict[str, Any]], show_board: bool = False, highlights: bool = False, style: str = "bold", ansi: bool = False, use_unicode: bool = False) -> int:
    """Execute built-in regression cases, printing PASS/FAIL lines.
    Returns failure count.
    """
    print("Running built-in test cases:\n")
    failures = 0
    for case in cases:
        pid = case["principle"]
        fen = case["fen"]
        expected = bool(case.get("expected"))
        expect_side = case.get("impacted_side")
        expect_contains = case.get("impact_contains", [])
        try:
            board = chess.Board(fen)
        except Exception as e:
            print(f"{pid:16} ERROR   fen_parse ({e}) FEN={fen}")
            failures += 1
            continue
        info_fn = engine.detectors_info.get(pid)
        det_fn = engine.detectors.get(pid)
        if not det_fn:
            print(f"{pid:16} MISSING detector FEN={fen}")
            failures += 1
            continue
        # Acquire structured info
        if info_fn:
            try:
                info = info_fn(board)
            except Exception as e:
                print(f"{pid:16} ERROR   info_fn ({e}) FEN={fen}")
                failures += 1
                continue
        else:
            try:
                raw = det_fn(board)
            except Exception as e:
                print(f"{pid:16} ERROR   detect ({e}) FEN={fen}")
                failures += 1
                continue
            if isinstance(raw, dict):
                info = raw
            else:
                info = {"principle": pid, "detected": bool(raw), "impact_squares": [], "impacted_side": None, "meta": {}}
        detected = bool(info.get("detected"))
        impact_squares = list(info.get("impact_squares") or [])
        impacted_side = info.get("impacted_side")
        status = "PASS" if detected == expected else "FAIL"
        # Validate side expectation
        if expect_side != impacted_side:
            if expect_side is not None or impacted_side is not None:
                status = "FAIL"
        # Validate squares containment (only when expected True and we have expected list)
        if expected and expect_contains:
            missing = [sq for sq in expect_contains if sq not in impact_squares]
            if missing:
                status = "FAIL"
                # append missing info to printout
                impact_squares = impact_squares + [f"(missing:{','.join(missing)})"]
        # Count a single failure per case (avoid double counting)
        if status == "FAIL":
            failures += 1
        sq_txt = f" [{', '.join(impact_squares)}]" if impact_squares else ""
        side_txt = f" side={impacted_side or '-'}"
        extra = ""
        if expected and expect_contains:
            extra = f" contains={expect_contains}"
        print(f"{pid:16} exp={str(expected):<5} act={str(detected):<5} {status}{sq_txt}{side_txt}{extra}  FEN={fen}")
        if show_board:
            # Choose highlight squares either from visualizer (if requested) or from impact_squares
            hl_set: Set[int] = set()
            if highlights:
                # Prefer impact_squares; fall back to visualizer
                if impact_squares:
                    for sq in impact_squares:
                        try:
                            hl_set.add(chess.parse_square(sq))
                        except Exception:
                            pass
                else:
                    for sq in _visualize_squares_for(engine, board, pid):
                        try:
                            hl_set.add(chess.parse_square(sq))
                        except Exception:
                            pass
            use_ansi = ansi or (highlights and style == "bold")
            print(board_ascii(board, use_unicode=use_unicode, highlight_squares=hl_set, style=style, ansi=use_ansi))
    print(f"\nBuiltin summary: {len(cases)} cases, failures={failures}\n")
    return failures


def run_json_suite(path: Path) -> int:
    """Run a test suite JSON file.
    Format:
    {
      "DoubledPawns": { "true": [fen, ...], "false": [fen, ...] },
      "BackwardPawn": { "true": [...], "false": [...] }
    }
    Returns the number of failures.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    fails = 0
    for pid, spec in data.items():
        dets = load_detectors_from_registry([pid])
        if not dets:
            print(f"[WARN] Principle {pid} not found in registry or detector missing")
            continue
        det = dets[0]
        for truthy in spec.get("true", []) or []:
            total += 1
            ok = evaluate([det], [truthy])[0][1].get(det.id, False)
            if not ok:
                print(f"[FAIL] {pid} expected True, got False\n    FEN: {truthy}")
                fails += 1
        for falsy in spec.get("false", []) or []:
            total += 1
            ok = evaluate([det], [falsy])[0][1].get(det.id, False)
            if ok:
                print(f"[FAIL] {pid} expected False, got True\n    FEN: {falsy}")
                fails += 1
    print(f"Run {total} assertions. Failures: {fails}.")
    return fails


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Test principle detectors against FEN positions")
    gsrc = p.add_mutually_exclusive_group()
    gsrc.add_argument("--principle", action="append", help="Principle ID from registry (can repeat)")
    gsrc.add_argument("--all", action="store_true", help="Test all registry principles")
    gsrc.add_argument("--module", help="Test a detector module under principles (e.g., doubled_pawns)")
    p.add_argument("--func", default="detect", help="Function name in module (default: detect)")
    p.add_argument("--id", help="Optional ID label when using --module")

    p.add_argument("--fen", action="append", help="FEN string (can repeat)")
    p.add_argument("--fen-file", type=Path, help="File with one FEN per line (comments with #)")
    p.add_argument("--run-json", type=Path, help="Run a JSON test suite and exit nonzero on failures")
    p.add_argument("--list", action="store_true", help="List available principles from registry and exit")
    p.add_argument("--board", dest="board", action="store_true", help="Show ASCII/Unicode board for each FEN")
    p.add_argument("--no-board", dest="board", action="store_false", help="Do not show board for each FEN")
    p.add_argument("--unicode", dest="use_unicode", action="store_true", help="Use Unicode chess symbols for board output")
    p.add_argument("--highlights", action="store_true", help="Show highlight squares from visualizers for detected principles")
    p.add_argument("--ansi", action="store_true", help="Use ANSI colors/bold to emphasize highlighted squares (ASCII mode)")
    p.add_argument("--style", choices=["bold", "bracket"], default="bold", help="Highlight style in ASCII mode")
    p.add_argument("--auto-fens", action="store_true", help="Automatically load all .fen files under tests/data directory when no --fen provided")
    p.add_argument("--builtin", action="store_true", help="Run built-in regression test cases (principle/FEN/expected)")
    p.set_defaults(board=True, use_unicode=False, ansi=False)

    args = p.parse_args(argv)

    if args.list:
        dets, _eng = load_detectors_from_registry()
        if not dets:
            print("No detectors loaded from registry.")
            return 0
        print("Available principles:")
        for d in dets:
            print(f" - {d.id}")
        return 0

    if args.run_json:
        return 1 if run_json_suite(args.run_json) else 0

    # Built-in cases defined at module scope in TEST_CASES

    # Build list of detectors
    detectors: List[Detector] = []
    engine: Optional[PrinciplesEngine] = None
    if args.module:
        detectors = [load_detector_from_module(args.module, func=args.func, id_hint=args.id)]
    else:
        if args.all or not args.principle:
            detectors, engine = load_detectors_from_registry()
        else:
            detectors, engine = load_detectors_from_registry(args.principle)
    if not detectors:
        print("No detectors to run. Use --list to see available principles.")
        return 1
    if engine is None:
        # create one for visualization even for module-only mode (may not find visualize though)
        engine = PrinciplesEngine(REGISTRY_PATH)

    # Gather FENs (skip requirement when only running --builtin)
    fens: List[str] = []
    if args.fen:
        fens.extend(args.fen)
    if args.fen_file:
        fens.extend(iter_fens_from_file(args.fen_file))
    if not fens and not args.builtin:
        # Auto FEN mode: discover tests/data/**/*.fen and include ALL valid FEN lines in each file
        if args.auto_fens:
            data_dir = PROJECT_ROOT / "tests" / "data"
            attempted_paths = []
            if data_dir.exists():
                for fen_file in sorted(data_dir.rglob("*.fen")):
                    attempted_paths.append(str(fen_file))
                    try:
                        for line in fen_file.read_text(encoding="utf-8").splitlines():
                            s = line.strip()
                            if not s or s.startswith('#'):
                                continue
                            # Validate FEN by trying to instantiate board
                            try:
                                chess.Board(s)
                                fens.append(s)
                            except Exception:
                                pass
                    except Exception:
                        continue
            else:
                attempted_paths.append(str(data_dir))
        if not fens and not args.builtin:
            print("[ERROR] No FENs discovered.")
            if args.auto_fens:
                print("  Auto-fens looked under:")
                for p in attempted_paths[:10]:
                    print("   -", p)
                if not (PROJECT_ROOT / "tests" / "data").exists():
                    print("  Hint: create directory 'tests/data' and add one or more .fen files containing FEN lines.")
            print("Provide at least one FEN via --fen / --fen-file, or use --auto-fens with .fen files in tests/data, or --run-json, or use --builtin for internal suite.")
            return 1
        else:
            print(f"[INFO] Loaded {len(fens)} FEN position(s) from tests/data")

    # Built-in suite execution (runs before regular FEN evaluation if both requested)
    if args.builtin:
        builtin_engine = PrinciplesEngine(REGISTRY_PATH)
        failures = run_builtin_suite(builtin_engine, TEST_CASES, show_board=args.board, highlights=args.highlights, style=args.style, ansi=args.ansi, use_unicode=args.use_unicode)
        if not any([args.fen, args.fen_file, args.auto_fens]):
            return 1 if failures else 0

    # Evaluate selected set
    # Evaluate user-provided FENs
    if fens:
        results = evaluate(detectors, fens)
        # For overlay we want all registry principles, not only selected
        all_dets, _tmp_eng = load_detectors_from_registry()
        all_ids = [d.id for d in all_dets]
        for fen, row in results:
            print("\nFEN:", fen)
            try:
                board = chess.Board(fen)
            except Exception:
                print("  [error parsing FEN]")
                continue
            # Overlay squares: union of visualizer highlights for all detected principles
            overlay_squares: Set[int] = set()
            if args.highlights:
                active = [pid for pid, info in row.items() if info.get("detected")]
                if active:
                    try:
                        overlays = engine.visualize(board, active)
                    except Exception:
                        overlays = {}
                    for h in (overlays.get("highlights") or []):
                        if isinstance(h, dict) and h.get("square"):
                            try:
                                overlay_squares.add(chess.parse_square(h["square"]))
                            except Exception:
                                pass
                # Fall back to impact_squares if no visualizers yielded output
                if not overlay_squares:
                    for info in row.values():
                        if info.get("detected"):
                            for sq in info.get("impact_squares") or []:
                                try:
                                    overlay_squares.add(chess.parse_square(sq))
                                except Exception:
                                    pass
            if args.board:
                use_ansi = args.ansi or (args.highlights and args.style == "bold")
                print(board_ascii(board, use_unicode=args.use_unicode, highlight_squares=overlay_squares, style=args.style, ansi=use_ansi))
            print("  Principles (selected):")
            for det_id, info in row.items():
                sqs = info.get("impact_squares") or []
                if args.highlights and not sqs:
                    sqs = _visualize_squares_for(engine, board, det_id)
                suffix = f" [{', '.join(sqs)}]" if sqs else ""
                side = info.get("impacted_side") or '-'
                print(f"  {det_id:>24}: {str(info.get('detected')):<5} side={side}{suffix}")
            # Also run complete list for cross-check (only if not already full set)
            if set([d.id for d in detectors]) != set(all_ids):
                print("  Principles (all):")
                # Build structured results for all
                all_struct = evaluate(all_dets, [fen])[0][1]
                for det_id in all_ids:
                    info = all_struct.get(det_id, {})
                    sqs = info.get("impact_squares") or []
                    if args.highlights and not sqs:
                        sqs = _visualize_squares_for(engine, board, det_id)
                    suffix = f" [{', '.join(sqs)}]" if sqs else ""
                    side = info.get("impacted_side") or '-'
                    print(f"  {det_id:>24}: {str(info.get('detected')):<5} side={side}{suffix}")
    return 0


#############################################
# Pytest wrappers consolidating test coverage
#############################################

# 1) Built-in structured test cases should all pass
def test_builtin_cases_pass():
    engine = PrinciplesEngine(REGISTRY_PATH)
    # Silent run (no board/highlights), assert zero failures
    failures = run_builtin_suite(engine, TEST_CASES, show_board=False, highlights=False)
    assert failures == 0, f"Builtin suite had {failures} failures"


# 2) Simple boolean cases (legacy style), consolidated from test_all_principles.py
CASES_SIMPLE = [
    (
        "backend.principles.doubled_pawns",
        [
            "8/8/8/8/8/2P5/2P5/8 w - - 0 1",  # white doubled on c-file
            "8/p7/p7/8/8/8/8/8 b - - 0 1",    # black doubled on a-file (stacked on a7+a6)
        ],
        [
            "8/8/8/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/8/2P5/8 w - - 0 1",
        ],
    ),
    (
        "backend.principles.isolated_pawn",
        [
            "8/8/8/8/3P4/8/8/8 w - - 0 1",  # white isolated pawn on d4
            "8/8/8/3p4/8/8/8/8 b - - 0 1",  # black isolated pawn on d5
        ],
        [
            # white d4 supported by e-pawn behind
            "8/8/8/8/3P4/4P3/8/8 w - - 0 1",
        ],
    ),
    (
        "backend.principles.backward_pawn",
        [
            "8/8/8/4p3/3P4/8/8/8 w - - 0 1",  # white d4, black pawn e5 controls d4->d5
        ],
        [
            "8/8/8/8/3P4/8/8/8 w - - 0 1",
        ],
    ),
    (
        "backend.principles.passed_pawn",
        [
            "8/8/4P3/8/8/8/8/8 w - - 0 1",  # white e6 with no enemy pawns ahead
            "8/8/8/8/8/4p3/8/8 b - - 0 1",  # black e3 passed
        ],
        [
            "8/4p3/4P3/8/8/8/8/8 w - - 0 1",  # white e6 blocked by black e7
        ],
    ),
    (
        "backend.principles.rook_behind_passed",
        [
            "8/8/4P3/8/8/8/8/4R3 w - - 0 1",  # white rook behind passed pawn on e-file
        ],
        [
            "8/8/4P3/8/8/8/8/8 w - - 0 1",    # no rook
            "8/8/4P3/8/8/8/8/3R4 w - - 0 1",  # rook not behind (ahead or different file)
        ],
    ),
    (
        "backend.principles.open_files",
        [
            "8/8/8/8/8/8/8/R7 w - - 0 1",  # rook on open a-file
        ],
        [
            "8/8/8/8/8/8/P7/R7 w - - 0 1",  # a-file not open (white pawn a2)
        ],
    ),
    (
        "backend.principles.outposts",
        [
            "8/8/8/3N4/8/8/8/8 w - - 0 1",  # white knight on d5, no enemy pawns -> outpost
        ],
        [
            "8/8/2p5/3N4/8/8/8/8 w - - 0 1",  # black pawn c6 attacks d5, not an outpost
        ],
    ),
    (
        "backend.principles.space_advantage",
        [
            # White heavy pieces controlling many squares in enemy half; black minimal
            "8/8/8/4Q3/4Q3/8/8/4Q3 w - - 0 1",
        ],
        [
            "8/8/8/8/8/8/8/8 w - - 0 1",
        ],
    ),
    (
        "backend.principles.bishop_pair",
        [
            # White has two bishops, black only one bishop
            "8/8/8/8/8/8/8/3BB2k w - - 0 1",
        ],
        [
            # Symmetric bishops count -> not bishop pair advantage
            "8/8/8/8/8/8/8/3B1B1k w - - 0 1",
        ],
    ),
]


@pytest.mark.parametrize("module_path,true_fens,false_fens", CASES_SIMPLE)
def test_simple_boolean_cases(module_path, true_fens, false_fens):
    mod = import_module(module_path)
    detect = getattr(mod, 'detect')
    for fen in true_fens:
        board = chess.Board(fen)
        assert detect(board), f"Expected True for {module_path} on FEN: {fen}"
    for fen in false_fens:
        board = chess.Board(fen)
        assert not detect(board), f"Expected False for {module_path} on FEN: {fen}"


if __name__ == "__main__":
    raise SystemExit(main())
