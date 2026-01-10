"""
Programmatic tester for principle detectors using FEN strings.

This module exposes simple functions and a PrincipleTester class so you can
exercise individual principles in code (e.g., from a notebook or another script)
without using a command-line interface.

Example:

    from principles.principle_tester import PrincipleTester

    tester = PrincipleTester()

    # Evaluate a single principle on some FENs
    fens = [
        "8/8/8/8/8/2P5/2P5/8 w - - 0 1",  # white doubled on file c (True)
        "rnbqkb1r/pp3ppp/4p3/2pn4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",  # no doubled pawns here (False)
    ]
    results = tester.test_doubled_pawns(fens)  # {fen: True/False}

    # Analyze all principles for a given FEN
    tags = tester.analyze_all("rnbqkb1r/pp3ppp/4p3/2pn4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6")
    print(tags)  # e.g., ['DoubledPawns', ...]

"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys

# Ensure we can import the principles package when running directly
HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent.parent  # .../backend
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import chess  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("python-chess is required. pip install python-chess") from e

try:
    from principles.engine import PrinciplesEngine  # type: ignore
except Exception as e:
    raise RuntimeError(f"Failed to import PrinciplesEngine: {e}") from e

REGISTRY_PATH = HERE.parent / "registry.json"


class PrincipleTester:
    """Helper to evaluate principle detectors on FEN positions."""

    def __init__(self, registry_path: Optional[Path] = None):
        path = Path(registry_path) if registry_path else REGISTRY_PATH
        self.engine = PrinciplesEngine(path)
        # map id->callable
        self.detectors = dict(self.engine.detectors)

    def list_principles(self) -> List[str]:
        return self.engine.list_ids()

    def analyze_all(self, fen: str) -> List[str]:
        """Return all principle IDs detected in the given FEN."""
        board = chess.Board(fen)
        return self.engine.analyze(board)

    def evaluate_principle(self, principle_id: str, fen: str) -> bool:
        """Evaluate a single principle on a FEN. Returns True if detected."""
        if principle_id not in self.detectors:
            raise KeyError(f"Unknown principle: {principle_id}")
        board = chess.Board(fen)
        return bool(self.detectors[principle_id](board))

    def evaluate_principle_on_fens(self, principle_id: str, fens: Iterable[str]) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for fen in fens:
            try:
                out[fen] = self.evaluate_principle(principle_id, fen)
            except Exception:
                out[fen] = False
        return out

    # Convenience wrappers for common principles present in the registry
    def test_open_files(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("OpenFiles", fens)

    def test_outposts(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("Outposts", fens)

    def test_bishop_pair(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("BishopPair", fens)

    def test_passed_pawn(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("PassedPawn", fens)

    def test_rook_behind_passed(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("RookBehindPassedPawn", fens)

    def test_space_advantage(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("SpaceAdvantage", fens)

    def test_isolated_pawn(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("IsolatedPawn", fens)

    def test_backward_pawn(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("BackwardPawn", fens)

    def test_doubled_pawns(self, fens: Iterable[str]) -> Dict[str, bool]:
        return self.evaluate_principle_on_fens("DoubledPawns", fens)


# Optional light self-check when running this module directly
if __name__ == "__main__":
    t = PrincipleTester()
    demo_fens = [
        "8/8/8/8/8/2P5/2P5/8 w - - 0 1",  # white doubled on file c (True)
        "rnbqkb1r/pp3ppp/4p3/2pn4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",  # no doubled pawns here (False)
    ]
    print("Doubled:", t.test_doubled_pawns(demo_fens))
    for fen in demo_fens:
        print(f"All for FEN: {fen}\n  -> {t.analyze_all(fen)}")
