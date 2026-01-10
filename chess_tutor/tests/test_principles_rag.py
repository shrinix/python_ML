import json
import os
from pathlib import Path

import pytest

import sys
# Ensure the parent of 'chess_tutor' (i.e., the workspace python_ML dir) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chess_tutor.tutor_core import ChessTutor
from chess_tutor.backend.rag_service import ExtendedRAGService, REGISTRY_REL_PATH
from chess_tutor.backend.principles.engine import PrinciplesEngine
import chess


def load_registry_ids(limit: int | None = 3):
    if not os.path.isfile(REGISTRY_REL_PATH):
        pytest.skip("principles registry.json not found")
    data = json.loads(Path(REGISTRY_REL_PATH).read_text(encoding="utf-8"))
    items = data.get("principles") or []
    ids = [it.get("id") for it in items if it.get("id")]
    if limit:
        ids = ids[:limit]
    if not ids:
        pytest.skip("no principle IDs available in registry")
    return ids


@pytest.fixture(scope="module")
def svc():
    tutor = ChessTutor()
    service = ExtendedRAGService(tutor)
    if not getattr(service, "_principle_texts", []):
        pytest.skip("principle texts not loaded")
    if getattr(service, "_principle_emb", None) is None:
        pytest.skip("principle embeddings unavailable (model missing)")
    return service


def test_registry_has_principles():
    ids = load_registry_ids(limit=None)
    assert len(ids) > 0


@pytest.mark.parametrize("pid", load_registry_ids())
def test_retrieval_returns_principles_for_id(svc: ExtendedRAGService, pid: str):
    buckets = svc.retrieve(pid, top_k=3, include_games=False)
    princ = buckets.get("principles") or []
    assert princ, f"no principle results for {pid}"
    top_ids = {d.meta.get("id") for d in princ if isinstance(d.meta, dict)}
    assert pid in top_ids, f"{pid} not found among returned principles: {top_ids}"


@pytest.mark.parametrize("pid", load_registry_ids())
def test_boosting_places_id_first(svc: ExtendedRAGService, pid: str):
    buckets = svc.retrieve(pid, top_k=3, include_games=False)
    princ = buckets.get("principles") or []
    assert princ, f"no principle results for {pid}"
    # Expect the directly-mentioned principle to rank first after boost
    assert princ[0].meta.get("id") == pid


@pytest.mark.parametrize("pid", load_registry_ids())
def test_synthesize_includes_principle_context(svc: ExtendedRAGService, pid: str):
    buckets = svc.retrieve(pid, top_k=2, include_games=False)
    ans, sources = svc.synthesize_answer(f"Explain {pid}", buckets)
    assert isinstance(ans, str) and ans.strip()
    # Since we queried by ID, Principle context section should appear
    assert "Principle context:" in ans
    assert pid in ans or any((m.get("id") == pid) for s in sources for m in [s.get("meta", {})])


def test_doubled_pawns_negative_example():
    fen = "rnbqkb1r/pp3ppp/4p3/2pn4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6"
    eng = PrinciplesEngine(Path(REGISTRY_REL_PATH))
    board = chess.Board(fen)
    tags = eng.analyze(board)
    assert "DoubledPawns" not in tags


def test_doubled_pawns_positive_example():
    fen = "8/8/8/8/8/2P5/2P5/8 w - - 0 1"  # white doubled on file c
    eng = PrinciplesEngine(Path(REGISTRY_REL_PATH))
    board = chess.Board(fen)
    tags = eng.analyze(board)
    assert "DoubledPawns" in tags
