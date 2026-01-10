from fastapi import FastAPI, HTTPException, Query
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Set
import sys, pathlib, json
from pathlib import Path

# Ensure package and project roots are on sys.path so imports work when run from backend/
HERE = pathlib.Path(__file__).resolve()
PKG_DIR = HERE.parent.parent  # .../chess_tutor
ROOT_DIR = PKG_DIR.parent     # .../python_ML
BACKEND_DIR = HERE.parent     # .../chess_tutor/backend
for p in (str(PKG_DIR), str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Try multiple import styles (package and local)
try:
    from chess_tutor.tutor_core import ChessTutor  # type: ignore
    from chess_tutor.config import INCLUDE_RELEVANT_GAMES_DEFAULT  # type: ignore
except Exception:
    try:
        from tutor_core import ChessTutor  # type: ignore
        from config import INCLUDE_RELEVANT_GAMES_DEFAULT  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to import ChessTutor: {e}. sys.path={sys.path}")

# PDF ingest (span extraction) – import after sys.path setup
try:
    from chess_tutor.pdf_ingest import extract_text_from_pdf  # type: ignore
except Exception:
    try:
        from pdf_ingest import extract_text_from_pdf  # type: ignore
    except Exception:
        extract_text_from_pdf = None  # type: ignore

app = FastAPI(title="Adaptive Chess Tutor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _log_env_diag():  # lightweight diagnostics on startup
    try:
        import os as _os
        import inspect as _inspect
        print("[env] Python:", sys.executable)
        print("[env] CWD:", _os.getcwd())
        try:
            import fitz  # type: ignore
            print("[env] fitz OK:", getattr(fitz, "__file__", "<no file>"))
        except Exception as e:
            print("[env] fitz import error:", e)
        try:
            import chess_tutor as _ct_pkg  # type: ignore
            print("[env] chess_tutor package:", _inspect.getfile(_ct_pkg))
        except Exception:
            pass
    except Exception:
        pass
        import os as _os
        import inspect as _inspect
        print("[env] Python:", sys.executable)
        print("[env] CWD:", _os.getcwd())
        try:
            import fitz  # type: ignore
            print("[env] fitz OK:", getattr(fitz, "__file__", "<no file>"))
        except Exception as e:
            print("[env] fitz import error:", e)
        try:
            import chess_tutor as _ct_pkg  # type: ignore
            print("[env] chess_tutor package:", _inspect.getfile(_ct_pkg))
        except Exception:
            pass
    except Exception:
        pass

_log_env_diag()

# Strong guidance if server is not running under the project's venv
try:
    import os as _os
    venv_python = (_os.path.dirname(_os.path.dirname(__file__)) + "/.venv/bin/python")
    if not sys.executable.endswith(".venv/bin/python"):
        print("[env][WARN] Server is not using project venv. Current:", sys.executable)
        print("[env][WARN] Start with:")
        print("[env][WARN]   source ../.venv/bin/activate && python -m uvicorn app:app --reload --port 8000")
except Exception:
    pass

tutor = ChessTutor()

# ------------------------------
# Admin: force rebuild & dynamic reload of PDF/game ingestion
# ------------------------------
from importlib import reload as _reload

@app.post("/admin/rebuild", tags=["admin"], summary="Rebuild indices and reload ingestion modules")
def admin_rebuild(force: bool = True):
    global tutor
    details = {}
    # Attempt to reload pdf_ingest to pick up newly installed PyMuPDF.
    try:
        import chess_tutor.pdf_ingest as _pdf_mod  # type: ignore
        _reload(_pdf_mod)
        details["pdf_ingest_reloaded"] = True
        try:
            import fitz  # type: ignore
            details["fitz_available"] = True
            details["fitz_file"] = getattr(fitz, "__file__", "<no file>")
        except Exception as e:
            details["fitz_available"] = False
            details["fitz_error"] = f"{e.__class__.__name__}: {e}"
    except Exception as e:
        details["pdf_ingest_reloaded"] = False
        details["pdf_ingest_error"] = f"{e.__class__.__name__}: {e}"
    # Re-instantiate tutor (triggers index build/load)
    try:
        tutor = ChessTutor()
        details["reinstantiated"] = True
        details["doc_count"] = len(getattr(tutor, "docs", []))
        details["game_segments"] = len(getattr(tutor, "game_docs", []))
    except Exception as e:
        details["reinstantiated"] = False
        details["reinstantiate_error"] = f"{e.__class__.__name__}: {e}"
    return {"status": "ok", "details": details}

@app.get("/admin/diag", tags=["admin"], summary="Runtime environment diagnostics")
def admin_diag():
    import sys as _sys, os as _os
    diag = {
        "python": _sys.executable,
        "cwd": _os.getcwd(),
        "sys_path": list(_sys.path),
    }
    for mod in ("pymupdf", "fitz", "sentence_transformers", "faiss"):
        try:
            m = __import__(mod)
            diag[mod] = {"ok": True, "file": getattr(m, "__file__", "<no file>")}
        except Exception as e:
            diag[mod] = {"ok": False, "error": f"{e.__class__.__name__}: {e}"}
    return diag

# Optional external variations map (index_store/variations.json)
_VARIATIONS_MAP: Dict[str, list] = {}
try:
    VARIATIONS_PATH = PKG_DIR / "index_store" / "variations.json"
    if VARIATIONS_PATH.exists():
        with open(VARIATIONS_PATH, "r", encoding="utf-8") as _vf:
            data = json.load(_vf)
            if isinstance(data, dict):
                _VARIATIONS_MAP = data
except Exception:
    _VARIATIONS_MAP = {}

# ------------------------------
# Course mode: principle tagging, exercises, progress (MVP, in-memory)
# ------------------------------
try:
    import chess  # type: ignore
except Exception:  # pragma: no cover - if python-chess missing, course endpoints will be disabled
    chess = None  # type: ignore
try:
    from backend.tactics import analyze_tactics, analyze_engine_lines  # type: ignore
except Exception:
    try:
        from tactics import analyze_tactics, analyze_engine_lines  # type: ignore
    except Exception:
        analyze_tactics = None  # type: ignore
        analyze_engine_lines = None  # type: ignore

# New: pluggable principles engine with robust import
PRINCIPLES_DIR = HERE.parent / "principles"
REGISTRY_PATH = PRINCIPLES_DIR / "registry.json"

try:
    from principles.engine import PrinciplesEngine  # type: ignore
except Exception:
    PrinciplesEngine = None  # type: ignore

PRINCIPLES_ENGINE = PrinciplesEngine(REGISTRY_PATH) if (PrinciplesEngine and REGISTRY_PATH.exists()) else None
# Force registration of AttackedPieces visualizer if missing
import sys
if PRINCIPLES_ENGINE is not None:
    try:
        from principles import attacked_pieces
        PRINCIPLES_ENGINE.visualizers['AttackedPieces'] = attacked_pieces.visualize
        print("[API DEBUG] AttackedPieces visualizer forcibly registered.", file=sys.stdout)
    except Exception as e:
        print(f"[API DEBUG] Error forcibly registering AttackedPieces visualizer: {e}", file=sys.stdout)

def _resolve_principle_id(pid: str) -> str:
    if PRINCIPLES_ENGINE is None:
        return pid
    try:
        return PRINCIPLES_ENGINE.resolve_id(pid)
    except Exception:
        return pid

# ------------------------------
# SQLite persistence (course.db)
# ------------------------------
import sqlite3, os
from datetime import datetime

DB_DIR = PKG_DIR / "index_store"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "course.db"

_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_conn.row_factory = sqlite3.Row


def _db_exec(sql: str, params: tuple = ()):
    cur = _conn.cursor()
    cur.execute(sql, params)
    _conn.commit()
    return cur


def _db_query(sql: str, params: tuple = ()) -> list[dict]:
    cur = _conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def _init_db():
    _db_exec(
        """
        CREATE TABLE IF NOT EXISTS principles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT
        )
        """
    )
    _db_exec(
        """
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            principle_id TEXT NOT NULL,
            game_id TEXT NOT NULL,
            ply INTEGER NOT NULL,
            fen TEXT NOT NULL,
            san TEXT NOT NULL
        )
        """
    )
    _db_exec(
        """
        CREATE TABLE IF NOT EXISTS exercises (
            id TEXT PRIMARY KEY,
            principle_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            fen TEXT NOT NULL,
            solution TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    _db_exec(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exercise_id TEXT NOT NULL,
            answer TEXT,
            correct INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


def _dedup_examples():
    # remove duplicates, keep the lowest id
    _db_exec(
        "DELETE FROM examples WHERE rowid NOT IN (SELECT MIN(rowid) FROM examples GROUP BY principle_id, game_id, ply)"
    )
    # enforce uniqueness going forward
    _db_exec(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_examples ON examples(principle_id, game_id, ply)"
    )


_init_db()
# best-effort dedup/enforce unique
try:
    _dedup_examples()
except Exception:
    pass

# Seed principles if empty (legacy baseline)
if not _db_query("SELECT 1 FROM principles LIMIT 1"):
    for pid in [
        "OpenFiles",
        "Outposts",
        "BishopPair",
        "PassedPawn",
        "RookBehindPassedPawn",
        "SpaceAdvantage",
        "IsolatedPawn",
    ]:
        _db_exec("INSERT OR IGNORE INTO principles(id, name) VALUES (?, ?)", (pid, pid))

# ------------------------------
# Course logic: principle tagging, exercises, progress
# ------------------------------
PRINCIPLES = PRINCIPLES_ENGINE.list_ids() if PRINCIPLES_ENGINE else []

# Sync registry principles into DB so Admin list shows new items
def _sync_principles_from_registry():
    try:
        if PRINCIPLES_ENGINE is None:
            return
        for spec in PRINCIPLES_ENGINE.list_specs():
            _db_exec(
                "INSERT OR IGNORE INTO principles(id, name, description) VALUES (?,?,?)",
                (spec.id, spec.name or spec.id, spec.description),
            )
    except Exception:
        pass

# Also refresh in-memory keys so new principles are indexed during rebuilds
def _refresh_principles_in_memory():
    global PRINCIPLES, PRINCIPLE_INDEX, PROGRESS
    if PRINCIPLES_ENGINE is None:
        PRINCIPLES = []
        PRINCIPLE_INDEX = {}
        PROGRESS = {}
        return
    PRINCIPLES = PRINCIPLES_ENGINE.list_ids()
    # Reinitialize index and progress maps for new principle IDs
    PRINCIPLE_INDEX = {k: [] for k in PRINCIPLES}
    PROGRESS = {k: {"correct": 0, "total": 0} for k in PRINCIPLES}

# Run at startup
try:
    _sync_principles_from_registry()
    _refresh_principles_in_memory()
except Exception:
    pass

# In-memory stores
PRINCIPLE_INDEX: Dict[str, List[dict]] = {k: [] for k in PRINCIPLES}
EXERCISES: Dict[str, Dict] = {}
PROGRESS: Dict[str, Dict] = {k: {"correct": 0, "total": 0} for k in PRINCIPLES}


def _board_from_moves(moves: List[str], limit: int) -> Any:
    try:
        return tutor._apply_moves(moves, limit=limit)
    except Exception:
        return None


def analyze_principles(board: Any) -> List[str]:
    if PRINCIPLES_ENGINE is None:
        return []
    return PRINCIPLES_ENGINE.analyze(board)


def build_principle_index(max_games: Optional[int] = None):
    if chess is None:
        return
    # Clear
    for k in PRINCIPLE_INDEX:
        PRINCIPLE_INDEX[k].clear()
    games = tutor.list_games(limit=max_games or 1000000)
    for g in games:
        gid = g.get("id")
        moves = g.get("moves", [])
        if not gid or not moves:
            continue
        for ply, san in enumerate(moves, start=1):
            b = _board_from_moves(moves, limit=ply-1)
            if b is None:
                continue
            try:
                fen = b.fen()
            except Exception:
                fen = None
            if not fen:
                continue
            tags = analyze_principles(b)
            if not tags:
                continue
            rec = {"game_id": gid, "ply": ply, "fen": fen, "san": san}
            for t in tags:
                if t in PRINCIPLE_INDEX:
                    PRINCIPLE_INDEX[t].append(rec)
    # Optionally cap per-principle examples to a manageable size
    for t, lst in PRINCIPLE_INDEX.items():
        if len(lst) > 1000:
            PRINCIPLE_INDEX[t] = lst[:1000]


# Build index on startup (best-effort)
try:
    build_principle_index()
except Exception:
    pass


class ExplainRequest(BaseModel):
    query: str
    include_relevant_games: Optional[bool] = None


class ExplainResponse(BaseModel):
    answer: str


class GameSummary(BaseModel):
    id: str
    length: int
    final_fen: Optional[str]
    white: Optional[str] = None
    black: Optional[str] = None
    event: Optional[str] = None
    site: Optional[str] = None
    date: Optional[str] = None
    result: Optional[str] = None


class PlyRequest(BaseModel):
    game_id: str
    ply: int
    fen: Optional[str] = None


class VariationItem(BaseModel):
    label: Optional[str] = None
    line: List[str]
    first_san: Optional[str] = None
    first_from: Optional[str] = None
    first_to: Optional[str] = None
    fens: List[str] = []
    score_cp: Optional[int] = None
    mate: Optional[int] = None

class TacticsRequest(BaseModel):
    game_id: Optional[str] = None
    ply: Optional[int] = None
    fen: Optional[str] = None

class TacticItem(BaseModel):
    kind: str
    move: Optional[str] = None
    from_sq: Optional[str] = None
    to_sq: Optional[str] = None
    note: Optional[str] = None

class TacticsResponse(BaseModel):
    tactics: List[TacticItem]
    overlays: Optional[dict] = None

class PrincipleTagDetail(BaseModel):
    id: str
    side: Optional[str] = None  # 'W' | 'B' | None
    squares: List[str] = []     # impacted squares (may include multiple)
    captured: List[str] = []    # for MaterialAdvantage: list of captured opposing pieces (symbols)


class PlyResponse(BaseModel):
    board: str
    san: List[str]
    info: str
    fen: Optional[str]
    principles: List[str] = []
    principle_details: List[PrincipleTagDetail] = []
    overlays_before: Optional[dict] = None
    overlays_after: Optional[dict] = None
    variations: Optional[List[VariationItem]] = None


class MovesResponse(BaseModel):
    id: str
    moves: List[str]


class GameExplainResponse(BaseModel):
    id: str
    text: str

# RAG chat models (extended)
class ChatTurn(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str

class ChatSource(BaseModel):
    snippet: str
    meta: dict

class ChatRequest(BaseModel):
    messages: List[ChatTurn]
    top_k: Optional[int] = 3
    include_games: Optional[bool] = True  # include game narrative sources
    include_principles: Optional[bool] = True  # toggle principle descriptions retrieval
    game_id: Optional[str] = None  # optional current game context
    ply: Optional[int] = None      # optional current ply within game
    principle_id: Optional[str] = None  # optional principle emphasis

class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource] = []  # ordered list of source snippets


# Course-mode models
class Principle(BaseModel):
    id: str
    name: str
    examples: int


class Example(BaseModel):
    game_id: str
    ply: int
    fen: str
    san: str


class ExerciseGenerateRequest(BaseModel):
    principle_id: str
    difficulty: Optional[str] = None  # beginner|intermediate|advanced
    kind: Optional[str] = None  # identify|bestmove


class ExerciseResponse(BaseModel):
    exercise_id: str
    principle_id: str
    kind: str
    fen: str
    prompt: str
    options: Optional[List[str]] = None
    solution: Optional[str] = None  # returned for now (MVP); hide in production


class ExerciseSubmitRequest(BaseModel):
    exercise_id: str
    answer: str


class ExerciseSubmitResponse(BaseModel):
    correct: bool
    feedback: str
    principle_progress: Optional[dict] = None


class ProgressResponse(BaseModel):
    progress: dict


class GamePrinciplesTag(BaseModel):
    ply: int
    principles: List[str]


class GamePrinciplesResponse(BaseModel):
    id: str
    tags: List[GamePrinciplesTag]


@app.get("/principles", response_model=List[Principle])
def get_principles():
    if chess is None:
        return []
    rows = _db_query(
        "SELECT p.id, COALESCE(p.name, p.id) as name, COUNT(e.id) as examples FROM principles p LEFT JOIN examples e ON e.principle_id=p.id GROUP BY p.id ORDER BY p.id"
    )
    return [Principle(id=r["id"], name=r["name"], examples=r["examples"]) for r in rows]


@app.get("/principles/{principle_id}/examples", response_model=List[Example])
def get_examples(principle_id: str, limit: int = 50):
    if chess is None:
        return []
    canon = _resolve_principle_id(principle_id)
    if canon != principle_id:
        principle_id = canon
    if not _db_query("SELECT 1 FROM principles WHERE id=?", (principle_id,)):
        raise HTTPException(status_code=404, detail="Unknown principle")
    rows = _db_query(
        "SELECT game_id, ply, fen, san FROM examples WHERE principle_id=? ORDER BY id LIMIT ?",
        (principle_id, int(max(1, min(limit, 200)))),
    )
    return [Example(game_id=r["game_id"], ply=r["ply"], fen=r["fen"], san=r["san"]) for r in rows]


# Authoring models/endpoints
class UpsertPrincipleRequest(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None


class AddExampleRequest(BaseModel):
    principle_id: str
    game_id: str
    ply: int
    fen: str
    san: str


class RebuildRequest(BaseModel):
    clear: Optional[bool] = False
    max_games: Optional[int] = None


@app.post("/author/principles/upsert")
def upsert_principle(req: UpsertPrincipleRequest):
    if not req.id:
        raise HTTPException(status_code=400, detail="Missing id")
    _db_exec(
        "INSERT INTO principles(id, name, description) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, description=excluded.description",
        (req.id, req.name or req.id, req.description),
    )
    return {"status": "ok"}


@app.post("/author/examples/add")
def add_example(req: AddExampleRequest):
    if not _db_query("SELECT 1 FROM principles WHERE id=?", (req.principle_id,)):
        raise HTTPException(status_code=404, detail="Unknown principle")
    _db_exec(
        "INSERT OR IGNORE INTO examples(principle_id, game_id, ply, fen, san) VALUES (?,?,?,?,?)",
        (req.principle_id, req.game_id, int(req.ply), req.fen, req.san),
    )
    return {"status": "ok"}


@app.delete("/author/examples/{example_id}")
def delete_example(example_id: int):
    _db_exec("DELETE FROM examples WHERE id=?", (int(example_id),))
    return {"status": "ok"}


@app.post("/author/rebuild_from_games")
def rebuild_from_games(req: RebuildRequest):
    try:
        if PRINCIPLES_ENGINE is not None:
            PRINCIPLES_ENGINE.reload()
        _sync_principles_from_registry()
        _refresh_principles_in_memory()
        build_principle_index(req.max_games)
        _sync_examples_from_index(clear=bool(req.clear))
        _dedup_examples()
        return {"status": "ok", "reindexed": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/author/sync_principles")
def author_sync_principles():
    try:
        if PRINCIPLES_ENGINE is not None:
            PRINCIPLES_ENGINE.reload()
        _sync_principles_from_registry()
        _refresh_principles_in_memory()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/author/reload_games")
def author_reload_games():
    try:
        tutor.reload_games()
        # After reloading games, refresh the in-memory principle index map (empty) so a Rebuild can repopulate
        for k in PRINCIPLE_INDEX:
            PRINCIPLE_INDEX[k].clear()
        return {"status": "ok", "games": len(tutor.games or [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/games_status")
def debug_games_status():
    try:
        from chess_tutor.config import GAMES_PATH as CFG_GAMES_PATH  # type: ignore
    except Exception:
        CFG_GAMES_PATH = str(DB_DIR / "games.json")
    path = Path(CFG_GAMES_PATH)
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    count = len(tutor.games or [])
    read_ok = None
    read_count = None
    read_error = None
    if exists:
        try:
            import json as _json
            data = _json.loads(path.read_text(encoding="utf-8"))
            read_ok = True
            read_count = len(data or [])
        except Exception as e:
            read_ok = False
            read_error = str(e)
    return {
        "config_games_path": str(path),
        "exists": exists,
        "size": size,
        "loaded_games_count": count,
        "file_read_ok": read_ok,
        "file_count": read_count,
        "file_read_error": read_error,
    }


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    ans = tutor.explain(req.query, include_relevant_games=req.include_relevant_games)
    return ExplainResponse(answer=ans)


try:
    from chess_tutor.config import (
        ENABLE_EXTENDED_RAG, RAG_TOP_K_DEFAULT, ENABLE_LLM, LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY_ENV
    )
except Exception:
    ENABLE_EXTENDED_RAG = True
    RAG_TOP_K_DEFAULT = 3
    ENABLE_LLM = False
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o-mini"
    OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

_RAG_SERVICE = None  # lazy init (recreated on mode change)
_LLM_RUNTIME_ENABLED = bool(ENABLE_LLM)

def _reinit_rag_service():
    global _RAG_SERVICE
    _RAG_SERVICE = None  # will be rebuilt on next /chat invocation

@app.get("/admin/llm_status")
def admin_llm_status():
    api_key_present = bool(os.environ.get(OPENAI_API_KEY_ENV))
    return {
        "enabled": _LLM_RUNTIME_ENABLED,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "api_key_present": api_key_present,
    }

class LLMToggleRequest(BaseModel):
    enable: bool

@app.post("/admin/llm_toggle")
def admin_llm_toggle(req: LLMToggleRequest):
    global _LLM_RUNTIME_ENABLED
    _LLM_RUNTIME_ENABLED = bool(req.enable)
    _reinit_rag_service()
    return {"status": "ok", "enabled": _LLM_RUNTIME_ENABLED}

# ------------------------------
# Tactics / Engine runtime configuration
# ------------------------------
class TacticsConfig(BaseModel):
    min_cp: Optional[int] = None           # minimum advantage in centipawns to qualify (e.g., 300 = +3.0)
    max_len: Optional[int] = None          # maximum number of moves in a tactic line (SAN count)
    max_lines: Optional[int] = None        # MultiPV lines to request from engine
    depth: Optional[int] = None            # engine search depth
    movetime_ms: Optional[int] = None      # alternative to depth (milliseconds)
    candidate_max_len: Optional[int] = None  # maximum length for candidate move lines

# Defaults
TACTICS_MIN_CP: int = 300
TACTICS_MAX_LEN: int = 5
ENGINE_MAX_LINES: int = 3
ENGINE_DEPTH: int = 12
ENGINE_MOVETIME_MS: Optional[int] = None
CANDIDATE_MAX_LEN: int = 5

@app.get("/admin/tactics_config")
def admin_tactics_config_get():
    return {
        "min_cp": TACTICS_MIN_CP,
        "max_len": TACTICS_MAX_LEN,
        "max_lines": ENGINE_MAX_LINES,
        "depth": ENGINE_DEPTH,
        "movetime_ms": ENGINE_MOVETIME_MS,
        "candidate_max_len": CANDIDATE_MAX_LEN,
    }

@app.post("/admin/tactics_config")
def admin_tactics_config_set(req: TacticsConfig):
    global TACTICS_MIN_CP, TACTICS_MAX_LEN, ENGINE_MAX_LINES, ENGINE_DEPTH, ENGINE_MOVETIME_MS
    if isinstance(req.min_cp, int) and req.min_cp >= 0:
        TACTICS_MIN_CP = req.min_cp
    if isinstance(req.max_len, int) and req.max_len >= 1:
        TACTICS_MAX_LEN = req.max_len
    if isinstance(req.max_lines, int) and 1 <= req.max_lines <= 10:
        ENGINE_MAX_LINES = req.max_lines
    if isinstance(req.depth, int) and req.depth >= 1:
        ENGINE_DEPTH = req.depth
        ENGINE_MOVETIME_MS = None  # depth overrides movetime
    if isinstance(req.movetime_ms, int) and req.movetime_ms >= 100:
        ENGINE_MOVETIME_MS = req.movetime_ms
    if isinstance(req.candidate_max_len, int) and req.candidate_max_len >= 1:
        CANDIDATE_MAX_LEN = req.candidate_max_len
    return {"status": "ok", "config": admin_tactics_config_get()}

# ------------------------------
# Candidate Moves endpoint
# ------------------------------
class CandidateRequest(BaseModel):
    game_id: Optional[str] = None
    ply: Optional[int] = None
    fen: Optional[str] = None

class CandidateResponse(BaseModel):
    candidates: List[VariationItem] = []

@app.post("/candidates", response_model=CandidateResponse)
def candidates(req: CandidateRequest):
    if chess is None or analyze_engine_lines is None:
        return CandidateResponse(candidates=[])
    # Resolve board state from fen or game_id/ply
    board = None
    if req.fen:
        try:
            board = chess.Board(req.fen)
        except Exception:
            board = None
    elif req.game_id is not None and isinstance(req.ply, int):
        g = tutor.get_game_by_id(req.game_id)
        if g and (g.get('moves') is not None):
            board = _board_from_moves(g.get('moves') or [], limit=req.ply)
    if board is None:
        raise HTTPException(status_code=400, detail="Unable to resolve board state for candidate moves")
    eng = analyze_engine_lines(
        board,
        max_lines=ENGINE_MAX_LINES,
        depth=ENGINE_DEPTH,
        movetime_ms=ENGINE_MOVETIME_MS,
    )
    all_lines = eng.get("lines") or []
    # Sort by score_cp descending (mates treated as highest)
    def _score_key(ln):
        mt = ln.get("mate")
        sc = ln.get("score_cp")
        if mt is not None and mt > 0:
            return 10_000_000  # prioritize mates
        return sc if isinstance(sc, int) else -10_000_000
    all_lines.sort(key=_score_key, reverse=True)
    items: List[VariationItem] = []
    for ln in all_lines[:ENGINE_MAX_LINES]:
        line = [str(s) for s in (ln.get("line") or [])]
        fens = [str(f) for f in (ln.get("fens") or [])]
        # Truncate to candidate max length
        if len(line) > int(CANDIDATE_MAX_LEN):
            line = line[:int(CANDIDATE_MAX_LEN)]
        if len(fens) > int(CANDIDATE_MAX_LEN):
            fens = fens[:int(CANDIDATE_MAX_LEN)]
        sc = ln.get("score_cp")
        mt = ln.get("mate")
        lab = None
        if mt is not None and mt > 0:
            lab = f"Mate in {abs(mt)}"
        elif isinstance(sc, int):
            lab = f"SF eval: {sc/100:.2f}"
        items.append(VariationItem(
            label=lab,
            line=line,
            first_san=ln.get("first_san"),
            first_from=ln.get("first_from"),
            first_to=ln.get("first_to"),
            fens=fens,
            score_cp=sc if isinstance(sc,int) else None,
            mate=mt if (mt is not None) else None,
        ))
    return CandidateResponse(candidates=items)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Enhanced RAG chat endpoint.

    If ENABLE_EXTENDED_RAG, uses backend.rag_service for augmented retrieval (principles, docs, games).
    Otherwise falls back to legacy lightweight retrieval.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")
    # Extract latest user question
    user_q = None
    for m in reversed(req.messages):
        if (m.role or '').lower() == 'user' and (m.content or '').strip():
            user_q = m.content.strip()
            break
    if not user_q:
        raise HTTPException(status_code=400, detail="no user message provided")

    top_k = max(1, int(req.top_k or RAG_TOP_K_DEFAULT))
    # Detect if user wants position explanation explicitly (keywords) or provided game context
    wants_position = False
    q_lower = user_q.lower()
    for kw in ["explain the position", "explain position", "explain this position", "position explain", "what is happening", "evaluate this position", "evaluation of this position"]:
        if kw in q_lower:
            wants_position = True
            break
    has_game_ctx = bool(req.game_id and isinstance(req.ply, int) and req.ply >= 0)

    if ENABLE_EXTENDED_RAG:
        global _RAG_SERVICE
        if _RAG_SERVICE is None:
            try:
                from backend.rag_service import get_extended_rag_service  # type: ignore
            except Exception:
                from rag_service import get_extended_rag_service  # type: ignore
            _RAG_SERVICE = get_extended_rag_service(tutor)
        # Enforce runtime LLM toggle by creating/disabling the LLM client on the service
        try:
            svc_llm = getattr(_RAG_SERVICE, '_llm', None)
            if _LLM_RUNTIME_ENABLED:
                if svc_llm is None:
                    try:
                        from backend.llm_adapter import LLMClient  # type: ignore
                    except Exception:
                        from llm_adapter import LLMClient  # type: ignore
                    setattr(_RAG_SERVICE, '_llm', LLMClient())
                    svc_llm = getattr(_RAG_SERVICE, '_llm', None)
                if svc_llm is not None:
                    setattr(svc_llm, 'enabled', True)
            else:
                if svc_llm is not None:
                    setattr(svc_llm, 'enabled', False)
        except Exception:
            pass
        buckets = _RAG_SERVICE.retrieve(user_q, top_k=top_k, include_games=bool(req.include_games))
        # Add position context bucket if game context present or user asks to explain position
        if has_game_ctx or wants_position:
            try:
                g = tutor.get_game_by_id(req.game_id) if req.game_id else None
                if g and (g.get('moves') is not None):
                    moves = g.get('moves') or []
                    limit = req.ply if isinstance(req.ply, int) else len(moves)
                    if limit < 0:
                        limit = 0
                    b = _board_from_moves(moves, limit=limit)
                    fen = None
                    principles = []
                    if b is not None and chess is not None:
                        try:
                            fen = b.fen()
                        except Exception:
                            fen = None
                        try:
                            principles = analyze_principles(b)
                        except Exception:
                            principles = []
                    desc_lines = []
                    if fen:
                        desc_lines.append(f"FEN: {fen}")
                        try:
                            parts = fen.split()
                            if len(parts) >= 6:
                                stm = 'White' if parts[1] == 'w' else 'Black'
                                fullmove = parts[5]
                                desc_lines.append(f"Side to move: {stm}")
                                desc_lines.append(f"Fullmove: {fullmove}")
                        except Exception:
                            pass
                    if principles:
                        # Build enriched principle display with side + impacted squares
                        enriched: List[str] = []
                        principle_sentences: List[str] = []  # explicit natural language sentences for LLM grounding
                        try:
                            info_map = getattr(PRINCIPLES_ENGINE, 'detectors_info', {}) if PRINCIPLES_ENGINE else {}
                        except Exception:
                            info_map = {}
                        try:
                            viz_map = getattr(PRINCIPLES_ENGINE, 'visualizers', {}) if PRINCIPLES_ENGINE else {}
                        except Exception:
                            viz_map = {}
                        for pid in principles:
                            side_short = None
                            squares: List[str] = []
                            captured: List[str] = []
                            fn = info_map.get(pid) if info_map else None
                            if callable(fn):
                                try:
                                    data = fn(b) or {}
                                    side = (data.get('impacted_side') or '').lower()
                                    if side == 'white':
                                        side_short = 'w'
                                    elif side == 'black':
                                        side_short = 'b'
                                    sqs = data.get('impact_squares') or []
                                    if isinstance(sqs, list):
                                        squares = [str(s) for s in sqs if s]
                                    cap = data.get('captured_pieces') or []
                                    if isinstance(cap, list):
                                        captured = [str(c) for c in cap if c]
                                except Exception:
                                    pass
                            if (not squares) and viz_map.get(pid):
                                # Fallback to visualizer highlights
                                try:
                                    indiv_viz = viz_map[pid](b)
                                    hl = (indiv_viz or {}).get('highlights') if isinstance(indiv_viz, dict) else None
                                    if isinstance(hl, list):
                                        squares = [str(h.get('square')) for h in hl if h.get('square')]
                                except Exception:
                                    pass
                            if len(squares) > 6:
                                squares = squares[:6]
                            # New formatting: PrincipleName Side (sq1,sq2,...) without side letters on squares
                            side_name = None
                            if side_short == 'w':
                                side_name = 'White'
                            elif side_short == 'b':
                                side_name = 'Black'
                            if squares:
                                sq_part = ",".join(squares[:4]) + (",…" if len(squares) > 4 else "")
                                cap_part = ""
                                if captured:
                                    # Summarize captured list compactly e.g. +Q,R,P,P
                                    cap_part = " +" + ",".join(captured[:6])
                                if side_name:
                                    enriched.append(f"{pid} {side_name} ({sq_part}){cap_part}")
                                    # Sentence: e.g. "Black has SpaceAdvantage with control over c3,c4,d3,d4." or material lead
                                    if pid == 'MaterialAdvantage':
                                        mat_phrase = f"{side_name} has a material advantage" + (f" (extra {', '.join(captured[:6])})" if captured else "")
                                        principle_sentences.append(mat_phrase + ".")
                                    elif pid == 'SpaceAdvantage':
                                        principle_sentences.append(f"{side_name} has Space Advantage on {sq_part.replace(',…','')}.")
                                    else:
                                        principle_sentences.append(f"{side_name} has {pid} involving squares {sq_part.replace(',…','')}.")
                                else:
                                    enriched.append(f"{pid} ({sq_part}){cap_part}")
                                    if pid == 'MaterialAdvantage':
                                        mat_phrase = "A material advantage is present" + (f" (extra {', '.join(captured[:6])})" if captured else "")
                                        principle_sentences.append(mat_phrase + ".")
                                    elif pid == 'SpaceAdvantage':
                                        principle_sentences.append(f"Space Advantage on {sq_part.replace(',…','')}.")
                                    else:
                                        principle_sentences.append(f"{pid} on squares {sq_part.replace(',…','')}.")
                            else:
                                cap_part = ""
                                if captured:
                                    cap_part = " +" + ",".join(captured[:6])
                                if side_name:
                                    enriched.append(f"{pid} {side_name}{cap_part}")
                                    if pid == 'MaterialAdvantage':
                                        mat_phrase = f"{side_name} has a material advantage" + (f" (extra {', '.join(captured[:6])})" if captured else "")
                                        principle_sentences.append(mat_phrase + ".")
                                    else:
                                        principle_sentences.append(f"{side_name} has {pid}.")
                                else:
                                    enriched.append(f"{pid}{cap_part}")
                                    if pid == 'MaterialAdvantage':
                                        principle_sentences.append("Material advantage present." + (f" Extra {', '.join(captured[:6])}." if captured else ""))
                                    else:
                                        principle_sentences.append(f"{pid} detected.")
                        desc_lines.append("Principles detected: " + ", ".join(enriched))
                        if principle_sentences:
                            # Prepend explicit sentences before the compact summary for clearer LLM grounding
                            desc_lines.append("Principle summary: " + " ".join(principle_sentences))
                    # Lightweight material/imbalance heuristic
                    material_eval = None
                    if b is not None and chess is not None:
                        try:
                            piece_values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}
                            white_score = 0
                            black_score = 0
                            for sq in chess.SQUARES:
                                p = b.piece_at(sq)
                                if not p:
                                    continue
                                val = piece_values.get(p.piece_type, 0)
                                if p.color == chess.WHITE:
                                    white_score += val
                                else:
                                    black_score += val
                            diff = white_score - black_score
                            if diff > 0:
                                material_eval = f"White is up {diff} pawn units of material"
                            elif diff < 0:
                                material_eval = f"Black is up {abs(diff)} pawn units of material"
                            else:
                                material_eval = "Material is balanced"
                        except Exception:
                            material_eval = None
                    if material_eval:
                        desc_lines.append(material_eval)
                    if not desc_lines:
                        desc_lines.append("(No board details available)")
                    # add optional source meta if present
                    meta_extra = {'type':'position','game_id': req.game_id or g.get('id'), 'ply': limit}
                    src = g.get('source');  
                    if src: meta_extra['source'] = src
                    if g.get('final_fen'): meta_extra['final_fen'] = g.get('final_fen')
                    # Try to infer a relevant page from existing doc hits
                    try:
                        for cat in ('docs','games'):
                            for d in (buckets.get(cat) or []):
                                m = d.meta or {}
                                if src and (m.get('source') == src) and ('page' in m):
                                    meta_extra['page'] = m.get('page')
                                    raise StopIteration
                    except StopIteration:
                        pass
                    pos_text = "Position context (game {gid}, ply {ply}):\n".format(gid=req.game_id or g.get('id'), ply=limit) + "\n".join(desc_lines)
                    # Include a compact ASCII board (ranks 8..1) to reduce misinterpretation
                    try:
                        if b is not None and chess is not None:
                            ascii_rows = []
                            for rank in range(7, -1, -1):
                                row = []
                                for file in range(8):
                                    sq = chess.square(file, rank)
                                    p = b.piece_at(sq)
                                    row.append(p.symbol() if p else '.')
                                ascii_rows.append(f"{rank+1} " + ' '.join(row))
                            files_label = "  a b c d e f g h"
                            pos_text += "\nBoard (white at bottom, ranks 8->1):\n" + "\n".join(ascii_rows) + "\n" + files_label
                            # Add pawn lists per color for clarity
                            try:
                                w_pawns = sorted([chess.square_name(sq) for sq in b.pieces(chess.PAWN, chess.WHITE)])
                                b_pawns = sorted([chess.square_name(sq) for sq in b.pieces(chess.PAWN, chess.BLACK)])
                                pos_text += "\nPawns — White: " + (", ".join(w_pawns) or "none") + "; Black: " + (", ".join(b_pawns) or "none")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    buckets['position'] = [type('Tmp', (), {'text': pos_text, 'meta': meta_extra, 'score': 1.0})()]
            except Exception:
                pass
        # If a principle_id is explicitly provided, inject a focus note and ensure principles bucket present
        if req.principle_id:
            try:
                desc = None
                if PRINCIPLES_ENGINE is not None:
                    for spec in PRINCIPLES_ENGINE.list_specs():
                        if str(spec.id).lower() == str(req.principle_id).lower():
                            desc = spec.description
                            break
                focus_text = f"Focus principle: {req.principle_id}\n" + (desc or "")
                focus_doc = type('Tmp', (), {'text': focus_text, 'meta': {'type':'principle','id': req.principle_id, 'field':'focus'}, 'score': 1.0})()
                if 'principles' not in buckets or buckets['principles'] is None:
                    buckets['principles'] = [focus_doc]
                else:
                    buckets['principles'] = [focus_doc] + list(buckets['principles'])
            except Exception:
                pass
        # Optionally drop principle category if user turned it off
        if not req.include_principles:
            buckets['principles'] = []
        answer, sources_meta = _RAG_SERVICE.synthesize_answer(user_q, buckets)
        sources = [ChatSource(snippet=s.get('snippet',''), meta=s.get('meta') or {}) for s in sources_meta][:top_k*3]
        return ChatResponse(answer=answer, sources=sources)

    # Legacy fallback (original behavior)
    docs = []
    game_docs = []
    try:
        docs = tutor.retrieve(user_q, top_k=top_k) or []
    except Exception:
        docs = []
    if req.include_games:
        try:
            game_docs = tutor.retrieve_game_segments(user_q, top_k=top_k) or []
        except Exception:
            game_docs = []
    parts: List[str] = []
    sources: List[ChatSource] = []
    for text, meta in docs:
        snippet = (text or '').strip()
        if not snippet:
            continue
        sources.append(ChatSource(snippet=snippet[:500], meta=meta or {}))
        sents = [s.strip() for s in snippet.split('\n') if s.strip()]
        parts.extend(sents[:2])
    if game_docs:
        parts.append("\nRelevant game excerpts:")
        for text, meta in game_docs:
            snippet = (text or '').strip()
            if not snippet:
                continue
            sources.append(ChatSource(snippet=snippet[:500], meta=meta or {}))
            gid = (meta or {}).get('id') if isinstance(meta, dict) else None
            length = (meta or {}).get('length') if isinstance(meta, dict) else None
            hdr = f"- {gid} (len={length})" if gid else "- Game excerpt"
            first = snippet.split('\n', 1)[0]
            parts.append(f"{hdr}: {first}")
    if not parts:
        parts = ["I couldn't find relevant material right now. Try rephrasing or asking about a specific game or concept."]
    answer = (f"Q: {user_q}\n\n" + "\n".join(parts)).strip()
    return ChatResponse(answer=answer, sources=sources[:top_k])


@app.post("/ply", response_model=PlyResponse)
def ply(req: PlyRequest):
    import sys
    print(f"[API DEBUG] /ply: Request: game_id={req.game_id}, ply={req.ply}, fen={getattr(req, 'fen', None)}", file=sys.stdout)
    print(f"[API DEBUG] /ply: Endpoint called. Request: {req}", file=sys.stdout)
    board_txt, san, info = tutor.board_after_ply(req.game_id, req.ply)
    if board_txt is None:
        raise HTTPException(status_code=400, detail=info or "Invalid request")
    g = tutor.get_game_by_id(req.game_id)
    fen_before = None
    fen_after = None
    principles: List[str] = []
    principle_details: List[PrincipleTagDetail] = []
    overlays_before = {"arrows": [], "highlights": []}
    overlays_after = {"arrows": [], "highlights": []}
    variations: List[VariationItem] = []
    # Merge external variations if any
    if g is not None:
        try:
            ext = _VARIATIONS_MAP.get(req.game_id)
            if ext:
                g = dict(g)
                g['variations'] = list(g.get('variations') or []) + list(ext)
        except Exception:
            pass
    if g and (g.get('moves') is not None):
        import chess
        moves = g.get('moves') or []
        # fen_before: board before the move (for reference)
        # fen_after: board after the move (for rendering and overlays/principles)
        if req.fen:
            b_after = chess.Board(req.fen)
            print(f"[API DEBUG] Using FEN for overlays/principles: {req.fen}", file=sys.stdout)
        else:
            b_after = _board_from_moves(moves, limit=req.ply)
        b_before = _board_from_moves(moves, limit=max(0, req.ply-1))
        # overlays_after: for board after the move (side to move)
        # overlays_before: for board before the move (previous mover)
        b_for_overlay = b_after
        b_for_overlay_before = b_before
        print(f"[API DEBUG] Using moves for board: {g.get('moves')}", file=sys.stdout)
        print(f"[API DEBUG] SAN for ply {req.ply}: {(g.get('moves') or [])[:req.ply]}", file=sys.stdout)
        print(f"[API DEBUG] FEN for ply {req.ply} (before move): {b_before.fen() if b_before else None}", file=sys.stdout)
        print(f"[API DEBUG] FEN for ply {req.ply} (after move): {b_after.fen() if b_after else None}", file=sys.stdout)
        try:
            fen_before = b_before.fen() if b_before is not None else None
        except Exception:
            fen_before = None
        try:
            fen_after = b_after.fen() if b_after is not None else None
        except Exception:
            fen_after = None
        try:
            fen = b_after.fen() if b_after is not None else None
        except Exception:
            fen = None
        try:
            if b_for_overlay is not None and chess is not None:
                print(f"[API DEBUG] Calculating principles/overlays for FEN: {b_for_overlay.fen()}", file=sys.stdout)
                principles = analyze_principles(b_for_overlay)
                print(f"[API DEBUG] Principles detected: {principles}", file=sys.stdout)
                if PRINCIPLES_ENGINE is not None and principles:
                    try:
                        info_map = getattr(PRINCIPLES_ENGINE, 'detectors_info', {}) or {}
                    except Exception:
                        info_map = {}
                    details: List[PrincipleTagDetail] = []
                    try:
                        viz_map = getattr(PRINCIPLES_ENGINE, 'visualizers', {}) or {}
                    except Exception:
                        viz_map = {}
                    for pid in principles:
                        side_short = None
                        squares: List[str] = []
                        captured: List[str] = []
                        fn = info_map.get(pid) if info_map else None
                        if callable(fn):
                            try:
                                data = fn(b_for_overlay) or {}
                                side = (data.get('impacted_side') or '').lower()
                                if side == 'white':
                                    side_short = 'W'
                                elif side == 'black':
                                    side_short = 'B'
                                sqs = data.get('impact_squares') or []
                                if isinstance(sqs, list):
                                    squares = [str(s) for s in sqs if s]
                                cap = data.get('captured_pieces') or []
                                if isinstance(cap, list):
                                    captured = [str(c) for c in cap if c]
                            except Exception:
                                pass
                        if (not squares) and viz_map.get(pid):
                            try:
                                indiv_viz = viz_map[pid](b_for_overlay)
                                hl = (indiv_viz or {}).get('highlights') if isinstance(indiv_viz, dict) else None
                                if isinstance(hl, list):
                                    squares = [str(h.get('square')) for h in hl if h.get('square')]
                            except Exception:
                                pass
                        if len(squares) > 6:
                            squares = squares[:6]
                        details.append(PrincipleTagDetail(id=pid, side=side_short, squares=squares, captured=captured))
                    principle_details = details
                    overlays_after = PRINCIPLES_ENGINE.visualize(b_for_overlay, principles)
                    # overlays_before: always use previous board state for attacks/highlights
                    overlays_before = {"arrows": [], "highlights": []}
                    attacked_viz = PRINCIPLES_ENGINE.visualizers.get('AttackedPieces')
                    print(f"[API DEBUG] PRINCIPLES_ENGINE.visualizers.get('AttackedPieces'): {attacked_viz}", file=sys.stdout)
                    if attacked_viz:
                        print(f"[API DEBUG] Calling AttackedPieces.visualizer with FEN: {b_for_overlay_before.fen()}", file=sys.stdout)
                        attacked_overlay = attacked_viz(b_for_overlay_before)
                        print(f"[API DEBUG] AttackedPieces.visualizer returned: {attacked_overlay}", file=sys.stdout)
                        if attacked_overlay:
                            overlays_before['arrows'].extend(attacked_overlay.get('arrows', []))
                            overlays_before['highlights'].extend(attacked_overlay.get('highlights', []))
                    print(f"[API DEBUG] Overlays_after calculated: {overlays_after}", file=sys.stdout)
                    print(f"[API DEBUG] Overlays_before calculated: {overlays_before}", file=sys.stdout)
                # Optional: augment with engine-based multi-move tactics (Stockfish)
                try:
                    eng_ok = True if analyze_engine_lines is not None else False
                    if eng_ok:
                        eng = analyze_engine_lines(
                            b_for_overlay,
                            max_lines=ENGINE_MAX_LINES,
                            depth=ENGINE_DEPTH,
                            movetime_ms=ENGINE_MOVETIME_MS,
                        )
                        all_lines = eng.get("lines") or []
                        # Filter: multi-move (>=2), length <= TACTICS_MAX_LEN, and eval threshold or mate
                        eng_lines = []
                        for ln in all_lines:
                            line = ln.get("line") or []
                            if not isinstance(line, list):
                                continue
                            if len(line) < 2:
                                continue
                            if len(line) > int(TACTICS_MAX_LEN):
                                continue
                            sc = ln.get("score_cp")
                            mt = ln.get("mate")
                            if (isinstance(sc, int) and sc >= int(TACTICS_MIN_CP)) or (mt is not None and mt > 0):
                                eng_lines.append(ln)
                        if eng_lines:
                            # Add as variations entries with labels
                            for ln in eng_lines:
                                lab = None
                                sc = ln.get("score_cp")
                                mate = ln.get("mate")
                                if mate is not None:
                                    lab = f"Mate in {abs(mate)}" if mate else "Mate found"
                                elif isinstance(sc, int):
                                    lab = f"SF eval: {sc/100:.2f}"
                                # Figure out first move SAN/from/to
                                first_san = ln.get("first_san")
                                first_from = ln.get("first_from")
                                first_to = ln.get("first_to")
                                variations.append(VariationItem(
                                    label=lab,
                                    line=[str(s) for s in (ln.get("line") or [])],
                                    first_san=first_san,
                                    first_from=first_from,
                                    first_to=first_to,
                                    fens=[str(f) for f in (ln.get("fens") or [])],
                                    score_cp=sc if isinstance(sc,int) else None,
                                    mate=mate if (mate is not None) else None,
                                ))
                            # Gate tactics in UI: inject a Tactics detail with first-move target squares
                            try:
                                tactic_sqs = []
                                for ln in eng_lines:
                                    to_sq = ln.get("first_to")
                                    if to_sq:
                                        tactic_sqs.append(str(to_sq))
                                if tactic_sqs:
                                    # add or extend existing Tactics detail
                                    found_idx = None
                                    for i, d in enumerate(principle_details or []):
                                        if getattr(d, 'id', None) == 'Tactics':
                                            found_idx = i
                                            break
                                    if found_idx is None:
                                        principle_details.append(PrincipleTagDetail(id='Tactics', side=None, squares=list(dict.fromkeys(tactic_sqs)), captured=[]))
                                    else:
                                        # merge squares
                                        merged = list(dict.fromkeys((principle_details[found_idx].squares or []) + tactic_sqs))
                                        principle_details[found_idx].squares = merged[:6]
                            except Exception:
                                pass
                            # Add first-move arrows to overlays_after for quick preview
                            try:
                                ov = eng.get("overlays") or {}
                                for a in (ov.get("arrows") or []):
                                    if a.get('from') and a.get('to'):
                                        overlays_after.setdefault('arrows', []).append({
                                            'from': a['from'], 'to': a['to'], 'color': a.get('color') or '#eab308'
                                        })
                            except Exception:
                                pass
                except Exception as _e:
                    print(f"[API DEBUG] Engine tactics error: {_e}", file=sys.stdout)
                # Augment: Undefended pieces principle for both sides with highlight squares
                try:
                    # Precompute starting squares for standard chess
                    W_PAWN_START = { chess.parse_square(f+"2") for f in 'abcdefgh' }
                    B_PAWN_START = { chess.parse_square(f+"7") for f in 'abcdefgh' }
                    W_START = {
                        (True, chess.PAWN): W_PAWN_START,
                        (True, chess.ROOK): { chess.parse_square('a1'), chess.parse_square('h1') },
                        (True, chess.KNIGHT): { chess.parse_square('b1'), chess.parse_square('g1') },
                        (True, chess.BISHOP): { chess.parse_square('c1'), chess.parse_square('f1') },
                        (True, chess.QUEEN): { chess.parse_square('d1') },
                        (True, chess.KING): { chess.parse_square('e1') },
                    }
                    B_START = {
                        (False, chess.PAWN): B_PAWN_START,
                        (False, chess.ROOK): { chess.parse_square('a8'), chess.parse_square('h8') },
                        (False, chess.KNIGHT): { chess.parse_square('b8'), chess.parse_square('g8') },
                        (False, chess.BISHOP): { chess.parse_square('c8'), chess.parse_square('f8') },
                        (False, chess.QUEEN): { chess.parse_square('d8') },
                        (False, chess.KING): { chess.parse_square('e8') },
                    }
                    undef_white: list[str] = []
                    undef_black: list[str] = []
                    for sq in chess.SQUARES:
                        pc = b_for_overlay.piece_at(sq)
                        if not pc:
                            continue
                        color = pc.color  # True=White, False=Black
                        defenders = b_for_overlay.attackers(color, sq)
                        if len(defenders) == 0:
                            # Skip pieces still on starting squares unless attacked by enemy
                            starts = (W_START.get((True, pc.piece_type)) if color else B_START.get((False, pc.piece_type))) or set()
                            on_start = sq in starts
                            enemy_attack = len(b_for_overlay.attackers(not color, sq)) > 0
                            if on_start and not enemy_attack:
                                continue
                            name = chess.square_name(sq)
                            if color:
                                undef_white.append(name)
                            else:
                                undef_black.append(name)
                    if undef_white:
                        principle_details.append(PrincipleTagDetail(id='UndefendedPieces', side='W', squares=undef_white[:6], captured=[]))
                        for s in undef_white[:6]:
                            overlays_after.setdefault('highlights', []).append({'square': s, 'principle': 'UndefendedPieces', 'color': '#60a5faaa'})
                    if undef_black:
                        principle_details.append(PrincipleTagDetail(id='UndefendedPieces', side='B', squares=undef_black[:6], captured=[]))
                        for s in undef_black[:6]:
                            overlays_after.setdefault('highlights', []).append({'square': s, 'principle': 'UndefendedPieces', 'color': '#60a5faaa'})
                except Exception as _e:
                    print(f"[API DEBUG] UndefendedPieces calc error: {_e}", file=sys.stdout)
        except Exception as e:
            print(f"[API DEBUG] Exception in overlays/principles: {e}", file=sys.stdout)
            principles = []
            overlays_after = {"arrows": [], "highlights": []}
            overlays_before = {"arrows": [], "highlights": []}
        try:
            var_list = g.get('variations') or []
            for v in var_list:
                at_ply = v.get('at_ply') if isinstance(v, dict) else None
                if at_ply is None or int(at_ply) != int(req.ply):
                    continue
                line = list(v.get('line') or [])
                label = v.get('label')
                first_san = line[0] if line else None
                first_from = None
                first_to = None
                if chess is not None and b_after is not None and first_san:
                    try:
                        tmp = b_after.copy()
                        mv = tmp.parse_san(first_san)
                        first_from = chess.square_name(mv.from_square)
                        first_to = chess.square_name(mv.to_square)
                    except Exception:
                        pass
                variations.append(VariationItem(label=label, line=line, first_san=first_san, first_from=first_from, first_to=first_to))
        except Exception:
            variations = []
    print(f"[API DEBUG] /ply: Overlays_before returned: {overlays_before}", file=sys.stdout)
    print(f"[API DEBUG] /ply: Overlays_after returned: {overlays_after}", file=sys.stdout)
    # For backward compatibility, keep 'fen' as fen_after
    return PlyResponse(board=board_txt, san=san or [], info=info, fen=fen_after, fen_before=fen_before, fen_after=fen_after, principles=principles, principle_details=principle_details, overlays_before=overlays_before, overlays_after=overlays_after, variations=variations or None)


@app.get("/games/{game_id}/moves", response_model=MovesResponse)
def get_game_moves(game_id: str):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    return MovesResponse(id=game_id, moves=g.get('moves', []))

@app.post("/tactics", response_model=TacticsResponse)
def tactics(req: TacticsRequest):
    if chess is None or analyze_tactics is None:
        return TacticsResponse(tactics=[], overlays={"arrows": [], "highlights": []})
    # Resolve board state from fen or game_id/ply
    board = None
    if req.fen:
        try:
            board = chess.Board(req.fen)
        except Exception:
            board = None
    elif req.game_id is not None and isinstance(req.ply, int):
        g = tutor.get_game_by_id(req.game_id)
        if g and (g.get('moves') is not None):
            board = _board_from_moves(g.get('moves') or [], limit=req.ply)
    if board is None:
        raise HTTPException(status_code=400, detail="Unable to resolve board state for tactics analysis")
    data = analyze_tactics(board)
    # Optionally enrich with engine lines (does not change schema, just adds overlays)
    try:
        if analyze_engine_lines is not None:
            eng = analyze_engine_lines(
                board,
                max_lines=ENGINE_MAX_LINES,
                depth=ENGINE_DEPTH,
                movetime_ms=ENGINE_MOVETIME_MS,
            )
            ov = eng.get("overlays") or {"arrows": [], "highlights": []}
            data_over = data.get("overlays") or {"arrows": [], "highlights": []}
            data_over.setdefault("arrows", []).extend(ov.get("arrows", []))
            data_over.setdefault("highlights", []).extend(ov.get("highlights", []))
            data["overlays"] = data_over
    except Exception:
        pass
    # Augment overlays with UndefendedPieces highlights for both sides
    try:
        hl = data.get("overlays") or {"arrows": [], "highlights": []}
        hl.setdefault("highlights", [])
        # Starting squares cache for standard chess
        W_PAWN_START = { chess.parse_square(f+"2") for f in 'abcdefgh' }
        B_PAWN_START = { chess.parse_square(f+"7") for f in 'abcdefgh' }
        W_START = {
            (True, chess.PAWN): W_PAWN_START,
            (True, chess.ROOK): { chess.parse_square('a1'), chess.parse_square('h1') },
            (True, chess.KNIGHT): { chess.parse_square('b1'), chess.parse_square('g1') },
            (True, chess.BISHOP): { chess.parse_square('c1'), chess.parse_square('f1') },
            (True, chess.QUEEN): { chess.parse_square('d1') },
            (True, chess.KING): { chess.parse_square('e1') },
        }
        B_START = {
            (False, chess.PAWN): B_PAWN_START,
            (False, chess.ROOK): { chess.parse_square('a8'), chess.parse_square('h8') },
            (False, chess.KNIGHT): { chess.parse_square('b8'), chess.parse_square('g8') },
            (False, chess.BISHOP): { chess.parse_square('c8'), chess.parse_square('f8') },
            (False, chess.QUEEN): { chess.parse_square('d8') },
            (False, chess.KING): { chess.parse_square('e8') },
        }
        undef_w = []
        undef_b = []
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if not pc:
                continue
            color = pc.color
            defenders = board.attackers(color, sq)
            if len(defenders) == 0:
                # Ignore pieces still on starting squares unless attacked by enemy
                starts = (W_START.get((True, pc.piece_type)) if color else B_START.get((False, pc.piece_type))) or set()
                on_start = sq in starts
                enemy_attack = len(board.attackers(not color, sq)) > 0
                if on_start and not enemy_attack:
                    continue
                nm = chess.square_name(sq)
                if color:
                    undef_w.append(nm)
                else:
                    undef_b.append(nm)
        for s in undef_w[:6]:
            hl["highlights"].append({"square": s, "principle": "UndefendedPieces", "color": "#60a5faaa"})
        for s in undef_b[:6]:
            hl["highlights"].append({"square": s, "principle": "UndefendedPieces", "color": "#60a5faaa"})
        data["overlays"] = hl
    except Exception:
        pass
    # Map raw items to response model
    items: List[TacticItem] = []
    for t in data.get("tactics", []):
        items.append(TacticItem(
            kind=str(t.get("kind")),
            move=t.get("move"),
            from_sq=t.get("from"),
            to_sq=t.get("to"),
            note=t.get("note")
        ))
    overlays = data.get("overlays") or {"arrows": [], "highlights": []}
    return TacticsResponse(tactics=items, overlays=overlays)


@app.get("/games/{game_id}/explain", response_model=GameExplainResponse)
def explain_game(game_id: str, include_relevant_games: Optional[bool] = None):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    text = tutor.explain_game(game_id, include_relevant_games=include_relevant_games)
    return GameExplainResponse(id=game_id, text=text)


@app.get("/games/{game_id}/principles", response_model=GamePrinciplesResponse)
def game_principles(game_id: str):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    moves = g.get('moves') or []
    out: List[GamePrinciplesTag] = []
    if chess is None:
        return GamePrinciplesResponse(id=game_id, tags=out)
    for ply, _san in enumerate(moves, start=1):
        b = _board_from_moves(moves, limit=ply-1)  # analyze BEFORE the move to align with examples
        if b is None:
            continue
        try:
            tags = analyze_principles(b)
        except Exception:
            tags = []
        if tags:
            out.append(GamePrinciplesTag(ply=ply, principles=tags))
    return GamePrinciplesResponse(id=game_id, tags=out)


def _sync_examples_from_index(clear: bool = False):
    if clear:
        _db_exec("DELETE FROM examples")
    for pid, items in PRINCIPLE_INDEX.items():
        for rec in items:
            _db_exec(
                "INSERT OR IGNORE INTO examples(principle_id, game_id, ply, fen, san) VALUES (?,?,?,?,?)",
                (pid, rec["game_id"], int(rec["ply"]), rec["fen"], rec["san"]),
            )


# Initial sync (best effort)
try:
    if not _db_query("SELECT 1 FROM examples LIMIT 1"):
        _sync_examples_from_index(clear=False)
except Exception:
    pass

# ------------------------------
# Ingestion endpoints (PDF / PGN upload)
# ------------------------------
try:
    from chess_tutor.game_extraction import extract_games_from_pdf, save_games  # type: ignore
    from chess_tutor.pdf_ingest import extract_text_from_pdf  # type: ignore
except Exception:
    try:
        from game_extraction import extract_games_from_pdf, save_games  # type: ignore
        from pdf_ingest import extract_text_from_pdf  # type: ignore
    except Exception:
        extract_games_from_pdf = None  # type: ignore
        extract_text_from_pdf = None  # type: ignore

def _build_principle_counts() -> Dict[str, int]:
    return {pid: len(items) for pid, items in PRINCIPLE_INDEX.items()}

class IngestSummary(BaseModel):
    source: str
    games_added: int
    total_games: int
    principles_indexed: Dict[str, int]
    issues: Optional[List[str]] = None

@app.post("/author/upload_pdf", response_model=IngestSummary)
def author_upload_pdf(
    file: UploadFile = File(...),
    source: str = Form("UploadedPDF"),
    min_moves: int = Form(8)
):
    if extract_text_from_pdf is None or extract_games_from_pdf is None:
        raise HTTPException(status_code=500, detail="PDF ingestion not available (PyMuPDF missing)")
    # Persist uploaded PDF to pdfs/ directory
    try:
        from chess_tutor.config import PDF_DIR  # type: ignore
    except Exception:
        try:
            from config import PDF_DIR  # type: ignore
        except Exception:
            PDF_DIR = str(Path(__file__).resolve().parent / "pdfs")  # type: ignore
    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.pdf").name.replace(" ", "_")
    dest = pdf_dir / safe_name
    data = file.file.read()
    dest.write_bytes(data)
    pages = extract_text_from_pdf(str(dest))
    games = extract_games_from_pdf(pages, source, min_moves=min_moves)
    prev_count = len(tutor.games or [])
    save_games(games)
    # Reload tutor games to include new ones
    try:
        tutor.reload_games()
    except Exception:
        pass
    # Rebuild principle index and sync examples
    try:
        build_principle_index()
        _sync_examples_from_index(clear=False)
        _dedup_examples()
    except Exception:
        pass
    return IngestSummary(
        source=source,
        games_added=len(games),
        total_games=len(tutor.games or []),
        principles_indexed=_build_principle_counts(),
    )

@app.post("/author/upload_pgn", response_model=IngestSummary)
def author_upload_pgn(
    file: UploadFile = File(...),
    source: str = Form("UploadedPGN")
):
    # Parse PGN using python-chess
    if chess is None:
        raise HTTPException(status_code=500, detail="python-chess not installed; PGN parsing unavailable")
    raw = file.file.read().decode("utf-8", errors="ignore")
    import io as _io
    import chess.pgn as _pgn
    games_added: List[Dict[str, Any]] = []
    issues: List[str] = []
    fh = _io.StringIO(raw)
    idx = 0
    while True:
        g = _pgn.read_game(fh)
        if g is None:
            break
        board = g.board()
        moves_san: List[str] = []
        for mv in g.mainline_moves():
            san = board.san(mv)
            moves_san.append(san)
            board.push(mv)
        idx += 1
        headers = getattr(g, 'headers', {}) or {}
        games_added.append({
            "id": f"{source}-PGN{idx}",
            "source": source,
            "moves": moves_san,
            "final_fen": board.fen(),
            "text": raw[:5000],
            "white": headers.get('White'),
            "black": headers.get('Black'),
            "event": headers.get('Event'),
            "site": headers.get('Site'),
            "date": headers.get('Date'),
            "result": headers.get('Result')
        })
    if not games_added:
        # Fallback: attempt plain PGN (no tags) segmentation by results markers
        # Split on result tokens to isolate sequences of moves
        tokens = raw.replace('\r','\n').split('\n')
        flat = ' '.join(t.strip() for t in tokens if t.strip())
        plain_segments = []
        import re as _re
        # Identify result markers; keep them to terminate a segment
        parts = _re.split(r'(?:\s+(?:1-0|0-1|1/2-1/2|\*)\s+)', flat)
        # The split drops delimiters; simple heuristic: if no headers (no '[' ), treat entire as one segment
        if '[' not in raw:
            if len(flat.split()) > 4:
                plain_segments.append(flat)
        if not plain_segments and len(parts) > 1:
            for p in parts:
                p = p.strip()
                if p:
                    plain_segments.append(p)
        if plain_segments:
            for seg in plain_segments:
                # Extract SAN tokens using simple regex for moves with numbers removed
                moves = []
                try:
                    seg_clean = _re.sub(r'\{[^}]*\}', ' ', seg)  # remove comments
                    seg_clean = _re.sub(r'\([^)]*\)', ' ', seg_clean)
                    seg_clean = _re.sub(r'\d+\.(\.\.)?', ' ', seg_clean)
                    parts_moves = seg_clean.split()
                    # Filter probable SAN tokens (exclude results)
                    for m in parts_moves:
                        if m in {'1-0','0-1','1/2-1/2','*'}:
                            break
                        if _re.match(r'^(O-O(-O)?|[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][1-8](?:=[QRBN])?[+#]?)$', m):
                            moves.append(m)
                    if moves:
                        idx += 1
                        board2 = chess.Board() if chess else None
                        if board2 and chess:
                            legal_moves = []
                            for mv_san in moves:
                                try:
                                    mv = board2.parse_san(mv_san)
                                    board2.push(mv)
                                    legal_moves.append(mv_san)
                                except Exception:
                                    continue
                            moves = legal_moves
                        if len(moves) >= 4:  # minimal length sanity
                            games_added.append({
                                'id': f"{source}-PGNPlain{idx}",
                                'source': source,
                                'moves': moves,
                                'final_fen': board2.fen() if board2 else None,
                                'text': seg[:4000],
                                'white': None,
                                'black': None,
                                'event': None,
                                'site': None,
                                'date': None,
                                'result': None
                            })
                except Exception as e:
                    issues.append(f"Plain parse error: {e.__class__.__name__}: {e}")
        if not games_added:
            issues.append('No games parsed (python-chess and plain fallback both failed)')
            return IngestSummary(source=source, games_added=0, total_games=len(tutor.games or []), principles_indexed=_build_principle_counts(), issues=issues or None)
    # Persist into games.json (merge)
    save_games(games_added)
    try:
        tutor.reload_games()
    except Exception:
        pass
    try:
        build_principle_index()
        _sync_examples_from_index(clear=False)
        _dedup_examples()
    except Exception:
        pass
    return IngestSummary(
        source=source,
        games_added=len(games_added),
        total_games=len(tutor.games or []),
        principles_indexed=_build_principle_counts(),
        issues=issues or None
    )

@app.post("/author/clear_pgn_games")
def author_clear_pgn_games():
    """Remove all PGN-uploaded games from the persistent games store.

    Criteria: any game whose id contains 'PGN' (case-insensitive) is treated as a PGN upload.
    This covers both python-chess parsed (..-PGN#) and plain fallback (..-PGNPlain#) identifiers.
    After removal the in-memory tutor games are reloaded and principle index rebuilt (best-effort).
    """
    try:
        from chess_tutor.game_extraction import load_games  # type: ignore
    except Exception:
        try:
            from game_extraction import load_games  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cannot import load_games: {e}")
    try:
        from chess_tutor.config import GAMES_PATH  # type: ignore
    except Exception:
        try:
            from config import GAMES_PATH  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cannot resolve GAMES_PATH: {e}")
    import json, os
    all_games = load_games() or []
    kept = [g for g in all_games if 'PGN' not in str(g.get('id','')).upper()]
    removed = len(all_games) - len(kept)
    # Overwrite file (not merge) so cleared games stay removed
    try:
        os.makedirs(os.path.dirname(GAMES_PATH), exist_ok=True)
        with open(GAMES_PATH, 'w', encoding='utf-8') as f:
            json.dump(kept, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write games file: {e}")
    # Reload tutor + rebuild principle index (best-effort)
    try:
        tutor.reload_games()
    except Exception:
        pass
    try:
        build_principle_index()
        _sync_examples_from_index(clear=False)
        _dedup_examples()
    except Exception:
        pass
    return {
        'removed': removed,
        'remaining': len(kept),
        'principles_indexed': _build_principle_counts()
    }

# ------------------------------
# Course endpoints
# ------------------------------
@app.get("/principles", response_model=List[Principle])
def get_principles():
    if chess is None:
        return []
    rows = _db_query(
        "SELECT p.id, COALESCE(p.name, p.id) as name, COUNT(e.id) as examples FROM principles p LEFT JOIN examples e ON e.principle_id=p.id GROUP BY p.id ORDER BY p.id"
    )
    return [Principle(id=r["id"], name=r["name"], examples=r["examples"]) for r in rows]


@app.get("/principles/{principle_id}/examples", response_model=List[Example])
def get_examples(principle_id: str, limit: int = 50):
    if chess is None:
        return []
    if not _db_query("SELECT 1 FROM principles WHERE id=?", (principle_id,)):
        raise HTTPException(status_code=404, detail="Unknown principle")
    rows = _db_query(
        "SELECT game_id, ply, fen, san FROM examples WHERE principle_id=? ORDER BY id LIMIT ?",
        (principle_id, int(max(1, min(limit, 200)))),
    )
    return [Example(game_id=r["game_id"], ply=r["ply"], fen=r["fen"], san=r["san"]) for r in rows]


@app.post("/exercise/generate", response_model=ExerciseResponse)
def generate_exercise(req: ExerciseGenerateRequest):
    if chess is None:
        raise HTTPException(status_code=400, detail="Course mode unavailable: python-chess not installed")
    pid = _resolve_principle_id(req.principle_id)
    # prefer DB examples
    rows = _db_query(
        "SELECT game_id, ply, fen, san FROM examples WHERE principle_id=? ORDER BY RANDOM() LIMIT 1",
        (pid,),
    )
    if not rows:
        # fallback to in-memory
        ex_list = PRINCIPLE_INDEX.get(pid) or []
        if not ex_list:
            raise HTTPException(status_code=404, detail="No examples for this principle")
        rec = ex_list[0]
    else:
        rec = rows[0]
    kind = req.kind or "bestmove"  # or "identify"
    exercise_id = f"{pid}:{rec['game_id']}:{rec['ply']}:{kind}:{int(datetime.utcnow().timestamp())}"
    if kind == "identify":
        prompt = "Which principle is best illustrated in this position?"
        options = [r["id"] for r in _db_query("SELECT id FROM principles ORDER BY id")] or PRINCIPLES
        solution = pid
    else:
        prompt = "Find the best move (as played in the model game)."
        options = None
        solution = rec["san"]
    _db_exec(
        "INSERT OR REPLACE INTO exercises(id, principle_id, kind, fen, solution, created_at) VALUES (?,?,?,?,?,?)",
        (exercise_id, pid, kind, rec["fen"], str(solution), datetime.utcnow().isoformat()),
    )
    EXERCISES[exercise_id] = {"principle_id": pid, "fen": rec["fen"], "solution": solution, "kind": kind}
    return ExerciseResponse(
        exercise_id=exercise_id,
        principle_id=pid,
        kind=kind,
        fen=rec["fen"],
        prompt=prompt,
        options=options,
        solution=solution,
    )


@app.post("/exercise/submit", response_model=ExerciseSubmitResponse)
def submit_exercise(req: ExerciseSubmitRequest):
    ex_row = _db_query("SELECT * FROM exercises WHERE id=?", (req.exercise_id,))
    ex = EXERCISES.get(req.exercise_id)
    if not ex and ex_row:
        ex = {
            "principle_id": ex_row[0]["principle_id"],
            "fen": ex_row[0]["fen"],
            "solution": ex_row[0]["solution"],
            "kind": ex_row[0]["kind"],
        }
    if not ex:
        raise HTTPException(status_code=404, detail="Exercise not found or expired")
    pid = ex["principle_id"]
    # grading
    if ex["kind"] == "identify":
        correct = req.answer.strip().lower() == pid.lower()
    else:
        correct = req.answer.strip().lower() == str(ex["solution"]).strip().lower()
    _db_exec(
        "INSERT INTO attempts(exercise_id, answer, correct, created_at) VALUES (?,?,?,?)",
        (req.exercise_id, req.answer, 1 if correct else 0, datetime.utcnow().isoformat()),
    )
    # compute progress from DB
    agg = _db_query(
        """
        SELECT SUM(a.correct) as correct, COUNT(*) as total
        FROM attempts a JOIN exercises e ON a.exercise_id=e.id
        WHERE e.principle_id=?
        """,
        (pid,),
    )
    principle_progress = agg[0] if agg else {"correct": 0, "total": 0}
    feedback = "Correct!" if correct else f"Not quite. Expected: {ex['solution']}"
    return ExerciseSubmitResponse(correct=correct, feedback=feedback, principle_progress=principle_progress)


@app.get("/")
def root():
    return {
        "name": "Adaptive Chess Tutor API",
        "version": "0.1",
        "endpoints": [
            "/games", "/games/{id}", "/ply", "/explain", "/health",
            "/principles", "/principles/{id}/examples", 
            "/games/{id}/principles",
            "/exercise/generate", "/exercise/submit", "/progress",
            "/author/principles/upsert", "/author/examples/add", "/author/examples/{id}", 
            "/author/rebuild_from_games", "/author/dedup_examples"
        ],
    }


@app.get("/games", response_model=List[GameSummary])
def list_games(limit: int = 50):
    gs = tutor.list_games(limit=limit)
    out: List[GameSummary] = []
    for g in gs:
        meta = {
            'white': g.get('white'),
            'black': g.get('black'),
            'event': g.get('event'),
            'site': g.get('site'),
            'date': g.get('date'),
            'result': g.get('result')
        }
        out.append(GameSummary(
            id=g['id'],
            length=len(g.get('moves', [])),
            final_fen=g.get('final_fen'),
            **meta
        ))
    return out


@app.get("/games/{game_id}")
def get_game(game_id: str):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    return g


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    ans = tutor.explain(req.query, include_relevant_games=req.include_relevant_games)
    return ExplainResponse(answer=ans)




@app.get("/games/{game_id}/moves", response_model=MovesResponse)
def get_game_moves(game_id: str):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    return MovesResponse(id=game_id, moves=g.get('moves', []))


@app.get("/games/{game_id}/explain", response_model=GameExplainResponse)
def explain_game(game_id: str, include_relevant_games: Optional[bool] = None):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    text = tutor.explain_game(game_id, include_relevant_games=include_relevant_games)
    return GameExplainResponse(id=game_id, text=text)


@app.get("/games/{game_id}/principles", response_model=GamePrinciplesResponse)
def game_principles(game_id: str):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    moves = g.get('moves') or []
    out: List[GamePrinciplesTag] = []
    if chess is None:
        return GamePrinciplesResponse(id=game_id, tags=out)
    for ply, _san in enumerate(moves, start=1):
        b = _board_from_moves(moves, limit=ply-1)  # analyze BEFORE the move to align with examples
        if b is None:
            continue
        try:
            tags = analyze_principles(b)
        except Exception:
            tags = []
        if tags:
            out.append(GamePrinciplesTag(ply=ply, principles=tags))
    return GamePrinciplesResponse(id=game_id, tags=out)


@app.post("/author/dedup_examples")
def author_dedup_examples():
    try:
        _dedup_examples()
        return {"status": "ok", "deduped": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/engine")
def debug_engine():
    try:
        detectors = list(PRINCIPLES_ENGINE.detectors.keys()) if PRINCIPLES_ENGINE else []
    except Exception:
        detectors = []
    try:
        total_examples = _db_query("SELECT COUNT(1) AS c FROM examples")[0]["c"]
    except Exception:
        total_examples = 0
    return {
        "chess_loaded": chess is not None,
        "registry_path": str(REGISTRY_PATH),
        "registry_exists": bool(REGISTRY_PATH.exists()),
        "engine_loaded": PRINCIPLES_ENGINE is not None,
        "detectors": detectors,
        "principles_in_memory": PRINCIPLES,
        "games_count": len(tutor.games or []),
        "total_examples": total_examples,
    }


@app.post("/author/resync_all")
def author_resync_all():
    try:
        # Reload games from disk
        tutor.reload_games()
        # Reload principles registry and refresh in-memory structures
        if PRINCIPLES_ENGINE is not None:
            PRINCIPLES_ENGINE.reload()
        _sync_principles_from_registry()
        _refresh_principles_in_memory()
        # Rebuild in-memory principle index and sync to DB, clearing existing examples
        build_principle_index()
        _sync_examples_from_index(clear=True)
        _dedup_examples()
        # Return summary counts
        games_cnt = len(tutor.games or [])
        princ_cnt = len(PRINCIPLES or [])
        ex_cnt = _db_query("SELECT COUNT(1) AS c FROM examples")[0]["c"]
        return {"status": "ok", "games": games_cnt, "principles": princ_cnt, "examples": ex_cnt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/author/dump_spans")
def author_dump_spans(game_id: str | None = None, source: str | None = None, page: int | None = Query(None, ge=1)):
    """Return raw spans (text + bold flag) for a given PDF source or by game_id's source.
    Params: game_id (e.g., "FundamentalChess Strategy in 100 Games.pdf-G1") or source (PDF filename under pdfs/), optional page number.
    """
    try:
      if not source and game_id:
          g = tutor.get_game_by_id(game_id)
          if not g:
              raise HTTPException(status_code=404, detail="game_id not found")
          source = g.get('source')
      if not source:
          raise HTTPException(status_code=400, detail="Provide game_id or source")
      # Resolve path under pdfs directory
      try:
          from chess_tutor.config import PDF_DIR as _PDF_DIR  # type: ignore
      except Exception:
          try:
              from config import PDF_DIR as _PDF_DIR  # type: ignore
          except Exception:
              _PDF_DIR = str(PKG_DIR / "pdfs")
      pdf_path = (Path(_PDF_DIR) / source)
      if not pdf_path.exists():
          raise HTTPException(status_code=404, detail=f"Source not found: {pdf_path}")
      if extract_text_from_pdf is None:
          raise HTTPException(status_code=500, detail="PDF span extraction not available")
      pages = extract_text_from_pdf(str(pdf_path))
      if page:
          pages = [p for p in pages if int(p.get('page', 0)) == int(page)]
      # Only include spans for brevity
      out = [{"page": p.get("page"), "spans": p.get("spans", [])[:2000]} for p in pages]
      return {"source": source, "pages": out}
    except HTTPException:
      raise
    except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
