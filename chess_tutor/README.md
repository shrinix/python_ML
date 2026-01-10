## CLI Loading & Principle Indexing

Use `cli_loader.py` to ingest sources and build a principle index JSON without starting the FastAPI server.

Examples:

```bash
# PDF book parsing
python cli_loader.py --pdf pdfs/fundamental_chess_strategy.pdf --source "FundamentalChess" --min-moves 8

# PGN file parsing
python cli_loader.py --pgn data/sample_games.pgn --source "SamplePGN"

# PGN parsing without updating games.json (index only)
python cli_loader.py --pgn data/sample_games.pgn --source "SamplePGN" --no-save-games
```

Artifacts written under `index_store/`:
- `games.json` (unless `--no-save-games`)
- `principle_index.json` mapping `principle_id -> [ { game_id, ply, fen, san, side?, squares?, captured? } ]`

Override registry path if maintaining a custom fork:

```bash
python cli_loader.py --pgn data/novel.pgn --registry backend/principles/registry.json
```

Principle occurrences are capped at 2000 per principle to keep JSON manageable.

# Adaptive Chess Tutor (CLI + Web UI)

This project provides an adaptive chess tutor that:
- Indexes chess curriculum PDFs for semantic search with FAISS
- Extracts games (move lists) from PDFs and computes final FENs
- Offers a CLI and a Web UI to browse games, step through moves, and ask questions

The implementation lives in the `chess_tutor/` folder inside this repo.

## Directory Structure

```
python_ML/
└─ chess_tutor/
   ├─ pdfs/                 # Put your chess PDFs here
   ├─ index_store/          # Persisted FAISS index and games.json
   ├─ backend/              # FastAPI server
   │  ├─ app.py
   │  └─ requirements.txt
   ├─ frontend/             # Static web UI (no build step)
   │  ├─ index.html
   │  ├─ main.js
   │  └─ styles.css
   ├─ cli.py                # CLI (package-aware)
   ├─ tutor_cli.py          # CLI (standalone-friendly)
   ├─ tutor_core.py         # Core logic (indexing, retrieval, game navigation)
   ├─ game_extraction.py    # Extract games from PDFs
   ├─ pdf_ingest.py         # PDF text & (stub) board image extraction
   ├─ fen_vision.py         # (stub) board image -> FEN
   ├─ config.py             # Config & feature flags
   └─ chess_tutor.py        # Legacy entry point (safe to run directly)
```

## Prerequisites
- Python 3.9+
- macOS (tested), zsh shell

Recommended: create a virtual environment.

```zsh
cd chess_tutor/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place your PDFs in:
```
python_ML/chess_tutor/pdfs/
```

## Run the CLI
From the repo root (python_ML):

```zsh
# Preferred (package):
python -m chess_tutor.tutor_cli
# Or legacy script:
python chess_tutor/chess_tutor.py
```

Useful commands inside the CLI:
- `list games` (numbered)
- `games <prefix>` (numbered)
- `play <id|#n>` then `next` / `prev` / `goto <ply>`
- `explain <topic>`
- `reload_games` (re-runs extraction)
- `rebuild` (rebuilds FAISS index)

## Run the Web API (FastAPI)

```zsh
cd chess_tutor/backend
source .venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Endpoints (selected):
- `GET /games` → list games
- `GET /games/{id}` → game detail
- `POST /ply` → board + SAN at a given ply: `{ "game_id": ..., "ply": 12 }`
- `POST /explain` → `{ "query": "isolated pawn" }`
- `POST /chat` → RAG chatbot: `{ "messages": [{"role":"user","content":"What is an isolated pawn?"}], "include_games": true }`
- `GET /health` → `{ "status": "ok" }`

## Open the Web UI
Static UI has no build step.

```zsh
cd chess_tutor/frontend
python3 -m http.server 5500
# Visit http://localhost:5500
```
The UI calls the API at `http://localhost:8000`.

## Dependencies and troubleshooting

### PyMuPDF (fitz) not detected or "No module named 'frontend'"

- This project uses PyMuPDF. You should import it as `import fitz`, and the package
  you install is `pymupdf`.
- If you see warnings like:
  - `⚠️ PyMuPDF not installed; skipping PDF text extraction …` or
  - `PyMuPDF import failed (ModuleNotFoundError: No module named 'frontend')`

  then either the server is using a different Python than your virtualenv, or the
  PyMuPDF wheel is not properly installed for that interpreter.

Fixes:

1) Ensure you run the API with the same venv you installed into:

```zsh
cd chess_tutor/backend
source .venv/bin/activate
python -c "import sys; print(sys.executable)"  # should point to chess_tutor/backend/.venv
python -c "import fitz; print('fitz OK:', getattr(fitz,'__file__','<no file>'))"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2) If importing `fitz` raises `No module named 'frontend'`, reinstall PyMuPDF in this venv:

```zsh
pip install --upgrade pip setuptools wheel
pip uninstall -y pymupdf
pip install --no-cache-dir pymupdf
```

If the problem persists on Apple Silicon, try pinning a recent stable version:

```zsh
pip install --no-cache-dir "pymupdf==1.24.11"
```

3) Running from Anaconda or system Python by mistake? Start uvicorn explicitly from the
   venv as shown above, or prefix the command with the venv's python:

```zsh
./.venv/bin/python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### FAISS optional

If FAISS is unavailable, the tutor falls back to a NumPy L2 index automatically. It's
slower but functional. You can install `faiss-cpu` for better performance.

### Empty index rebuilds

If you delete `chess_tutor/index_store/`, the app will recreate it on startup. When PDFs
cannot be ingested (e.g., missing PyMuPDF), it writes empty artifacts so the server still
starts. Once ingestion works, run a rebuild from the CLI or delete `index_store/` again to
force a fresh index.

## RAG Chatbot (Design & Usage)

The project includes a retrieval-augmented chatbot accessible from the homepage "Chat" panel and via `POST /chat`.

### What it does

It answers chess questions using material from:
- Curriculum PDF chunks (semantic search over FAISS)
- Extracted game narratives (semantic search over FAISS)
- Strategy principles registry (short curated descriptions; in-memory embeddings)

The answer is a stitched, extractive summary of the most relevant snippets, plus a compact list of sources.

### Architecture overview

- Vector indices (FAISS)
  - Built/loaded by `chess_tutor/tutor_core.py` into `index_store/`:
    - `faiss.idx` / `docs.npy` / `metas.npy` for PDF chunks
    - `faiss_games.idx` / `game_text.npy` / `game_meta.npy` for game segments
- Extended RAG service
  - `backend/rag_service.py`
  - On startup, it embeds principle descriptions from `backend/principles/registry.json` (small corpus, in-memory)
  - Performs hybrid retrieval: FAISS for PDFs/games + cosine for principle texts, with a small boost for explicit principle mentions
  - Synthesizes an answer by concatenating 1-2 lines per hit category and returns structured `sources`
- API layer
  - `backend/app.py` exposes `POST /chat` and delegates to the RAG service when enabled
- Frontend
  - `frontend/main.js` renders a chat panel with toggles for “Include games” and “Include principles”, and shows sources per reply

### Code changes (where to look)

- Added: `chess_tutor/backend/rag_service.py`
  - Extended retrieval logic, in-memory principle embeddings, answer assembly
- Updated: `chess_tutor/backend/app.py`
  - Enhanced `/chat` endpoint with:
    - Request toggles: `include_games`, `include_principles`
    - Uses `rag_service` when `ENABLE_EXTENDED_RAG` is true; otherwise falls back to legacy retrieval
- Updated: `chess_tutor/frontend/main.js`
  - Chat toggles (Include games/principles)
  - Inline source list under assistant messages
  - Loading indicator
- Updated: `chess_tutor/config.py`
  - New flags to configure the chatbot (see below)

### API: POST /chat

Request body:
```
{
  "messages": [ {"role": "user"|"assistant", "content": "..."}, ... ],
  "top_k": 3,                          # optional, default from config
  "include_games": true,               # optional (default true)
  "include_principles": true           # optional (default true)
  "game_id": "Game_001",              # optional – current game shown in UI
  "ply": 24                            # optional – current ply (0-based or 1-based board state reference)
}
```

Response body:
```
{
  "answer": "Q: …\n\nPrinciple context: …\nGeneral material: …\nGame excerpts: …",
  "sources": [
    { "snippet": "…", "meta": { "type": "principle", "id": "IsolatedPawn" } },
    { "snippet": "…", "meta": { "id": "<game-id>", "length": 42 } }
    { "snippet": "FEN: ...", "meta": { "type": "position", "game_id": "Game_001", "ply": 24 } }
  ]
}
```

When `game_id` (and optionally `ply`) are provided or the user explicitly asks to "explain the position", the backend injects a `position` bucket summarizing:
- FEN (if derivable)
- Detected principles in that position
- Simple material balance heuristic
This contextual snippet is then available to the LLM (if enabled) and appears in the source list.

If the game's originating PDF can be inferred, the position source gains a `page` field (first matching page from existing indexed document chunks) and `source` (PDF filename). Example `meta` for a position source:
```
{
  "type": "position",
  "game_id": "Game_001",
  "ply": 24,
  "source": "StrategyGuide.pdf",
  "page": 17,
  "final_fen": "r1bq..."
}
```

### Explain Position Button (Frontend)

The main UI now includes an "Explain position" button in the Chat panel. Clicking it auto-sends the phrase "Explain the position" along with current `game_id`, `ply`, and if available a `principle` (either deep-linked from Admin Examples or the first detected principle at the board).

Deep-linking from Admin Examples adds `principle=<ID>` to the URL; if absent, the first detected principle at the current ply is used to enrich chat context (as a focus principle source).

Examples:
```zsh
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain the position"}],"game_id":"Game_001","ply":18}' | jq
```
```zsh
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"You: explain the position in this game"}],"game_id":"Game_001","ply":30,"include_games":false}' | jq
```

Example (curl):
```zsh
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is an isolated pawn?"}],"include_games":true}' | jq
```

### Configuration

All flags are in `chess_tutor/config.py`:
```
ENABLE_EXTENDED_RAG = True       # enable the extended RAG pipeline
RAG_TOP_K_DEFAULT = 3            # default number of hits per category
RAG_MAX_HISTORY = 12             # max history turns considered (currently for payload trimming)
RAG_RETURN_GAME_SOURCES = True   # keep game segments as sources
RAG_INCLUDE_PRINCIPLE_CODE = False  # optionally include detector code as context (noisy)
ENABLE_LLM = False               # when True and provider is configured, use an LLM to generate the final answer
LLM_PROVIDER = "openai"           # currently supported: "openai"
LLM_MODEL = "gpt-4o-mini"         # override via env var LLM_MODEL if desired
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
```

### Frontend usage

- Open the homepage and use the Chat panel at the top
- Use the two checkboxes to include/exclude game excerpts or principle summaries
- Source snippets appear below each assistant reply

### Indexing & data

- PDF and game indices are persisted under `chess_tutor/index_store/` and handled by `tutor_core`
- Principle descriptions are embedded in-memory at startup (no separate build step needed)
- If you add or change PDFs, (re)build/load with the existing flows (`reload_games`, `build_index` through `ChessTutor`)

### Troubleshooting

- First run downloads a sentence-transformers model; allow some time
- If you see FAISS import errors, ensure you’ve installed backend requirements:
  - `pip install -r chess_tutor/backend/requirements.txt`
- If no PDFs are present, retrieval may be sparse—place files under `chess_tutor/pdfs/` and rebuild via the CLI or API flows

### Optional: enable LLM generation (OpenAI)

1) Install the OpenAI SDK (optional):
```zsh
pip install openai
```
2) Set your API key and enable LLM mode:
```zsh
export OPENAI_API_KEY=...   # use your key
export LLM_MODEL=gpt-4o-mini  # optional override
```
3) Edit `chess_tutor/config.py`:
```
ENABLE_LLM = True
LLM_PROVIDER = "openai"
```
4) Restart the backend. The `/chat` endpoint now uses the LLM to produce the final answer while still returning sources.


## Troubleshooting
- If CLI shows the old menu, clear caches:
  ```zsh
  find chess_tutor -name '__pycache__' -type d -exec rm -rf {} +
  ```
- If you run `python chess_tutor/chess_tutor.py`, it should detect and load the correct CLI.
- Ensure PDFs exist in `chess_tutor/pdfs/`. If no games were extracted, try `reload_games`.

## Notes & Roadmap
- Board image → FEN is a stub in `fen_vision.py`.
- OCR for scanned PDFs not yet implemented.
- Consider replacing unicode board with a JS chessboard on the frontend.

---

## Course Mode (Strategy Tutor)

The backend includes a course mode that teaches core strategy principles by using positions from the extracted games as examples and exercises.

- Principles (initial set): `OpenFiles`, `Outposts`, `BishopPair`, `PassedPawn`, `RookBehindPassedPawn`, `SpaceAdvantage`, `IsolatedPawn`.
- On API startup, the server scans existing games and heuristically tags positions per principle, building an in-memory index of examples.
- Exercises are generated from these examples; progress is tracked in-memory (resets on restart).

Prerequisites:
- `python-chess` (already in `backend/requirements.txt`).

### How to add or modify course materials

1) Add more games (from books or PDFs)
- Place PDFs in `chess_tutor/pdfs/`.
- Re-run extraction via CLI (`reload_games`) or re-run the CLI/indexer so `index_store/games.json` updates.
- Restart the FastAPI server to rebuild the principle index at startup.

2) Adjust the principle taxonomy
- Edit the list `PRINCIPLES` in `backend/app.py` to add/remove principle IDs.
- Keep IDs short and PascalCase. Example: `"WeakSquares"`, `"MinorityAttack"`.

3) Tune tagging heuristics
- In `backend/app.py`, update the helper functions used by `analyze_principles(...)`, e.g.:
  - `_has_rook_on_open_file`, `_open_files`
  - `_knight_outpost`
  - `_has_bishop_pair`
  - `_has_passed_pawn`, `_rook_behind_passed_pawn`
  - `_space_advantage`
  - `_is_isolated_pawn`
- These run for each position. Adjust thresholds or logic to better match your teaching criteria.
- Restart the server to rebuild the index.

4) Verify examples per principle
- Call the endpoints:
  - `GET /principles` → confirm counts per principle
  - `GET /principles/{id}/examples?limit=20` → inspect `game_id`, `ply`, `FEN`, `SAN`

5) Create and test exercises
- Generate an exercise from a principle:
  ```zsh
  curl -s http://localhost:8000/exercise/generate \
    -H 'Content-Type: application/json' \
    -d '{"principle_id":"OpenFiles","kind":"identify"}' | jq
  ```
- Submit an answer:
  ```zsh
  curl -s http://localhost:8000/exercise/submit \
    -H 'Content-Type: application/json' \
    -d '{"exercise_id":"<copy-from-generate>","answer":"OpenFiles"}' | jq
  ```
- Check progress:
  ```zsh
  curl -s http://localhost:8000/progress | jq
  ```

6) Add curated notes (optional)
- Today, per-principle notes are not persisted. You can extend the backend to store: `Principles`, `Positions`, `Exercises`, `Attempts`, `Progress` in SQLite and return curated text per principle/example.
- Suggested next steps: add a DB layer (SQLAlchemy/SQLite), a `/principles/{id}` detail endpoint with notes, and an authoring endpoint/UI to edit tags and notes.

### Limitations (current MVP)
- In-memory example index and progress (reset on server restart).
- Heuristic tagging is approximate; some positions may be mislabeled.
- No dedicated frontend pages yet for Course Mode (use the API directly or add UI panels).

### Suggested extensions
- Persist principles, examples, and progress in SQLite.
- Add `/principles/rebuild` endpoint to rebuild the index without restart.
- Add frontend: Course tab, Principle pages, Exercises UI with hints.
- Integrate engine (Stockfish) to detect critical moments and improve feedback.

---

## Course Mode Persistence & Authoring (SQLite)

Course data is persisted to SQLite so you can curate content and keep progress across restarts.

- DB file: `chess_tutor/index_store/course.db`
- Tables: `principles`, `examples`, `exercises`, `attempts`

### Inspect or back up the database

```zsh
# Inspect tables
sqlite3 chess_tutor/index_store/course.db \
  'SELECT name FROM sqlite_master WHERE type="table";'

# Count examples per principle
sqlite3 chess_tutor/index_store/course.db \
  'SELECT principle_id, COUNT(*) FROM examples GROUP BY principle_id ORDER BY 2 DESC;'

# Backup
cp chess_tutor/index_store/course.db chess_tutor/index_store/course.backup.$(date +%Y%m%d%H%M%S).db

# Reset (removes all curated data)
rm -f chess_tutor/index_store/course.db
```

### Authoring API

All endpoints are on the FastAPI server (default `http://localhost:8000`). Use `jq` for readability.

1) Upsert a principle (create or update)

```zsh
curl -s http://localhost:8000/author/principles/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "MinorityAttack",
    "name": "Minority Attack",
    "description": "Queenside pawn storm in Carlsbad structures to create a weakness on c6/c3."
  }' | jq
```

2) Add an example to a principle

- You can take a `fen`/`san` from your own curation, from `/principles/{id}/examples`, or compute from a game ply via `/ply`.

```zsh
curl -s http://localhost:8000/author/examples/add \
  -H 'Content-Type: application/json' \
  -d '{
    "principle_id": "OpenFiles",
    "game_id": "Game_001",
    "ply": 17,
    "fen": "r1bq1rk1/pp3ppp/2n1pn2/2bp4/3P4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "san": "Rac1"
  }' | jq
```

3) Delete an example by id

- Example ids are numeric. To find ids, query SQLite directly (or add a list endpoint if desired).

```zsh
# List a few examples with ids
sqlite3 chess_tutor/index_store/course.db \
  'SELECT id, principle_id, game_id, ply FROM examples ORDER BY id DESC LIMIT 10;'

# Delete by id
curl -s -X DELETE http://localhost:8000/author/examples/123 | jq
```

4) Rebuild examples from current games (re-tagging)

- Runs heuristic tagging on the current games and syncs examples to the DB.
- Use `clear=true` to replace existing examples; otherwise it only inserts missing ones.

```zsh
curl -s http://localhost:8000/author/rebuild_from_games \
  -H 'Content-Type: application/json' \
  -d '{"clear": true, "max_games": 500}' | jq
```

### Exercises and progress (DB-backed)

- Generate an exercise (uses DB examples; falls back to in-memory if empty):

```zsh
curl -s http://localhost:8000/exercise/generate \
  -H 'Content-Type: application/json' \
  -d '{"principle_id":"OpenFiles","kind":"bestmove"}' | jq
```

- Submit an answer (response includes up-to-date per-principle progress from attempts):

```zsh
curl -s http://localhost:8000/exercise/submit \
  -H 'Content-Type: application/json' \
  -d '{"exercise_id":"<paste-id>","answer":"Rac1"}' | jq
```

- Note: `GET /progress` currently returns an in-memory counter and may reset on restart; the `exercise/submit` response shows DB-derived progress for the relevant principle.

### Tips

- After adding PDFs or adjusting heuristics, call `/author/rebuild_from_games` to refresh examples without a server restart.
- Prefer using principle ids in PascalCase (e.g., `PassedPawn`, `Outposts`).
- Curate a few high-quality examples per principle before enabling exercises for learners.
