# Chess Tutor Design and Implementation Document

## Overview
The Adaptive Chess Tutor is a Python-based application that provides an interactive chess learning environment. It combines a FastAPI backend, a static JavaScript frontend, and a set of core modules for chess principle detection, game navigation, and retrieval-augmented generation (RAG) chat. The system is designed to:
- Index chess games and curriculum PDFs
- Detect and visualize chess principles (e.g., attacked pieces, isolated pawns)
- Provide a web UI for browsing, stepping through games, and asking questions
- Offer overlays (arrows, highlights) to illustrate tactical/strategic concepts

## Architecture

### Directory Structure
- `backend/`: FastAPI server, principle detection, RAG service
- `frontend/`: Static web UI (HTML, JS, CSS)
- `index_store/`: Persisted indices (FAISS, games, principle index)
- `pdfs/`: Source PDFs for ingestion
- `cli_loader.py`, `tutor_cli.py`: CLI tools for indexing and navigation
- `tutor_core.py`: Core logic for indexing, retrieval, and game navigation

### Backend (FastAPI)
- **Entrypoint:** `backend/app.py`
- **Key Endpoints:**
  - `POST /ply`: Returns board state, FEN, principles, overlays, and variations for a given game and ply
  - `POST /chat`: RAG chatbot for chess questions
  - `GET /games`, `/games/{id}/moves`, `/games/{id}/explain`: Game navigation and explanations
- **Principle Detection:**
  - Pluggable engine (`backend/principles/engine.py`) loads detectors from `registry.json`
  - Example: `attacked_pieces.py` detects attacked pieces, returns impacted squares and overlays
- **Overlay Calculation:**
  - Overlays (arrows, highlights) are computed statelessly from the board (FEN)
  - Returned as part of the `/ply` response for frontend rendering
- **RAG Service:**
  - Hybrid retrieval over PDF/game FAISS indices and in-memory principle embeddings
  - Synthesizes answers and sources for the chatbot

### Frontend (Static JS)
- **Entrypoint:** `frontend/index.html`, `frontend/main.js`
- **Features:**
  - Game browser, move navigation, principle badges
  - Board rendering from FEN, overlays (arrows, highlights)
  - Chat panel with RAG chatbot, toggles for games/principles
  - Explain position button auto-sends context to backend
- **Overlay Rendering:**
  - Receives overlays from `/ply` (arrows, highlights)
  - Renders arrows using SVG, highlights squares with CSS
  - Overlays persist until the attack is resolved, regardless of side to move

### Principle Detection & Overlays
- **Stateless:** All detection is based on the current board state (FEN)
- **Attacked Pieces Example:**
  - `attacked_pieces.py` scans all pieces, records attacks as (from, to) pairs
  - Returns impacted side, squares, and overlays (arrows)
  - Overlays are always shown if attacks exist, until resolved

### Data Flow
1. **Frontend** sends `/ply` request with `game_id`, `ply`, and current `fen`
2. **Backend** computes board state, detects principles, calculates overlays
3. **Backend** returns board, FEN, principles, overlays, and variations
4. **Frontend** renders board, overlays, principle badges, and variations
5. **Frontend** can send chat requests with context for RAG answers

### Debugging & Logging
- Debug prints/logs are present in both backend and frontend for FEN, overlays, and attack detection
- Ensures board state sync and overlay persistence can be diagnosed

## Key Files
- `backend/app.py`: FastAPI API, board state, overlays, chat
- `backend/principles/attacked_pieces.py`: Stateless attack detection, overlay logic
- `frontend/main.js`: Board rendering, overlay drawing, chat, UI logic
- `README.md`: Setup, usage, and troubleshooting

## Extensibility
- New principles can be added by implementing a detector in `backend/principles/` and registering in `registry.json`
- Overlays and highlights are extensible via the principle visualizer interface
- RAG service can be extended with new retrieval sources

## Setup & Usage

## Running the Application (Frontend & Backend)

### 1. Environment Setup
- **Python 3.9+** is required. Recommended: use a virtual environment.

```zsh
cd chess_tutor/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Backend (API Server)
- From `chess_tutor/backend` (with venv activated):

```zsh
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- The API will be available at `http://localhost:8000`.

### 3. Frontend (Web UI)
- From `chess_tutor/frontend`:

```zsh
python3 -m http.server 5500
```
- Open your browser to [http://localhost:5500](http://localhost:5500)
- The UI will connect to the backend API at `http://localhost:8000`

### 4. Ingesting Data
- Place chess PDFs in `chess_tutor/pdfs/`
- Use the CLI to ingest and index:

```zsh
python cli_loader.py --pdf pdfs/your_book.pdf --source "MyBook"
```
- Or use the web UI/CLI to reload games and rebuild indices as needed.

### 5. Debugging the App

- **Backend Debugging:**
  - The backend prints debug info (FEN, overlays, attacks) to stdout/stderr.
  - To see debug output, run the backend in a terminal and watch for `[API DEBUG]` and `[DEBUG]` lines.
  - If overlays/arrows are missing, compare the FEN sent by the frontend (see frontend console) with the backend logs.
  - Use `print()` or logging in Python to add more debug output as needed.

- **Frontend Debugging:**
  - Open browser DevTools (F12) and check the Console for `[DEBUG]` logs (FEN sent, overlays received).
  - Network tab shows API requests/responses for `/ply`, `/chat`, etc.
  - You can add `console.log()` statements in `main.js` for additional debugging.

- **Common Issues:**
  - **No overlays/arrows:** Ensure FEN sent by frontend matches backend board state. Check debug logs.
  - **No games loaded:** Make sure PDFs are in `pdfs/` and indexed. Use CLI or API to reload.
  - **Module import errors:** Activate the correct Python venv before running the backend.
  - **PyMuPDF/FAISS errors:** See troubleshooting in `README.md` for installation tips.

### 6. Stopping the App
- Stop the backend and frontend servers with `Ctrl+C` in their respective terminals.

## Troubleshooting
- Ensure Python venv is activated for backend
- Use debug logs to verify FEN and overlay sync
- See `README.md` for PyMuPDF/FAISS issues

---
This document summarizes the design and implementation of the Adaptive Chess Tutor codebase in `chess_tutor/`.
