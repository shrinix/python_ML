from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Dual import to work in package or direct run
try:
    from ..tutor_core import ChessTutor
except ImportError:
    from chess_tutor.tutor_core import ChessTutor  # type: ignore

app = FastAPI(title="Adaptive Chess Tutor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tutor = ChessTutor()


class ExplainRequest(BaseModel):
    query: str


class ExplainResponse(BaseModel):
    answer: str


class GameSummary(BaseModel):
    id: str
    length: int
    final_fen: Optional[str]


class PlyRequest(BaseModel):
    game_id: str
    ply: int


class PlyResponse(BaseModel):
    board: str
    san: List[str]
    info: str


@app.get("/games", response_model=List[GameSummary])
def list_games(limit: int = 50):
    gs = tutor.list_games(limit=limit)
    return [GameSummary(id=g['id'], length=len(g.get('moves', [])), final_fen=g.get('final_fen')) for g in gs]


@app.get("/games/{game_id}")
def get_game(game_id: str):
    g = tutor.get_game_by_id(game_id)
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    return g


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    return ExplainResponse(answer=tutor.explain(req.query))


@app.post("/ply", response_model=PlyResponse)
def ply(req: PlyRequest):
    board, san, info = tutor.board_after_ply(req.game_id, req.ply)
    if board is None:
        raise HTTPException(status_code=400, detail=info or "Invalid request")
    return PlyResponse(board=board, san=san or [], info=info)


@app.get("/health")
def health():
    return {"status": "ok"}
