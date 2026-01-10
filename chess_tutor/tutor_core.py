"""Core tutor class (indexing, retrieval, explanations).

This module prefers FAISS for vector search, but gracefully falls back to a
NumPy-based L2 index when FAISS is unavailable. It also tolerates the
absence of PyMuPDF (fitz) by stubbing PDF ingestion functions so the
backend can still start and work off any prebuilt indices or games.json.
"""
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple

# Optional FAISS with local warning suppression for SWIG deprecations on Python 3.12+
import warnings
try:
    with warnings.catch_warnings():
        # Suppress noisy DeprecationWarnings from SWIG-wrapped types
        warnings.filterwarnings("ignore", ".*SwigPy.*", DeprecationWarning)
        warnings.filterwarnings("ignore", ".*swigvarlink.*", DeprecationWarning)
        import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

# Support running as package only (modern usage)
from .config import (
    PDF_DIR, EMBED_MODEL, INDEX_PATH, SHOW_GAME_BOARDS, INCLUDE_RELEVANT_GAMES_DEFAULT, GAMES_PATH
)

# Try to import PDF ingest (PyMuPDF). If unavailable *or* any dependency inside
# pdf_ingest fails (e.g. config relative import), provide stubs that reflect the
# root cause instead of always blaming PyMuPDF.
_PDF_INGEST_IMPORT_ERROR = None
try:
    try:
        from .pdf_ingest import extract_text_from_pdf, chunk_text  # type: ignore
    except Exception:
        from pdf_ingest import extract_text_from_pdf, chunk_text  # type: ignore
except Exception as _e:  # pragma: no cover - defensive path
    _PDF_INGEST_IMPORT_ERROR = _e
    def extract_text_from_pdf(path: str):  # type: ignore
        reason = f"{_e.__class__.__name__}: {_e}"
        print(f"⚠️ PDF ingest unavailable ({reason}); skipping text for: {path}")
        return []
    def chunk_text(pages, chunk_size=400, overlap=50):  # type: ignore
        return []

# Try to import game extraction utils. If unavailable (due to fitz transitively), stub them.
try:
    try:
        from .game_extraction import (extract_games_from_pdf, save_games, load_games)  # type: ignore
    except Exception:
        from game_extraction import (extract_games_from_pdf, save_games, load_games)  # type: ignore
except Exception:
    import json as _json
    from pathlib import Path as _Path
    def extract_games_from_pdf(pages, fname):  # type: ignore
        return []
    def save_games(games):  # type: ignore
        try:
            _Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
            with open(GAMES_PATH, "w", encoding="utf-8") as f:
                _json.dump(games, f)
        except Exception as e:
            print("Failed to save games:", e)
    def load_games():  # type: ignore
        p = _Path(GAMES_PATH)
        if p.exists():
            try:
                return _json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print("Failed to read games.json:", e)
        return []

try:
    import chess
except ImportError:
    chess = None


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


# ------------------------------
# Minimal NumPy L2 index fallback
# ------------------------------
class NumpyL2Index:
    def __init__(self, dim: int):
        self._dim = int(dim)
        self._vecs = np.zeros((0, self._dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vecs.shape[0])

    def add(self, vecs: np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32)
        vecs = _ensure_2d(vecs)
        if vecs.shape[1] != self._dim:
            raise ValueError(f"Dim mismatch: have {self._dim}, got {vecs.shape[1]}")
        if self._vecs.size == 0:
            self._vecs = vecs.copy()
        else:
            self._vecs = np.vstack([self._vecs, vecs])

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(queries, dtype=np.float32)
        X = _ensure_2d(X)
        if self.ntotal == 0:
            D = np.full((X.shape[0], k), np.inf, dtype=np.float32)
            I = np.full((X.shape[0], k), -1, dtype=np.int64)
            return D, I
        # L2 distances: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        x2 = np.sum(X * X, axis=1, keepdims=True)  # (nq,1)
        y2 = np.sum(self._vecs * self._vecs, axis=1, keepdims=True).T  # (1,nb)
        G = X @ self._vecs.T  # (nq, nb)
        dist = x2 + y2 - 2.0 * G
        # Get top-k smallest distances
        k = min(k, self.ntotal)
        idx = np.argpartition(dist, kth=k-1, axis=1)[:, :k]
        part = np.take_along_axis(dist, idx, axis=1)
        order = np.argsort(part, axis=1)
        I = np.take_along_axis(idx, order, axis=1).astype(np.int64)
        D = np.take_along_axis(part, order, axis=1).astype(np.float32)
        return D, I

    # Persistence helpers for fallback
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self._vecs)

    @classmethod
    def load(cls, path: str) -> "NumpyL2Index":
        vecs = np.load(path, allow_pickle=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        idx = cls(vecs.shape[1] if vecs.ndim == 2 else 0)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        idx.add(vecs)
        return idx


def load_raw_docs():
    docs, metas = [], []
    aggregated_games = []
    if not os.path.isdir(PDF_DIR):
        print(f"⚠️ PDF directory '{PDF_DIR}' not found. Proceeding empty.")
        return docs, metas
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️ No PDF files in '{PDF_DIR}'.")
        return docs, metas
    for fname in pdf_files:
        print(f"📄 Reading {fname}...")
        path = os.path.join(PDF_DIR, fname)
        try:
            pages = extract_text_from_pdf(path)
            games = extract_games_from_pdf(pages, fname)
            if games:
                aggregated_games.extend(games)
            chunks = chunk_text(pages)
            for ch in chunks:
                docs.append(ch["content"])
                metas.append({"source": fname, "page": ch["page"]})
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    if aggregated_games:
        save_games(aggregated_games)
    else:
        print("⚠️ No games extracted.")
    print(f"Loaded {len(docs)} text chunks from {len(pdf_files)} PDFs.")
    return docs, metas


def build_index():
    docs, metas = load_raw_docs()
    cleaned = [(d.strip(), m) for d, m in zip(docs, metas) if d and d.strip()]
    if not cleaned:
        print("No usable documents. Returning empty index.")
        model = SentenceTransformer(EMBED_MODEL)
        dummy_vec = model.encode(["dummy"]).astype("float32")
        dim = int(dummy_vec.shape[1]) if dummy_vec.ndim == 2 else int(dummy_vec.shape[0])
        # Ensure index directory exists before writing empty artifacts
        os.makedirs(INDEX_PATH, exist_ok=True)
        if faiss is not None:
            index = faiss.IndexFlatL2(dim)
            game_index = faiss.IndexFlatL2(dim)
            np.save(os.path.join(INDEX_PATH, "game_text.npy"), np.array([], dtype=object), allow_pickle=True)
            np.save(os.path.join(INDEX_PATH, "game_meta.npy"), np.array([], dtype=object), allow_pickle=True)
            faiss.write_index(game_index, os.path.join(INDEX_PATH, "faiss_games.idx"))
        else:
            index = NumpyL2Index(dim)
            game_index = NumpyL2Index(dim)
            np.save(os.path.join(INDEX_PATH, "game_text.npy"), np.array([], dtype=object), allow_pickle=True)
            np.save(os.path.join(INDEX_PATH, "game_meta.npy"), np.array([], dtype=object), allow_pickle=True)
            # Save empty emb files
            index.save(os.path.join(INDEX_PATH, "emb_docs.npy"))
            game_index.save(os.path.join(INDEX_PATH, "emb_games.npy"))
        return index, model, [], [], game_index, [], []
    docs, metas = zip(*cleaned)
    docs = list(docs)
    metas = list(metas)
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Encoding {len(docs)} docs with model {EMBED_MODEL} ...")
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    embeddings = _ensure_2d(embeddings)
    dim = embeddings.shape[1]
    if faiss is not None:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f"FAISS index populated with {index.ntotal} vectors (dim={dim}).")
    else:
        index = NumpyL2Index(dim)
        index.add(embeddings)
        print(f"NumPy index populated with {index.ntotal} vectors (dim={dim}).")

    # Build per-game index using raw game text from games.json
    games = load_games()
    game_docs = [g.get('text','') for g in games if g.get('text')]
    game_meta = [
        {"id": g.get('id'), "source": g.get('source'), "length": len(g.get('moves', []))}
        for g in games if g.get('text')
    ]
    print(f"Encoding {len(game_docs)} game segments...")
    game_emb = model.encode(game_docs, show_progress_bar=False)
    game_emb = np.asarray(game_emb, dtype="float32")
    game_emb = _ensure_2d(game_emb)
    if faiss is not None:
        game_index = faiss.IndexFlatL2(dim)
        if game_emb.size:
            game_index.add(game_emb)
    else:
        game_index = NumpyL2Index(dim)
        if game_emb.size:
            game_index.add(game_emb)
    os.makedirs(INDEX_PATH, exist_ok=True)
    if faiss is not None:
        faiss.write_index(index, os.path.join(INDEX_PATH, "faiss.idx"))
        faiss.write_index(game_index, os.path.join(INDEX_PATH, "faiss_games.idx"))
    else:
        # Persist embeddings for fallback
        index.save(os.path.join(INDEX_PATH, "emb_docs.npy"))
        game_index.save(os.path.join(INDEX_PATH, "emb_games.npy"))
    np.save(os.path.join(INDEX_PATH, "docs.npy"), np.array(docs, dtype=object), allow_pickle=True)
    np.save(os.path.join(INDEX_PATH, "metas.npy"), np.array(metas, dtype=object), allow_pickle=True)
    np.save(os.path.join(INDEX_PATH, "game_text.npy"), np.array(game_docs, dtype=object), allow_pickle=True)
    np.save(os.path.join(INDEX_PATH, "game_meta.npy"), np.array(game_meta, dtype=object), allow_pickle=True)
    print(f"✅ Index saved to '{INDEX_PATH}'.")
    return index, model, docs, metas, game_index, game_docs, game_meta


def load_index():
    model = SentenceTransformer(EMBED_MODEL)
    docs = np.load(os.path.join(INDEX_PATH, "docs.npy"), allow_pickle=True)
    metas = np.load(os.path.join(INDEX_PATH, "metas.npy"), allow_pickle=True)
    game_docs = np.load(os.path.join(INDEX_PATH, "game_text.npy"), allow_pickle=True)
    game_meta = np.load(os.path.join(INDEX_PATH, "game_meta.npy"), allow_pickle=True)
    # Try FAISS first
    idx_path = os.path.join(INDEX_PATH, "faiss.idx")
    g_idx_path = os.path.join(INDEX_PATH, "faiss_games.idx")
    if faiss is not None and os.path.exists(idx_path) and os.path.exists(g_idx_path):
        index = faiss.read_index(idx_path)
        game_index = faiss.read_index(g_idx_path)
        return index, model, docs, metas, game_index, game_docs, game_meta
    # Fallback: load NumPy indices
    emb_docs_path = os.path.join(INDEX_PATH, "emb_docs.npy")
    emb_games_path = os.path.join(INDEX_PATH, "emb_games.npy")
    if os.path.exists(emb_docs_path):
        index = NumpyL2Index.load(emb_docs_path)
    else:
        # Create empty index if absent
        # Infer dim from model by encoding a dummy vector
        dummy = model.encode(["dummy"]).astype("float32")
        dim = int(dummy.shape[1]) if dummy.ndim == 2 else int(dummy.shape[0])
        index = NumpyL2Index(dim)
    if os.path.exists(emb_games_path):
        game_index = NumpyL2Index.load(emb_games_path)
    else:
        dummy = model.encode(["dummy"]).astype("float32")
        dim = int(dummy.shape[1]) if dummy.ndim == 2 else int(dummy.shape[0])
        game_index = NumpyL2Index(dim)
    return index, model, docs, metas, game_index, game_docs, game_meta


class ChessTutor:
    def __init__(self):
        # Decide whether an index exists: FAISS or NumPy fallback artifacts
        faiss_artifacts = ["faiss.idx", "faiss_games.idx", "docs.npy", "metas.npy", "game_text.npy", "game_meta.npy"]
        numpy_artifacts = ["emb_docs.npy", "emb_games.npy", "docs.npy", "metas.npy", "game_text.npy", "game_meta.npy"]
        has_faiss = all(os.path.exists(os.path.join(INDEX_PATH, f)) for f in faiss_artifacts)
        has_numpy = all(os.path.exists(os.path.join(INDEX_PATH, f)) for f in numpy_artifacts)
        if has_faiss or has_numpy:
            print("📚 Loading existing index...")
            self.index, self.model, self.docs, self.metas, self.game_index, self.game_docs, self.game_meta = load_index()
        else:
            print("⚙️ Building new index...")
            self.index, self.model, self.docs, self.metas, self.game_index, self.game_docs, self.game_meta = build_index()
        self.games = load_games()
        # Sanitize moves: trim any illegal tail to avoid narrative bleed-through
        self._sanitize_all_games_in_memory()
        self.progress = {}
        self.engine = None

    def _sanitize_moves(self, moves):
        if chess is None or not moves:
            return moves or []
        b = chess.Board()
        clean = []
        for san in moves:
            try:
                mv = b.parse_san(san)
            except Exception:
                break
            b.push(mv)
            clean.append(san)
        return clean

    def _sanitize_all_games_in_memory(self):
        if not self.games:
            return
        for g in self.games:
            seq = g.get('moves') or []
            clean = self._sanitize_moves(seq)
            if clean != seq:
                g['moves'] = clean

    def retrieve(self, query, top_k=3):
        if self.docs is None or len(self.docs) == 0:
            print("⚠️ No documents available.")
            return []
        k = min(top_k, len(self.docs))
        vec = self.model.encode([query]).astype("float32")
        try:
            D, I = self.index.search(vec, k)
        except Exception as e:
            print(f"Index search error: {e}")
            return []
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.docs):
                results.append((self.docs[idx], self.metas[idx]))
        return results

    def retrieve_game_segments(self, query, top_k=3):
        if self.game_docs is None or len(self.game_docs) == 0:
            return []
        k = min(top_k, len(self.game_docs))
        vec = self.model.encode([query]).astype("float32")
        try:
            D, I = self.game_index.search(vec, k)
        except Exception as e:
            print(f"Game index search error: {e}")
            return []
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.game_docs):
                results.append((self.game_docs[idx], self.game_meta[idx]))
        return results

    def detect_topic(self, query):
        return query.strip().split()[0].lower() if query.strip() else "general"

    def update_progress(self, topic, delta):
        self.progress[topic] = min(1.0, self.progress.get(topic, 0.0) + delta)

    def show_progress(self):
        if not self.progress:
            print("No progress tracked yet.")
            return
        for t, v in self.progress.items():
            print(f"{t}: {v*100:.1f}%")

    def get_game_by_id(self, game_id):
        # Robust lookup: trim and try URL-decoded id as well
        try:
            from urllib.parse import unquote
        except Exception:
            def unquote(x):
                return x
        q = str(game_id or '').strip()
        # direct match
        for g in self.games:
            gid = str(g.get('id') or '').strip()
            if gid == q:
                return g
        # try URL-decoded
        q2 = unquote(q)
        if q2 and q2 != q:
            for g in self.games:
                gid = str(g.get('id') or '').strip()
                if gid == q2:
                    return g
        # fallback: collapse runs of whitespace
        import re as _re
        def norm(s: str) -> str:
            return _re.sub(r"\s+", " ", s.strip())
        nq = norm(q2 or q)
        for g in self.games:
            if norm(str(g.get('id') or '')) == nq:
                return g
        return None

    # === New game navigation helpers ===
    def _compute_game_boards(self, game):
        """Return list of boards after each ply (index 0 = start position)."""
        if chess is None or not game:
            return []
        boards = [chess.Board()]  # starting position
        b = chess.Board()
        for san in game.get('moves', []):
            try:
                mv = b.parse_san(san)
                b.push(mv)
            except Exception:
                break
            boards.append(b.copy())
        return boards

    def game_ply_count(self, game):
        return len(game.get('moves', [])) if game else 0

    def board_after_ply(self, game_id, ply):
        """Return (ascii_board, san_sequence_up_to_ply, info) for given half-move number (1-based)."""
        game = self.get_game_by_id(game_id)
        if not game:
            return None, None, "Game not found"
        if chess is None:
            return None, None, "python-chess not installed"
        # Use sanitized sequence for navigation
        if game.get('moves'):
            game['moves'] = self._sanitize_moves(game['moves'])
        boards = self._compute_game_boards(game)
        max_ply = len(boards) - 1
        if ply < 0:
            ply = 0
        if ply > max_ply:
            ply = max_ply
        board = boards[ply]
        try:
            board_txt = board.unicode(borders=True)
        except Exception:
            board_txt = str(board)
        san_seq = (game.get('moves') or [])[:ply]
        info = f"Game {game_id} ply {ply}/{max_ply} ({(ply+1)//2 if ply>0 else 0} full moves)"
        return board_txt, san_seq, info

    def _apply_moves(self, moves, start_fen=None, limit=None):
        if chess is None:
            return None
        board = chess.Board(fen=start_fen) if start_fen else chess.Board()
        use_moves = moves if limit is None else moves[:limit]
        for san in use_moves:
            try:
                mv = board.parse_san(san)
                board.push(mv)
            except Exception:
                break
        return board

    def render_board_ascii(self, moves, start_fen=None, limit=None):
        if chess is None:
            return None
        board = self._apply_moves(moves, start_fen=start_fen, limit=limit)
        if board is None:
            return None
        try:
            return board.unicode(borders=True)
        except Exception:
            return str(board)

    def explain(self, query, include_relevant_games: Optional[bool] = None):
        if include_relevant_games is None:
            include_relevant_games = INCLUDE_RELEVANT_GAMES_DEFAULT
        results = self.retrieve(query)
        if not results:
            base_text = "No material found for that query."
        else:
            text = "\n---\n".join(r[0] for r in results)
            topic = self.detect_topic(query)
            self.update_progress(topic, 0.1)
            base_text = f"📘 Topic: {topic}\n{text}"
        if include_relevant_games:
            rel_games = self._relevant_games(query)
            if rel_games:
                lines = ["\nRelevant games (heuristic):"]
                for g in rel_games:
                    mv_slice = " ".join(g['moves'][:16])
                    line = f"- {g['id']}: {mv_slice}{'...' if len(g['moves'])>16 else ''} | final FEN: {g['final_fen']}"
                    if SHOW_GAME_BOARDS and chess is not None:
                        board_txt = self.render_board_ascii(g['moves'])
                        if board_txt:
                            line += f"\n{board_txt}\n"
                    lines.append(line)
                return base_text + "\n" + "\n".join(lines)
        return base_text

    def explain_game(self, game_id: str, include_relevant_games: Optional[bool] = None):
        if include_relevant_games is None:
            include_relevant_games = INCLUDE_RELEVANT_GAMES_DEFAULT
        game = self.get_game_by_id(game_id)
        if not game:
            return "Game not found"
        # Prefer the stored raw text for this specific game to ensure sync with PGN
        game_text = (game.get('text') or '').strip()
        if game_text:
            base_text = f"📘 Game: {game_id}\n{game_text}"
        else:
            # Fallback: attempt retrieval but constrain results to this game id
            mv = " ".join(game.get('moves', [])[:10])
            base_query = f"{game.get('source','')}: {mv}"
            segs = self.retrieve_game_segments(base_query, top_k=8)
            parts = [s[0] for s in segs if (s[1] and s[1].get('id') == game_id)]
            if not parts:
                # Final fallback to general retrieval
                gen = self.retrieve(base_query, top_k=2)
                parts = [g[0] for g in gen]
            base_text = f"📘 Game: {game_id}\n" + ("\n---\n".join(parts) if parts else "No material found for this game.")
        if include_relevant_games:
            mv = " ".join(game.get('moves', [])[:10])
            rel_games = self._relevant_games(mv)
            if rel_games:
                lines = ["\nRelevant games (heuristic):"]
                for g in rel_games:
                    mv_slice = " ".join(g['moves'][:16])
                    line = f"- {g['id']}: {mv_slice}{'...' if len(g['moves'])>16 else ''} | final FEN: {g['final_fen']}"
                    if SHOW_GAME_BOARDS and chess is not None:
                        board_txt = self.render_board_ascii(g['moves'])
                        if board_txt:
                            line += f"\n{board_txt}\n"
                    lines.append(line)
                base_text += "\n" + "\n".join(lines)
        return base_text

    def list_games(self, limit=10):
        return self.games[:limit]

    def find_games_by_move_prefix(self, prefix, limit=5):
        pref = prefix.strip().lower()
        matches = []
        for g in self.games:
            if not g.get('moves'):
                continue
            san_seq = " ".join(g['moves']).lower()
            if san_seq.startswith(pref):
                matches.append(g)
            if len(matches) >= limit:
                break
        return matches

    def _relevant_games(self, query, limit=3):
        if not self.games:
            return []
        tokens = [t for t in re.split(r"\W+", query.lower()) if t]
        if not tokens:
            return []
        scored = []
        for g in self.games:
            mv_lower = [m.lower() for m in g.get('moves', [])][:20]
            joined = " ".join(mv_lower)
            score = sum(1 for t in tokens if t and t in joined)
            if score > 0:
                scored.append((score, g))
        scored.sort(key=lambda x: (-x[0], len(x[1].get('moves', []))))
        return [g for _, g in scored[:limit]]

    def reload_games(self):
        games = []
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        for fname in pdf_files:
            path = os.path.join(PDF_DIR, fname)
            try:
                pages = extract_text_from_pdf(path)
                games.extend(extract_games_from_pdf(pages, fname))
            except Exception as e:
                print(f"Game reload error {fname}: {e}")
        if games:
            save_games(games)
            self.games = load_games()
            print(f"✅ Reloaded {len(self.games)} games.")
        else:
            print("No games found during reload.")
