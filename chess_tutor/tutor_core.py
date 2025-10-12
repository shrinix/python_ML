"""Core tutor class (indexing, retrieval, explanations)."""
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Support running as package or standalone script
try:  # package style
    from .config import (PDF_DIR, EMBED_MODEL, INDEX_PATH, SHOW_GAME_BOARDS)
    from .pdf_ingest import extract_text_from_pdf, chunk_text
    from .game_extraction import (extract_games_from_pdf, save_games, load_games)
except ImportError:  # standalone fallback
    from config import (PDF_DIR, EMBED_MODEL, INDEX_PATH, SHOW_GAME_BOARDS)  # type: ignore
    from pdf_ingest import extract_text_from_pdf, chunk_text  # type: ignore
    from game_extraction import (extract_games_from_pdf, save_games, load_games)  # type: ignore

try:
    import chess
except ImportError:
    chess = None


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
        index = faiss.IndexFlatL2(dim)
        return index, model, [], []
    docs, metas = zip(*cleaned)
    docs = list(docs)
    metas = list(metas)
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Encoding {len(docs)} docs with model {EMBED_MODEL} ...")
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index populated with {index.ntotal} vectors (dim={dim}).")
    os.makedirs(INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_PATH, "faiss.idx"))
    np.save(os.path.join(INDEX_PATH, "docs.npy"), np.array(docs, dtype=object), allow_pickle=True)
    np.save(os.path.join(INDEX_PATH, "metas.npy"), np.array(metas, dtype=object), allow_pickle=True)
    print(f"✅ Index saved to '{INDEX_PATH}'.")
    return index, model, docs, metas


def load_index():
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(os.path.join(INDEX_PATH, "faiss.idx"))
    docs = np.load(os.path.join(INDEX_PATH, "docs.npy"), allow_pickle=True)
    metas = np.load(os.path.join(INDEX_PATH, "metas.npy"), allow_pickle=True)
    return index, model, docs, metas


class ChessTutor:
    def __init__(self):
        if os.path.exists(INDEX_PATH) and all(os.path.exists(os.path.join(INDEX_PATH, f)) for f in ["faiss.idx", "docs.npy", "metas.npy"]):
            print("📚 Loading existing index...")
            self.index, self.model, self.docs, self.metas = load_index()
        else:
            print("⚙️ Building new index...")
            self.index, self.model, self.docs, self.metas = build_index()
        self.games = load_games()
        self.progress = {}
        self.engine = None

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
        for g in self.games:
            if g.get('id') == game_id:
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
        san_seq = game.get('moves', [])[:ply]
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

    def explain(self, query):
        results = self.retrieve(query)
        if not results:
            base_text = "No material found for that query."
        else:
            text = "\n---\n".join(r[0] for r in results)
            topic = self.detect_topic(query)
            self.update_progress(topic, 0.1)
            base_text = f"📘 Topic: {topic}\n{text}"
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
        else:
            base_text += "\nNo relevant games found"
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
