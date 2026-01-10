"""Extended RAG service for the Adaptive Chess Tutor.

Builds an augmented corpus from:
 - Existing PDF text chunks (already indexed in tutor_core via FAISS)
 - Game narrative segments (via tutor_core game_index)
 - Principle registry descriptions (registry.json)
 - Optional principle detector source code (if enabled)

Provides a retrieval function that combines FAISS semantic hits with principle docs
and performs a light re-ranking / boosting around explicit principle mentions.

Note: We DO NOT (yet) merge principle docs into the FAISS index on disk.
They are embedded on startup (cheap: <20 items) and searched via numpy cosine.

Future enhancements:
 - Hybrid sparse + dense scoring
 - LLM generation of final answer (currently extractive summarization)
 - Source filtering / deduplication
"""
from __future__ import annotations

import json, os, inspect, math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re

import numpy as np

try:  # package style
    from chess_tutor.config import (
        ENABLE_EXTENDED_RAG,
        RAG_TOP_K_DEFAULT,
        RAG_INCLUDE_PRINCIPLE_CODE,
        ENABLE_LLM,
        RAG_VALIDATION_MODE,
    )
except Exception:  # fallback standalone
    from config import (
        ENABLE_EXTENDED_RAG,
        RAG_TOP_K_DEFAULT,
        RAG_INCLUDE_PRINCIPLE_CODE,
        ENABLE_LLM,
        RAG_VALIDATION_MODE,
    )  # type: ignore

REGISTRY_REL_PATH = os.path.join(os.path.dirname(__file__), "principles", "registry.json")

# Optional chess import for validation
try:
    import chess  # type: ignore
except Exception:
    chess = None  # type: ignore


@dataclass
class RetrievedDoc:
    text: str
    meta: Dict[str, Any]
    score: float


class ExtendedRAGService:
    def __init__(self, tutor):
        """Initialize with existing tutor_core.ChessTutor instance."""
        self.tutor = tutor
        self.enabled = bool(ENABLE_EXTENDED_RAG)
        self._principle_texts: List[str] = []
        self._principle_metas: List[Dict[str, Any]] = []
        self._principle_emb: np.ndarray | None = None
        self._load_principle_docs()
        # Reuse existing embedding model from tutor
        self.model = getattr(tutor, "model", None)
        if self.model is not None and self._principle_texts:
            self._embed_principles()
        # Try to initialize LLM client (optional). Prefer environment override for testability.
        self._llm = None
        try:
            env_flag = str(os.environ.get("ENABLE_LLM", "")).strip().lower()
            use_llm = (env_flag in {"1", "true", "yes"}) or bool(ENABLE_LLM)
            if use_llm:
                try:
                    from backend.llm_adapter import LLMClient  # type: ignore
                except Exception:
                    from llm_adapter import LLMClient  # type: ignore
                self._llm = LLMClient()
                if not getattr(self._llm, 'enabled', False):
                    print("[LLM][info] LLMClient created but .enabled False (runtime toggle?)")
                else:
                    print("[LLM][info] LLM enabled; provider=", getattr(self._llm,'provider',None), "model=", getattr(self._llm,'model',None))
            else:
                print("[LLM][info] ENABLE_LLM flag false; using extractive mode.")
        except Exception as e:
            print(f"[LLM][warn] Failed to init LLM client: {e.__class__.__name__}: {e}")
            self._llm = None

    # ----------------------------
    # Corpus construction
    # ----------------------------
    def _load_principle_docs(self):
        if not os.path.isfile(REGISTRY_REL_PATH):
            return
        try:
            with open(REGISTRY_REL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        items = data.get("principles") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return
        for spec in items:
            pid = spec.get("id")
            desc = (spec.get("description") or "").strip()
            if not pid:
                continue
            # Base document: description
            if desc:
                self._principle_texts.append(f"Principle {pid}: {desc}")
                self._principle_metas.append({"type": "principle", "id": pid, "field": "description"})
            # Optionally include detector code
            if RAG_INCLUDE_PRINCIPLE_CODE:
                module_name = spec.get("module")
                if module_name:
                    src = self._load_detector_source(module_name)
                    if src:
                        trimmed = self._trim_source(src)
                        self._principle_texts.append(f"Detector {pid} code: {trimmed}")
                        self._principle_metas.append({"type": "principle_code", "id": pid, "module": module_name})

    def _load_detector_source(self, module_name: str) -> str | None:
        try:
            # dynamic import relative to principles package
            import importlib
            mod = importlib.import_module(f"backend.principles.{module_name}")
        except Exception:
            try:
                mod = importlib.import_module(f"principles.{module_name}")
            except Exception:
                return None
        try:
            return inspect.getsource(mod)
        except Exception:
            return None

    def _trim_source(self, src: str, max_lines: int = 60) -> str:
        lines = [l for l in src.splitlines() if l.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[: max_lines]) + "\n..."

    def _embed_principles(self):
        try:
            vecs = self.model.encode(self._principle_texts, show_progress_bar=False)
            vecs = np.asarray(vecs, dtype="float32")
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            self._principle_emb = vecs
        except Exception:
            self._principle_emb = None

    # ----------------------------
    # Retrieval
    # ----------------------------
    def retrieve(self, query: str, top_k: int | None = None, include_games: bool = True) -> Dict[str, List[RetrievedDoc]]:
        """Return dict of category -> list[RetrievedDoc]."""
        if not self.enabled:
            # Fallback to tutor direct retrieval (legacy)
            k = top_k or RAG_TOP_K_DEFAULT
            docs = self.tutor.retrieve(query, top_k=k) or []
            out = [RetrievedDoc(text=d[0], meta=d[1] or {}, score=1.0) for d in docs]
            gdocs = []
            if include_games:
                g = self.tutor.retrieve_game_segments(query, top_k=k) or []
                gdocs = [RetrievedDoc(text=x[0], meta=x[1] or {}, score=1.0) for x in g]
            return {"docs": out, "games": gdocs}

        k = top_k or RAG_TOP_K_DEFAULT
        results: Dict[str, List[RetrievedDoc]] = {"docs": [], "principles": [], "games": []}

        # General PDF chunks (FAISS)
        try:
            pdf_hits = self.tutor.retrieve(query, top_k=k) or []
            results["docs"] = [RetrievedDoc(text=t, meta=m or {}, score=1.0) for t, m in pdf_hits]
        except Exception:
            results["docs"] = []

        # Principle docs (cosine similarity)
        if self._principle_emb is not None and self._principle_emb.size:
            try:
                q_vec = self.tutor.model.encode([query]).astype("float32")
                q_vec = q_vec.reshape(1, -1)
                # cosine similarity: v dot w / ||v|| ||w||
                denom = (np.linalg.norm(q_vec, axis=1, keepdims=True) * np.linalg.norm(self._principle_emb, axis=1))
                sims = (self._principle_emb @ q_vec.T).flatten() / (denom.flatten() + 1e-9)
                # boost if explicit principle id token appears in query
                boosts = []
                q_lower = query.lower()
                for meta, base in zip(self._principle_metas, sims):
                    pid = meta.get("id", "").lower()
                    boost = 0.3 if pid and pid.lower() in q_lower else 0.0
                    boosts.append(base + boost)
                order = np.argsort(-np.asarray(boosts))[:k]
                princip_hits = []
                for idx in order:
                    princip_hits.append(
                        RetrievedDoc(
                            text=self._principle_texts[idx],
                            meta=self._principle_metas[idx],
                            score=float(boosts[idx]),
                        )
                    )
                results["principles"] = princip_hits
            except Exception:
                results["principles"] = []

        # Game segments (FAISS) optional
        if include_games:
            try:
                g_hits = self.tutor.retrieve_game_segments(query, top_k=k) or []
                results["games"] = [RetrievedDoc(text=t, meta=m or {}, score=1.0) for t, m in g_hits]
            except Exception:
                results["games"] = []

        return results

    # ----------------------------
    # Answer assembly
    # ----------------------------
    def synthesize_answer(self, query: str, buckets: Dict[str, List[RetrievedDoc]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Compose a final answer using an LLM if enabled; otherwise extractive summary.

        Returns: (answer_text, sources_meta_list)
        """
        lines: List[str] = []
        sources: List[Dict[str, Any]] = []

        def first_sentences(text: str, max_sentences: int = 2) -> List[str]:
            parts = [p.strip() for p in text.split("\n") if p.strip()]
            if not parts:
                return []
            out = []
            for p in parts:
                out.append(p)
                if len(out) >= max_sentences:
                    break
            return out

        # Position context first (if any)
        pos = buckets.get("position") or []
        if pos:
            lines.append("Position context:")
            for doc in pos:
                # include up to 2 lines from position description
                for s in first_sentences(doc.text, 2):
                    lines.append(f"- {s}")
                sources.append({"snippet": doc.text[:500], "meta": doc.meta})

        # Principles next (if any)
        princip = buckets.get("principles") or []
        if princip:
            lines.append("Principle context:")
            for doc in princip:
                for s in first_sentences(doc.text, 1):
                    lines.append(f"- {s}")
                sources.append({"snippet": doc.text[:500], "meta": doc.meta})

        # General docs
        gen = buckets.get("docs") or []
        if gen:
            lines.append("General material:")
            for doc in gen:
                for s in first_sentences(doc.text, 1):
                    lines.append(f"- {s}")
                sources.append({"snippet": doc.text[:500], "meta": doc.meta})

        # Games
        game = buckets.get("games") or []
        if game:
            lines.append("Game excerpts:")
            for doc in game:
                meta = doc.meta or {}
                gid = meta.get("id") or meta.get("source")
                first = first_sentences(doc.text, 1)
                if first:
                    lines.append(f"- {gid}: {first[0]}")
                sources.append({"snippet": doc.text[:500], "meta": meta})

        if not lines:
            lines = ["I couldn't find relevant material. Try rephrasing or asking about a specific principle or move sequence."]

        # If an LLM is available, generate a concise answer using the retrieved context
        if self._llm is not None and getattr(self._llm, 'enabled', False):
            context_blocks = []
            def add(cat):
                for d in (buckets.get(cat) or [])[:3]:
                    context_blocks.append(f"[{cat}] {d.meta} :: {d.text[:1200]}")
            add("position"); add("principles"); add("docs"); add("games")
            system_prompt = (
                "You are a helpful chess tutor. Use ONLY the provided context. "
                "If a Position context is present, you MUST base the analysis on that exact board state and echo the exact FEN given there. "
                "Do NOT invent or alter a FEN. Respect the side to move from the Position context. "
                "Prefer concrete explanations (why, plans, key squares) and keep it concise."
            )
            user_prompt = (
                f"Question: {query}\n\nContext (snippets):\n" + "\n".join(context_blocks[:10]) +
                "\n\nInstructions:\n"
                "- If Position context is present, start by echoing the exact FEN from it (e.g., 'FEN: ...').\n"
                "- Do not provide any different FEN.\n"
                "- Consider the listed 'Side to move' if present.\n"
                "- Cross-check pawn-structure claims against the provided ASCII board and the 'Pawns — White: …; Black: …' lists; avoid asserting a pawn on a square if not shown.\n"
                "- Then give a brief, accurate evaluation and 2-3 key ideas grounded in the position."
            )
            gen = self._llm.generate(system_prompt, user_prompt, temperature=0.2)
            if gen:
                # Post-validate against FEN from position context
                try:
                    fen = None
                    for d in (buckets.get("position") or [])[:1]:
                        m = re.search(r"FEN:\s*([^\n]+)", d.text)
                        if m:
                            fen = m.group(1).strip()
                            break
                    if fen and chess is not None:
                        fixed = _validate_and_correct_answer(gen, fen)
                        if fixed:
                            gen = fixed
                except Exception:
                    pass
                return gen, sources

        # Fallback: extractive stitching
        else:
            if self._llm is None:
                print("[LLM][debug] Skipping generation: _llm is None")
            elif not getattr(self._llm,'enabled',False):
                print("[LLM][debug] Skipping generation: _llm.enabled=False")
        answer = f"Q: {query}\n\n" + "\n".join(lines)
        return answer, sources


# Convenience factory (lazy optional use)
def get_extended_rag_service(tutor) -> ExtendedRAGService:
    return ExtendedRAGService(tutor)


# ----------------------------
# Answer validation helpers
# ----------------------------
def _validate_and_correct_answer(answer: str, fen: str) -> str:
    """Option B: rewrite/remove conflicting statements for a clean final answer.

    - Correct "White/Black to move" to match FEN if mentioned.
    - Remove any sentence that asserts a pawn on a square not present in the FEN
      (e.g., "black pawn on d6" when no such pawn exists).
    """
    try:
        if chess is None:
            return answer
        board = chess.Board(fen)
        wp = {chess.square_name(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)}
        bp = {chess.square_name(sq) for sq in board.pieces(chess.PAWN, chess.BLACK)}
        stm = 'White' if board.turn == chess.WHITE else 'Black'

        mode = (RAG_VALIDATION_MODE or 'rewrite').lower()

        if mode == 'annotate':
            corrections = []
            # Side-to-move consistency: if answer claims other side explicitly
            stm_pat = re.search(r"\b(white|black)\s+(to move|turn to move)\b", answer, re.IGNORECASE)
            if stm_pat and stm_pat.group(1).lower() != stm.lower():
                corrections.append(f"Side to move is {stm} per FEN, not {stm_pat.group(1).title()}.")

            # Pawn-square claims
            pattern = re.compile(r"\b(white|black)?\s*pawn\s+(?:on|at)\s+([a-h][1-8])\b", re.IGNORECASE)
            for m in pattern.finditer(answer):
                color = (m.group(1) or '').lower()
                sq = m.group(2).lower()
                if color == 'white' and (sq not in wp):
                    corrections.append(f"No white pawn on {sq} in this position.")
                elif color == 'black' and (sq not in bp):
                    corrections.append(f"No black pawn on {sq} in this position.")
                elif not color and (sq not in wp and sq not in bp):
                    corrections.append(f"No pawn on {sq} in this position.")

            if corrections:
                note = "\n\nNote: Position-based checks:\n- " + "\n- ".join(sorted(set(corrections)))
                return answer.strip() + note
            return answer
        else:
            # Rewrite mode: replace incorrect side-to-move and remove false pawn-claim sentences
            def fix_side_to_move(text: str) -> str:
                def repl(_m):
                    return f"{stm} to move"
                return re.sub(r"\b(White|Black)\s+(to move|turn to move)\b", repl, text, flags=re.IGNORECASE)

            pawn_pat = re.compile(r"\b(white|black)?\s*pawn\s+(?:on|at)\s+([a-h][1-8])\b", re.IGNORECASE)
            sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
            cleaned = []
            for sent in sentences:
                s = sent.strip()
                if not s:
                    continue
                s = fix_side_to_move(s)
                drop = False
                for m in pawn_pat.finditer(s):
                    color = (m.group(1) or '').lower()
                    sq = m.group(2).lower()
                    if color == 'white' and (sq not in wp):
                        drop = True; break
                    if color == 'black' and (sq not in bp):
                        drop = True; break
                    if not color and (sq not in wp and sq not in bp):
                        drop = True; break
                if not drop:
                    cleaned.append(s)
            final = '\n'.join(cleaned).strip()
            return final if final else answer
    except Exception:
        return answer
