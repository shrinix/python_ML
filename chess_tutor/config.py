"""Global configuration for the Adaptive Chess Tutor."""
import os
from pathlib import Path

# Absolute, package-relative paths
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = str(BASE_DIR / "pdfs")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = str(BASE_DIR / "index_store")
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
GAMES_PATH = str(Path(INDEX_PATH) / "games.json")
SHOW_GAME_BOARDS = True
EXTRACT_GAMES_VERBOSE = True

# Feature toggles (future use)
ENABLE_OCR = False  # placeholder for enhancement
INCLUDE_RELEVANT_GAMES_DEFAULT = False  # include "Relevant games" in explanations by default

# RAG (chatbot) feature flags
ENABLE_EXTENDED_RAG = True  # When True, /chat will augment retrieval with principle descriptions & detectors
RAG_TOP_K_DEFAULT = 3       # Default number of text chunks per source category
RAG_MAX_HISTORY = 12        # Max turns from conversation history to send for contextual answer synthesis
RAG_RETURN_GAME_SOURCES = True  # Allow returning game segments as sources (can be disabled when focusing on theory)
RAG_INCLUDE_PRINCIPLE_CODE = False  # If True, include detector source code as context (may be noisy)
RAG_VALIDATION_MODE = os.environ.get("RAG_VALIDATION_MODE", "rewrite").lower()  # "rewrite" | "annotate"

# Optional LLM settings (adapter-based). Defaults keep the app fully offline.
ENABLE_LLM = os.environ.get("ENABLE_LLM", "false").lower() in {"1","true","yes"}
LLM_PROVIDER = "openai"          # currently supported: "openai"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"  # set this env var when using OpenAI


