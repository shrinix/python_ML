"""Global configuration for the Adaptive Chess Tutor."""
import os

PDF_DIR = "pdfs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "index_store"
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
GAMES_PATH = os.path.join(INDEX_PATH, "games.json")
SHOW_GAME_BOARDS = True
EXTRACT_GAMES_VERBOSE = True

# Feature toggles (future use)
ENABLE_OCR = False  # placeholder for enhancement
