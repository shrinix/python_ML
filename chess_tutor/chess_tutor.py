"""
Adaptive Chess Tutor Prototype
Author: ChatGPT
Description:
  - Loads PDF chess curriculum (advanced strategy)
  - Builds FAISS-based semantic search
  - Provides CLI chatbot for explanations & progress tracking
  - Optional Stockfish integration for board evaluation
"""

# This monolithic script has been refactored into modular components:
#  - config.py
#  - pdf_ingest.py
#  - game_extraction.py
#  - fen_vision.py
#  - tutor_core.py
#  - cli.py (entry point)
# Keeping this file for backward compatibility; it now simply launches the CLI.
"""
Legacy entry point for running the Adaptive Chess Tutor.
Preferred usage:
  python -m chess_tutor            (from project root)
  python -m chess_tutor.tutor_cli  (explicit CLI)
Fallback: python chess_tutor/chess_tutor.py (now supported without relative import errors)
"""
import sys, pathlib, importlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
PARENT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# Ensure package context variable so relative imports inside modules succeed
if __package__ in (None, ""):
    __package__ = "chess_tutor"

CANDIDATES = ["chess_tutor.tutor_cli", "chess_tutor.cli", "tutor_cli", "cli"]
loaded = None
errors = {}
for name in CANDIDATES:
    try:
        mod = importlib.import_module(name)
        if hasattr(mod, "main"):
            loaded = mod
            break
    except Exception as e:
        errors[name] = repr(e)

if not loaded:
    print("Failed to import CLI entry point. Tried:")
    for k, v in errors.items():
        print(f"  {k}: {v}")
    sys.exit(1)

print(f"[INFO] Using CLI module: {getattr(loaded, '__file__', '?')}")
main = getattr(loaded, "main")

if __name__ == "__main__":
    main()