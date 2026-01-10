"""Reindex games from PDFs after parser/regex adjustments.
Run:
    python -m chess_tutor.reindex_games
or from repo root:
    python python_ML/chess_tutor/reindex_games.py
"""
import os, json, sys
from pathlib import Path

try:
    from .config import PDF_DIR, GAMES_PATH, EXTRACT_GAMES_VERBOSE
    from .pdf_ingest import extract_text_from_pdf
    from .game_extraction import extract_games_from_pdf, save_games
except Exception:
    # Fallback absolute imports
    from config import PDF_DIR, GAMES_PATH, EXTRACT_GAMES_VERBOSE  # type: ignore
    from pdf_ingest import extract_text_from_pdf  # type: ignore
    from game_extraction import extract_games_from_pdf, save_games  # type: ignore


def main():
    if not os.path.isdir(PDF_DIR):
        print(f"PDF_DIR '{PDF_DIR}' not found")
        sys.exit(1)
    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(0)
    all_games = []
    for fname in pdfs:
        path = os.path.join(PDF_DIR, fname)
        try:
            pages = extract_text_from_pdf(path)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue
        games = extract_games_from_pdf(pages, fname)
        if EXTRACT_GAMES_VERBOSE:
            print(f"{fname}: {len(games)} games")
        all_games.extend(games)
    # Overwrite games.json
    if not all_games:
        print("No games extracted.")
        return
    # Write directly (bypass merge behavior of save_games to force refresh)
    Path(os.path.dirname(GAMES_PATH)).mkdir(parents=True, exist_ok=True)
    with open(GAMES_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_games, f, indent=2)
    print(f"Reindexed {len(all_games)} total games -> {GAMES_PATH}")
    # Show specific game if requested
    target = os.environ.get('SHOW_GAME_ID')
    if target:
        for g in all_games:
            if g.get('id') == target:
                print(f"Target {target} moves ({len(g['moves'])}): {' '.join(g['moves'])}")
                break

if __name__ == '__main__':
    main()
