"""Stub for board image -> FEN classification."""
from typing import List, Dict
from .pdf_ingest import extract_board_images, extract_text_from_pdf

try:
    import chess
except ImportError:
    chess = None


def classify_board_image_to_fen(board_image):
    if chess is None:
        return None
    return chess.STARTING_FEN


def discover_fens_from_pdf(pdf_path):
    fens = []
    for page, img in extract_board_images(pdf_path):
        fen = classify_board_image_to_fen(img)
        if fen:
            fens.append({"type": "image", "page": page, "fen": fen})
    try:
        pages = extract_text_from_pdf(pdf_path)
        # future: parse SAN blocks -> FEN
    except Exception as e:
        print(f"Text FEN extraction error: {e}")
    return fens
