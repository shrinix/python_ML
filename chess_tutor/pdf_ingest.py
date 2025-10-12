"""PDF ingestion: text & (stub) board image extraction."""
from io import BytesIO
import re
import fitz
from typing import List, Dict, Tuple
from .config import PDF_DIR, EXTRACT_GAMES_VERBOSE

try:
    from PIL import Image
except ImportError:
    Image = None


def extract_text_from_pdf(path: str) -> List[Dict]:
    doc = fitz.open(path)
    text = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        if txt.strip():
            text.append({"page": i + 1, "content": txt})
    return text


def chunk_text(pages: List[Dict], chunk_size=400, overlap=50):
    chunks = []
    for entry in pages:
        words = entry["content"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({"page": entry["page"], "content": chunk})
    return chunks


def extract_board_images(pdf_path: str, min_size=200):
    if Image is None:
        if EXTRACT_GAMES_VERBOSE:
            print("Pillow not installed; skipping image extraction")
        return []
    boards = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open {pdf_path}: {e}")
        return boards
    for pno, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                w, h = base.get('width'), base.get('height')
                if not w or not h:
                    continue
                aspect = w / h
                if w >= min_size and h >= min_size and 0.85 <= aspect <= 1.15:
                    pil_img = Image.open(BytesIO(base['image']))
                    boards.append((pno, pil_img))
            except Exception as ie:
                print(f"Image extract error on page {pno}: {ie}")
    return boards
