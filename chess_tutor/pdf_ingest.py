"""PDF ingestion: text & (stub) board image extraction.

Robust import strategy so this module works whether imported as part of the
`chess_tutor` package or executed in a loose script context. Previously a
strict relative import (`from .config import ...`) could raise an ImportError
even when PyMuPDF was installed, causing the upstream loader to fall back to
stubs and emit a misleading warning "PyMuPDF not installed". This file now
tries both relative and absolute imports for `config`, and degrades gracefully
if neither is available.
"""
from io import BytesIO
import re
from typing import List, Dict, Tuple

# PyMuPDF provides the `fitz` module. Treat any exception as absence but *do not*
# raise so callers can decide how to degrade.
try:  # pragma: no cover - dependency availability
    import fitz  # type: ignore
    _FITZ_IMPORT_ERROR = None  # type: ignore
except Exception as _fe:  # pragma: no cover
    # Some PyMuPDF builds require importing 'pymupdf' first to register the extension module
    try:  # pragma: no cover
        import pymupdf  # type: ignore
        import fitz  # type: ignore  # retry after registering
        _FITZ_IMPORT_ERROR = None  # type: ignore
    except Exception as _fe2:  # pragma: no cover
        fitz = None  # type: ignore
        _FITZ_IMPORT_ERROR = _fe2  # type: ignore

# Configuration import: attempt package-relative then absolute fallback.
try:
    from .config import PDF_DIR, EXTRACT_GAMES_VERBOSE  # type: ignore
except Exception:  # pragma: no cover
    try:
        from config import PDF_DIR, EXTRACT_GAMES_VERBOSE  # type: ignore
    except Exception:
        # Final fallback defaults if config cannot be imported; keeps module usable.
        PDF_DIR = "pdfs"  # type: ignore
        EXTRACT_GAMES_VERBOSE = False  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None


def extract_text_from_pdf(path: str) -> List[Dict]:
    if fitz is None:
        # Surface precise reason to caller so it can log informative message.
        err = _FITZ_IMPORT_ERROR or Exception("unknown import failure")
        raise RuntimeError(f"PyMuPDF import failed ({err.__class__.__name__}: {err})")
    doc = fitz.open(path)
    text = []
    for i, page in enumerate(doc):
        # Plain text (existing behavior)
        txt = page.get_text("text")
        entry: Dict = {"page": i + 1, "content": txt or ""}
        # Rich spans with font info (to detect bold mainline)
        try:
            d = page.get_text("dict")
            spans = []
            for block in d.get("blocks", []) or []:
                for line in block.get("lines", []) or []:
                    for sp in line.get("spans", []) or []:
                        s_txt = sp.get("text", "")
                        if not s_txt:
                            continue
                        font = (sp.get("font") or "").lower()
                        flags = int(sp.get("flags") or 0)
                        # Heuristic: font name contains 'bold' OR large weight bit in flags
                        is_bold = ("bold" in font) or bool(flags & 2**4)
                        spans.append({
                            "text": s_txt,
                            "bold": is_bold,
                            "font": sp.get("font"),
                            "size": sp.get("size")
                        })
            entry["spans"] = spans
            if EXTRACT_GAMES_VERBOSE:
                bold_count = sum(1 for s in spans if s.get("bold"))
                total = len(spans)
                print(f"[pdf] page {i+1}: spans={total}, bold={bold_count}")
        except Exception:
            entry["spans"] = []
        if (entry["content"] or entry["spans"]):
            text.append(entry)
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
