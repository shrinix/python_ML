#!/usr/bin/env bash
# Helper script to start backend with correct venv to avoid Anaconda/base interpreter issues.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/backend"
if [ ! -d "../.venv" ]; then
  echo "[run-backend] Missing .venv. Create with: python3 -m venv ../.venv && source ../.venv/bin/activate && pip install -r ../requirements.txt" >&2
  exit 1
fi
source ../.venv/bin/activate
echo "[run-backend] Using Python: $(which python)"
python -c "import pymupdf, fitz; print('[run-backend] PyMuPDF OK, fitz:', getattr(fitz,'__file__','<no file>'))" || {
  echo "[run-backend] PyMuPDF import failed. Attempting reinstall..." >&2
  pip install --upgrade --no-cache-dir --force-reinstall pymupdf || {
    echo "[run-backend] Reinstall failed." >&2; exit 2; }
  python -c "import pymupdf, fitz; print('[run-backend] After reinstall fitz:', getattr(fitz,'__file__','<no file>'))" || {
    echo "[run-backend] Still failing to import fitz. Check architecture (Python $(python -V))" >&2; exit 3; }
}
exec python -m uvicorn app:app --reload --port 8000
