#!/usr/bin/env python3
"""Build and persist extended principle/document embeddings for chess_tutor.

This script scans registry.json (principles) and any text chunks from existing
index_store doc/game indices (if available) and produces a consolidated
"extended_principles.npz" file containing:
  - embeddings: float32 matrix (N x D)
  - texts: list[str] parallel to embeddings
It uses the same sentence-transformer model defined in config (all-MiniLM-L6-v2).
If FAISS is installed, also writes a FAISS index file for faster similarity search.

Run:
    python scripts/build_principle_index.py --output-dir index_store/

Options:
    --model-name  Override embedding model name
    --registry    Path to registry.json (default backend/registry.json)

The backend extended RAG service will prefer this pre-built index if present.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_texts_from_registry(registry_path: Path) -> list[str]:
    if not registry_path.exists():
        print(f"[warn] registry.json not found at {registry_path}")
        return []
    data = json.loads(registry_path.read_text())
    texts = []
    for item in data:
        desc = item.get("description") or ""
        name = item.get("name") or ""
        if desc:
            texts.append(f"Principle: {name}\n{desc}")
    return texts


def load_existing_extended_npz(output_dir: Path) -> list[str]:
    # Attempt to gather existing doc/game chunk texts if they are serialized separately
    # Placeholder: extend later if chunk text persistence is added
    return []


def build_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    if not texts:
        raise SystemExit("No texts found to embed; aborting.")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise SystemExit("sentence-transformers not installed. Install and retry.")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return emb.astype("float32")


def persist_npz(embeddings: np.ndarray, texts: list[str], path: Path):
    np.savez_compressed(path, embeddings=embeddings, texts=np.array(texts, dtype=object))
    print(f"[ok] wrote NPZ index: {path}")


def persist_faiss(embeddings: np.ndarray, path: Path):
    if not HAVE_FAISS:
        print("[info] faiss not available; skipping faiss index")
        return
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(path))
    print(f"[ok] wrote FAISS index: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="index_store", help="Directory to write index files")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--registry", default="backend/registry.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_path = Path(args.registry)
    texts = load_texts_from_registry(registry_path)
    texts += load_existing_extended_npz(out_dir)

    if not texts:
        print("[warn] No texts discovered; nothing to build.")
        return

    embeddings = build_embeddings(texts, args.model_name)

    npz_path = out_dir / "extended_principles.npz"
    persist_npz(embeddings, texts, npz_path)

    faiss_path = out_dir / "extended_principles.faiss"
    persist_faiss(embeddings, faiss_path)

    print("[done] Principle index build complete.")


if __name__ == "__main__":
    main()
