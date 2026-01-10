from pathlib import Path
import json
from json import JSONDecodeError


def _safe_json_load(path: Path):
    """Load JSON from a file, supporting both standard JSON and NDJSON (one JSON object per line).

    Returns a list of dicts if multiple objects are present, or a dict/list from standard JSON.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except JSONDecodeError:
        # Fallback: stream-decode multiple concatenated JSON objects
        decoder = json.JSONDecoder()
        items = []
        s = text
        idx = 0
        length = len(s)
        while idx < length:
            # skip whitespace between objects
            while idx < length and s[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                obj, next_idx = decoder.raw_decode(s, idx)
                items.append(obj)
                idx = next_idx
            except JSONDecodeError:
                # Stop on first failure in streaming mode
                break
        if items:
            return items
        # No items parsed in fallback; re-raise the original error
        raise


def _load_documents() -> list[str]:
    # Resolve data paths relative to repository root (module -> rag/ -> project root)
    project_root = Path(__file__).resolve().parent.parent
    runbooks_path = project_root / "data" / "runbooks.json"
    history_path = project_root / "data" / "historical_incidents.json"

    runbooks = _safe_json_load(runbooks_path)
    history = _safe_json_load(history_path)

    # Normalize to lists
    if isinstance(runbooks, dict):
        runbooks = [runbooks]
    if isinstance(history, dict):
        history = [history]

    documents: list[str] = []
    for r in runbooks or []:
        # Be resilient to missing keys
        r_type = r.get("type", "Unknown")
        guidance = r.get("guidance", "")
        documents.append(f"Runbook {r_type}: {guidance}")
    for h in history or []:
        incident = h.get("incident", "Unknown")
        resolution = h.get("resolution", "")
        documents.append(f"Incident: {incident} Resolution: {resolution}")
    return documents


def build_vector_store():
    # Lazy imports to reduce module import-time coupling and version conflicts
    # Embeddings: prefer langchain_openai, fallback to legacy langchain
    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore
    except Exception:
        try:
            from langchain.embeddings import OpenAIEmbeddings  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import OpenAIEmbeddings from supported packages"
            ) from e

    # Vector store: prefer langchain_community, fallback to legacy langchain
    try:
        from langchain_community.vectorstores import FAISS  # type: ignore
    except Exception:
        try:
            from langchain.vectorstores import FAISS  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import FAISS vectorstore from supported packages"
            ) from e

    documents = _load_documents()
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(documents, embeddings)