from rag.vector_store import build_vector_store

_VECTOR_STORE = None


def _get_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = build_vector_store()
    return _VECTOR_STORE


def retrieve_context(query: str) -> str:
    docs = _get_vector_store().similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])