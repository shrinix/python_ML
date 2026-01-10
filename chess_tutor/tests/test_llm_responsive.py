"""Minimal test to ensure the RAG LLM path is exercised when ENABLE_LLM is true.

We avoid hitting a real provider by monkeypatching LLMClient with a mock object.
The test asserts:
 - ExtendedRAGService picks up ENABLE_LLM from env
 - _llm is initialized and enabled
 - synthesize_answer returns the mock generation string instead of extractive fallback
"""
import os
import sys
import pathlib
import importlib
import numpy as np
import types

import pytest

# Ensure backend package importable regardless of pytest cwd
HERE = pathlib.Path(__file__).resolve()
PKG_DIR = HERE.parents[1]  # .../chess_tutor
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# Ensure flag true for this test
os.environ["ENABLE_LLM"] = "true"

# Import rag_service AFTER setting env so config sees it
after_reload_modules = []

@pytest.fixture
def mock_llm(monkeypatch):
    class MockLLM:
        def __init__(self):
            self.enabled = True
            self.provider = "mock"
            self.model = "mock-model"
        def generate(self, system, user, temperature=0.2, max_tokens=None):
            return "FEN: mock-fen\nThis is a mock LLM answer about isolated pawns."
    # Patch backend.llm_adapter.LLMClient to return mock
    import backend.llm_adapter as llm_adapter
    monkeypatch.setattr(llm_adapter, "LLMClient", lambda: MockLLM())
    return MockLLM()

class StubModel:
    def encode(self, texts, show_progress_bar=False):
        # Return deterministic zeros with expected shape (384 chosen to look like MiniLM)
        return np.zeros((len(texts), 384), dtype="float32")

class StubTutor:
    def __init__(self):
        self.model = StubModel()
    def retrieve(self, query, top_k=3):
        return [("Sample PDF chunk describing isolated pawn structures.", {"source": "stub"})]
    def retrieve_game_segments(self, query, top_k=3):
        return []


def test_llm_responsive(mock_llm, monkeypatch):
    # Reload rag_service to pick up ENABLE_LLM change
    import backend.rag_service as rag_service
    importlib.reload(rag_service)
    svc = rag_service.ExtendedRAGService(StubTutor())
    assert svc._llm is not None and getattr(svc._llm, "enabled", False), "LLM should be enabled"
    # Empty buckets triggers fallback path unless LLM generates
    buckets = {"position": [], "principles": [], "docs": [], "games": []}
    answer, sources = svc.synthesize_answer("What is an isolated pawn?", buckets)
    assert "mock llm answer" in answer.lower(), f"Expected mock generation, got: {answer}"
    # Ensure sources list returned (even if empty)
    assert isinstance(sources, list)

