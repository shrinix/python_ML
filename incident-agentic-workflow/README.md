# Incident Agentic Workflow

Unified agent implementations with pluggable orchestration (LangGraph or AutoGen).

## Overview
- Agents implement a common `BaseAgent.run(state: IncidentState) -> IncidentState` interface.
- Unified LLM-backed agents live in `agents/*_agent.py`:
  - `ClassifierAgent`, `DiagnosisAgent`, `ImpactAgent`, `ResolutionAgent`, `ConfidenceAgent`.
- Orchestration:
  - LangGraph flow: `orchestration/langgraph_flow.py` (class-based agents).
  - AutoGen demo: `orchestration/autogen_flow.py`.
- Domain state: `domain/models.py` holds a superset of fields used across flows.

## Switch Orchestrator
Set the environment variable `ORCHESTRATOR` to choose the orchestrator:

```bash
# In your virtualenv
export ORCHESTRATOR=langgraph  # or autogen
python incident-agentic-workflow/main.py
```

Defaults to `langgraph` when not set.

## Dependencies
Install required packages:

```bash
source /Users/shriniwasiyengar/git/python_ML/.venv/bin/activate
pip install -r incident-agentic-workflow/requirements.txt
```

Includes `langgraph`, `langchain-community`, `langchain-openai`, `faiss-cpu`, `pyautogen`.

## Notes
- Older agent files like `agents/classifier.py`, `agents/diagnosis.py`, and `agents/recommendation.py` have been replaced by the unified `*_agent.py` modules.
- `rag/vector_store.py` uses `langchain_openai.OpenAIEmbeddings` and `langchain_community.vectorstores.FAISS`.
- Ensure `OPENAI_API_KEY` is set for LLM/embeddings.
