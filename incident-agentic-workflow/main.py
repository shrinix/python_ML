import os
from pathlib import Path

# Choose orchestrator: 'langgraph' or 'autogen'
orchestrator = os.getenv("ORCHESTRATOR", "langgraph").lower()

if orchestrator == "langgraph":
    from orchestration.langgraph_flow import build_langgraph
    app = build_langgraph()
    from domain.models import IncidentState
    result = app.invoke(IncidentState(description="Payments API slow after last deployment"))
    print(result)
elif orchestrator == "autogen":
    from orchestration.autogen_flow import run_autogen_pipeline
    incident = "Payments API slow after last deployment"
    result = run_autogen_pipeline(incident)
    print(result)
else:
    raise ValueError(f"Unknown ORCHESTRATOR '{orchestrator}' (expected 'langgraph' or 'autogen')")
