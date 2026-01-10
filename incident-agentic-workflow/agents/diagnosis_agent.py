from rag.retriever import retrieve_context
from langchain_community.chat_models import ChatOpenAI
from agents.base import BaseAgent
from domain.models import IncidentState

llm = ChatOpenAI(temperature=0)


class DiagnosisAgent(BaseAgent):
    def run(self, state: IncidentState) -> IncidentState:
        context = retrieve_context(state.description)
        itype = state.incident_type or "UNKNOWN"
        prompt = f"""
        Context:\n{context}

        Diagnose likely root cause for a {itype} incident.
        Incident: {state.description}
        """
        state.diagnosis = llm.predict(prompt)
        # legacy alias
        state.cause = state.diagnosis
        return state


def diagnose(text: str, incident_type: str) -> str:
    agent = DiagnosisAgent()
    state = IncidentState(description=text, incident_type=incident_type)
    out = agent.run(state)
    return out.diagnosis or ""