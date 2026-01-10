from rag.retriever import retrieve_context
from langchain_community.chat_models import ChatOpenAI
from agents.base import BaseAgent
from domain.models import IncidentState

llm = ChatOpenAI(temperature=0)


class ResolutionAgent(BaseAgent):
    def run(self, state: IncidentState) -> IncidentState:
        context = retrieve_context(state.description)
        prompt = f"""
        Context:\n{context}

        Given the diagnosis, propose a safe resolution.
        Diagnosis: {state.diagnosis}
        """
        state.resolution = llm.predict(prompt)
        # legacy alias
        state.recommendation = state.resolution
        return state


def resolve(text: str, diagnosis: str) -> str:
    agent = ResolutionAgent()
    state = IncidentState(description=text, diagnosis=diagnosis)
    out = agent.run(state)
    return out.resolution or ""