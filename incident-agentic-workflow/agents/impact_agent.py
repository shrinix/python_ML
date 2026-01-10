from langchain_community.chat_models import ChatOpenAI
from agents.base import BaseAgent
from domain.models import IncidentState

llm = ChatOpenAI(temperature=0)


class ImpactAgent(BaseAgent):
    def run(self, state: IncidentState) -> IncidentState:
        prompt = f"""
        Assess business impact (LOW, MEDIUM, HIGH).
        Incident: {state.description}
        """
        state.impact = llm.predict(prompt)
        return state


def assess_impact(text: str) -> str:
    agent = ImpactAgent()
    state = IncidentState(description=text)
    out = agent.run(state)
    return out.impact or ""