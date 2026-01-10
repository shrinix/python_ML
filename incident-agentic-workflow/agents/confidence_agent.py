from agents.base import BaseAgent
from domain.models import IncidentState


def allow_automation(confidence: float) -> bool:
    return confidence >= 0.8


class ConfidenceAgent(BaseAgent):
    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def run(self, state: IncidentState) -> IncidentState:
        conf = state.confidence or 0.0
        state.route = "AUTO" if conf >= self.threshold else "HUMAN"
        return state