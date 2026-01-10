# agents/base.py
from abc import ABC, abstractmethod
from domain.models import IncidentState

class BaseAgent(ABC):
    @abstractmethod
    def run(self, state: IncidentState) -> IncidentState:
        pass