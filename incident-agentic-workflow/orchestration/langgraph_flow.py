# orchestration/langgraph_flow.py
from langgraph.graph import StateGraph
from domain.models import IncidentState
from agents.classifier_agent import ClassifierAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.resolution_agent import ResolutionAgent

classifier = ClassifierAgent()
diagnoser = DiagnosisAgent()
resolver = ResolutionAgent()

def build_langgraph():
    graph = StateGraph(IncidentState)

    graph.add_node("classify", classifier.run)
    graph.add_node("diagnose", diagnoser.run)
    graph.add_node("resolve", resolver.run)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "diagnose")
    graph.add_edge("diagnose", "resolve")

    return graph.compile()