from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from agents.classifier_agent import classify_incident
from agents.confidence_agent import allow_automation
from agents.diagnosis_agent import diagnose
from agents.impact_agent import assess_impact
from agents.resolution_agent import resolve

class IncidentState(TypedDict):
    incident_text: str
    incident_type: str
    confidence: float
    diagnosis: str
    impact: str
    resolution: str
    route: Literal["AUTO", "HUMAN"]


def classify_node(state):
    result = classify_incident(state["incident_text"])
    state["incident_type"] = result["type"]
    state["confidence"] = result["confidence"]
    return state


def route_node(state):
    state["route"] = "AUTO" if allow_automation(state["confidence"]) else "HUMAN"
    return state


def diagnosis_node(state):
    state["diagnosis"] = diagnose(state["incident_text"], state["incident_type"])
    return state


def impact_node(state):
    state["impact"] = assess_impact(state["incident_text"])
    return state


def resolution_node(state):
    state["resolution"] = resolve(state["incident_text"], state["diagnosis"])
    return state


def build_graph():
    g = StateGraph(IncidentState)

    g.add_node("classify", classify_node)
    g.add_node("route", route_node)
    g.add_node("diagnose", diagnosis_node)
    g.add_node("impact", impact_node)
    g.add_node("resolve", resolution_node)

    g.set_entry_point("classify")
    g.add_edge("classify", "route")

    g.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {
            "AUTO": "diagnose",
            "HUMAN": END,  # terminate early when human handoff is required
        },
    )

    g.add_edge("diagnose", "impact")
    g.add_edge("impact", "resolve")
    g.set_finish_point("resolve")

    return g.compile()