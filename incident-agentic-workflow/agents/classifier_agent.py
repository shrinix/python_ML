import json
import re
from rag.retriever import retrieve_context
from langchain_community.chat_models import ChatOpenAI
from agents.base import BaseAgent
from domain.models import IncidentState

llm = ChatOpenAI(temperature=0)


def _parse_classification(text: str) -> dict:
    """Parse LLM output into a dict with keys: type (str), confidence (float)."""
    cleaned = text.strip()
    # Remove code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", cleaned, flags=re.MULTILINE)
    # Try direct JSON
    try:
        obj = json.loads(cleaned)
        return {
            "type": str(obj.get("type", "UNKNOWN")).upper(),
            "confidence": float(obj.get("confidence", 0.0)),
        }
    except Exception:
        pass
    # Try to extract first JSON object block
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            obj = json.loads(m.group(0))
            return {
                "type": str(obj.get("type", "UNKNOWN")).upper(),
                "confidence": float(obj.get("confidence", 0.0)),
            }
        except Exception:
            pass
    # Heuristic fallback
    categories = ["PERFORMANCE", "SECURITY", "CHANGE"]
    upper = cleaned.upper()
    detected = next((c for c in categories if c in upper), "UNKNOWN")
    conf_match = re.search(r"(confidence|probability)[:\s-]*([01]?\.?\d+)", upper)
    confidence = float(conf_match.group(2)) if conf_match else 0.0
    return {"type": detected, "confidence": confidence}


class ClassifierAgent(BaseAgent):
    def run(self, state: IncidentState) -> IncidentState:
        context = retrieve_context(state.description)
        prompt = f"""
        Context:\n{context}

        Classify the incident into one of: PERFORMANCE, SECURITY, CHANGE.
        Return ONLY a JSON object with keys: "type" (one of PERFORMANCE|SECURITY|CHANGE) and "confidence" (0.0-1.0).

        Incident: {state.description}
        """
        result = llm.predict(prompt)
        parsed = _parse_classification(result)
        state.incident_type = parsed.get("type")
        state.confidence = parsed.get("confidence")
        # keep legacy alias updated
        state.category = state.incident_type
        return state


# Backward-compatible functional wrapper used by current orchestration
def classify_incident(incident_text: str) -> dict:
    agent = ClassifierAgent()
    state = IncidentState(description=incident_text)
    out = agent.run(state)
    return {"type": out.incident_type, "confidence": out.confidence}