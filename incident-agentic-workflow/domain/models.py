# domain/models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class IncidentState:
    # Core description of the incident
    description: str

    # Classification
    incident_type: Optional[str] = None
    confidence: Optional[float] = None

    # Analysis / diagnosis
    diagnosis: Optional[str] = None
    impact: Optional[str] = None

    # Recommended / chosen resolution
    resolution: Optional[str] = None

    # Routing decision (e.g., AUTO/HUMAN)
    route: Optional[str] = None

    # Legacy aliases (kept for compatibility with earlier attempts)
    category: Optional[str] = None  # alias of incident_type
    cause: Optional[str] = None     # alias of diagnosis
    recommendation: Optional[str] = None  # alias of resolution