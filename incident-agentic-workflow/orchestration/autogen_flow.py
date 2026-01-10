"""AutoGen-based orchestration demo.

Supports both `autogen` and `pyautogen` distributions.
Install in your virtualenv: `pip install autogen` or `pip install pyautogen`.
 (no Docker).
"""

def _import_autogen():
    """Import AutoGen and return AssistantAgent, UserProxyAgent classes.

    Tries multiple known import layouts to handle version differences.
    """
    # Preferred: top-level `autogen`
    try:
        import autogen as ag
        AssistantAgent = getattr(ag, "AssistantAgent", None)
        UserProxyAgent = getattr(ag, "UserProxyAgent", None)
        if AssistantAgent and UserProxyAgent:
            return AssistantAgent, UserProxyAgent
    except Exception:
        pass

    # Fallback: submodule layout
    try:
        from autogen.agentchat import AssistantAgent, UserProxyAgent  # type: ignore
        return AssistantAgent, UserProxyAgent
    except Exception:
        pass

    # Alternate distribution name: `pyautogen`
    try:
        import pyautogen as ag
        AssistantAgent = getattr(ag, "AssistantAgent", None)
        UserProxyAgent = getattr(ag, "UserProxyAgent", None)
        if AssistantAgent and UserProxyAgent:
            return AssistantAgent, UserProxyAgent
    except Exception:
        pass

    raise ImportError(
        "AutoGen not found or incompatible. Install with `pip install autogen` or `pip install pyautogen` in your active venv."
    )


def build_autogen():
    AssistantAgent, UserProxyAgent = _import_autogen()

    def _llm_config_from_config():
        """Build AutoGen `llm_config`, preferring config.settings override, then env.

        Order of precedence:
        1) `config.settings.AUTOGEN_MODEL` if present
        2) `AUTOGEN_MODEL` environment variable
        3) hardcoded default "gpt-4o-mini"
        """
        import os
        try:
            from config import settings
            model_cfg = getattr(settings, "AUTOGEN_MODEL", None)
        except Exception:
            model_cfg = None

        model = model_cfg or os.getenv("AUTOGEN_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        # Build minimal OpenAI entry; avoid unsupported keys like 'provider'.
        openai_entry = {"model": model}
        if api_key:
            openai_entry["api_key"] = api_key
        cfg = {"config_list": [openai_entry], "temperature": 0.0}
        return cfg

    llm_config = _llm_config_from_config()

    classifier = AssistantAgent(
        name="Classifier",
        system_message="Classify the incident into PERFORMANCE, SECURITY, or CHANGE with a confidence score (0-1). Return ONLY JSON with keys 'type' and 'confidence'.",
        code_execution_config={"use_docker": False},
        llm_config=llm_config,
    )

    diagnoser = AssistantAgent(
        name="Diagnoser",
        system_message="Given incident text and type, provide a one-line diagnosis of likely root cause.",
        code_execution_config={"use_docker": False},
        llm_config=llm_config,
    )

    recommender = AssistantAgent(
        name="Recommender",
        system_message="Given diagnosis, propose a safe, reversible resolution step in one sentence.",
        code_execution_config={"use_docker": False},
        llm_config=llm_config,
    )

    # Return only assistants; orchestration will run them sequentially
    return classifier, diagnoser, recommender


def run_autogen_pipeline(incident_text: str):
    """Run a simple sequential pipeline across AutoGen assistants.

    Returns a dict with classification, diagnosis, and recommendation.
    """
    AssistantAgent, _ = _import_autogen()
    # Build agents with consistent llm_config
    classifier, diagnoser, recommender = build_autogen()

    def _to_text(resp) -> str:
        try:
            if isinstance(resp, dict):
                return resp.get("content") or resp.get("message") or str(resp)
            return getattr(resp, "content", None) or str(resp)
        except Exception:
            return str(resp)

    # 1) Classification
    cls_prompt = (
        f"Incident: {incident_text}\n"
        "Return ONLY JSON with keys 'type' (PERFORMANCE|SECURITY|CHANGE) and 'confidence' (0-1)."
    )
    cls_resp = classifier.generate_reply(messages=[{"role": "user", "content": cls_prompt}])
    classification_text = _to_text(cls_resp)

    # 2) Diagnosis
    diag_prompt = (
        f"Incident: {incident_text}\n"
        f"Classification result: {classification_text}\n"
        "Provide a one-line diagnosis of likely root cause."
    )
    diag_resp = diagnoser.generate_reply(messages=[{"role": "user", "content": diag_prompt}])
    diagnosis_text = _to_text(diag_resp)

    # 3) Recommendation
    rec_prompt = (
        f"Incident: {incident_text}\n"
        f"Diagnosis: {diagnosis_text}\n"
        "Propose a safe, reversible resolution step in one sentence."
    )
    rec_resp = recommender.generate_reply(messages=[{"role": "user", "content": rec_prompt}])
    recommendation_text = _to_text(rec_resp)

    return {
        "classification": classification_text,
        "diagnosis": diagnosis_text,
        "recommendation": recommendation_text,
    }