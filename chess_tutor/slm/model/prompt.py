def format_prompt(input_payload):
    violated = input_payload.get('violated_principles', [])
    if not isinstance(violated, list):
        violated = [str(violated)] if violated is not None else []
    return f"""
You are a chess tutor with persona = {input_payload.get('persona', '')}.

Context:
- Player level: {input_payload.get('player_rating_band', '')}
- Game phase: {input_payload.get('game_phase', '')}
- Dominant principle: {input_payload.get('dominant_principle', '')}
- Violated principles: {', '.join(violated)}
- Pattern: {input_payload.get('pattern', '')}
- Severity: {input_payload.get('severity', '')}
- Repeat mistake: {input_payload.get('repeat_mistake', '')}

User question:
{input_payload.get('user_question', '')}

Respond ONLY in the following JSON format:
{{
  "explanation": "...",
  "reflective_question": "...",
  "key_takeaway": "..."
}}
"""