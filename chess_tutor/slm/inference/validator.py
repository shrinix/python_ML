import re
import json

FORBIDDEN_PATTERNS = [
    r"\bStockfish\b",
    r"\bbest move\b",
    r"\beval\b",
    r"\b[QRBN][a-h][1-8]\b"
]

def is_valid_output(text: str) -> bool:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    return parse_json(text) is not None

def parse_json(text):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None