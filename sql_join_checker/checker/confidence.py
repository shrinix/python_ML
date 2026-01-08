def aggregate_confidence(findings):
    if not findings:
        return 0.0
    return round(max(f["confidence"] for f in findings), 2)