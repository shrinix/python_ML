def analyze_samples(joins, samples):
    if not samples:
        return []

    findings = []
    intermediate = None
    base = None

    for idx, (ltab, lcol, rtab, rcol) in enumerate(joins):
        if ltab not in samples or rtab not in samples:
            print(f"[DEBUG] Missing sample for: {ltab} or {rtab}")
            continue

        left = samples[ltab] if intermediate is None else intermediate
        right = samples[rtab]

        print(f"[DEBUG] Join step {idx+1}: {ltab}.{lcol} x {rtab}.{rcol}")
        print(f"[DEBUG] Left shape: {left.shape}, Right shape: {right.shape}")

        joined = left.merge(
            right,
            left_on=lcol,
            right_on=rcol,
            how="inner"
        )

        print(f"[DEBUG] Joined shape: {joined.shape}")

        if base is None:
            base = max(len(left), len(right))

        growth = len(joined) / base if base else 0
        print(f"[DEBUG] Growth factor: {growth}")

        if growth > 10:
            findings.append({
                "type": "ROW_EXPLOSION",
                "join_step": idx + 1,
                "growth_factor": round(growth, 2),
                "confidence": min(1.0, 0.7 + growth / 50)
            })

        intermediate = joined

    return findings