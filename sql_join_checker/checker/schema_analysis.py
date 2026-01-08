from checker.schema import is_foreign_key
from checker.graph import build_join_graph, find_connected_components

def analyze_schema(tables, joins, schema_tables, fks):
    findings = []

    graph = build_join_graph(tables, joins)
    components = find_connected_components(graph)

    if len(components) > 1:
        findings.append({
            "type": "DISCONNECTED_JOIN_GRAPH",
            "components": [list(c) for c in components],
            "confidence": 0.95
        })

    for ltab, lcol, rtab, rcol in joins:
        ltype = schema_tables[ltab]["columns"].get(lcol)
        rtype = schema_tables[rtab]["columns"].get(rcol)

        if ltype != rtype:
            findings.append({
                "type": "TYPE_MISMATCH",
                "join": f"{ltab}.{lcol} = {rtab}.{rcol}",
                "confidence": 0.9
            })

        elif not is_foreign_key((ltab, lcol), (rtab, rcol), fks):
            findings.append({
                "type": "NON_FK_JOIN",
                "join": f"{ltab}.{lcol} = {rtab}.{rcol}",
                "confidence": 0.6
            })

    return findings