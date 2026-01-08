from sqlglot import parse_one
from sqlglot.expressions import Join

def suggest_fixes(findings, fks):
    """
    Suggest correct joins based on schema foreign keys.
    Looks at findings (problematic joins) and returns FK-based suggestions.
    """
    fixes = []

    for f in findings:
        if f["type"] in ["TYPE_MISMATCH", "NON_FK_JOIN", "ROW_EXPLOSION"]:
            join_text = f.get("join")
            if not join_text:
                continue

            left_table = join_text.split("=")[0].strip().split(".")[0]
            right_table = join_text.split("=")[1].strip().split(".")[0]

            for fk in fks:
                fk_tables = {fk["from_table"], fk["to_table"]}
                if {left_table, right_table} == fk_tables:
                    fixes.append(
                        f"{fk['from_table']}.{fk['from_column']} = {fk['to_table']}.{fk['to_column']}"
                    )

    # Remove duplicates
    return list(set(fixes))


def apply_suggested_fixes(sql, findings, suggested_fixes):
    """
    Rewrites SQL by replacing problematic joins with suggested FK joins.

    Args:
        sql (str): original SQL query
        findings (list): list of join findings with types and join text
        suggested_fixes (list): list of FK-based join strings like "orders.customer_id = customers.id"

    Returns:
        fixed_sql (str): rewritten SQL with suggested fixes applied
    """
    tree = parse_one(sql)
    joins = list(tree.find_all(Join))

    fix_idx = 0  # track which suggested fix to apply
    for join in joins:
        if not join.args.get("on"):
            # missing ON clause → can apply fix if available
            if fix_idx < len(suggested_fixes):
                fix_expr = suggested_fixes[fix_idx]
                join.set("on", parse_one(fix_expr))
                fix_idx += 1
        else:
            join_expr = join.args["on"]
            join_text = f"{join_expr.left} = {join_expr.right}" if hasattr(join_expr, "left") else str(join_expr)
            # check if this join was flagged
            for f in findings:
                if f.get("join") == join_text and fix_idx < len(suggested_fixes):
                    fix_expr = suggested_fixes[fix_idx]
                    join.set("on", parse_one(fix_expr))
                    fix_idx += 1

    return tree.sql()