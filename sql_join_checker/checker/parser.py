from sqlglot import parse_one
from sqlglot.expressions import Table, Join, Column

def extract_tables_and_joins(sql: str):
    """
    Returns:
        tables: set of table names
        joins: list of tuples (left_table, left_col, right_table, right_col)
    """
    tree = parse_one(sql)

    tables = set()
    joins = []

    # Map aliases to real table names
    alias_to_table = {}
    for t in tree.find_all(Table):
        if t.this:
            table_name = str(t.this)
            tables.add(table_name)
            if t.alias_or_name != t.this:
                alias_to_table[str(t.alias_or_name)] = table_name

    # collect joins
    for join in tree.find_all(Join):
        if not join.args.get("on"):
            continue

        on_expr = join.args["on"]

        cols = list(on_expr.find_all(Column))
        if len(cols) != 2:
            continue

        c1, c2 = cols
        # Resolve aliases to real table names, always as strings
        ltab = alias_to_table.get(str(c1.table), str(c1.table))
        rtab = alias_to_table.get(str(c2.table), str(c2.table))
        joins.append((ltab, c1.name, rtab, c2.name))

    print(f"[DEBUG] tables: {tables}")
    print(f"[DEBUG] joins: {joins}")
    return tables, joins