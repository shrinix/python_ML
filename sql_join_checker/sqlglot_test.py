from sqlglot import parse_one
from sqlglot.expressions import Join

tree = parse_one("""
SELECT *
FROM orders
JOIN customers
ON orders.amount = customers.id
""")

joins = list(tree.find_all(Join))
print(len(joins))  # MUST print: 1