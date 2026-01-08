SELECT *
FROM orders
JOIN customers
ON orders.amount = customers.id