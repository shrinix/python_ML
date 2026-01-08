SELECT *
FROM orders
JOIN customers ON orders.amount = customers.id
JOIN payments ON payments.order_id = orders.order_id