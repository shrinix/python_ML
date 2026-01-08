SELECT *
FROM orders o
JOIN customers c
  ON o.amount = c.id              -- WRONG: amount vs id
JOIN payments p
  ON p.amount = o.customer_id     -- WRONG: amount vs customer_id
JOIN products pr
  ON pr.price = p.payment_id      -- WRONG: price vs payment_id