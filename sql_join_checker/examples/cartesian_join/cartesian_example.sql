SELECT *
FROM orders o
JOIN customers c      -- missing ON clause → Cartesian product
JOIN payments p       -- missing ON clause → multiplies rows again