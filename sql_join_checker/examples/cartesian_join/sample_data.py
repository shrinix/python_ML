import pandas as pd

samples = {
    "orders": pd.DataFrame({
        "order_id": [1, 2],
        "customer_id": [10, 20],
        "amount": [100.0, 200.0]
    }),
    "customers": pd.DataFrame({
        "id": [10, 20],
        "name": ["Alice", "Bob"]
    }),
    "payments": pd.DataFrame({
        "payment_id": [100, 101],
        "order_id": [1, 2],
        "amount": [100.0, 200.0],
        "payment_date": ["2026-01-01", "2026-01-02"]
    })
}