import pandas as pd

samples = {
    "orders": pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [10, 20, 10],
        "amount": [100.0, 200.0, 150.0]
    }),
    "customers": pd.DataFrame({
        "id": [10, 20],
        "name": ["Alice", "Bob"]
    })
}