
import pandas as pd

# 20 orders, all with customer_id = 1
orders = pd.DataFrame([
    {"order_id": i, "customer_id": 1} for i in range(1, 21)
])

# 20 customers, all with id = 1
customers = pd.DataFrame([
    {"id": 1, "name": f"Customer_{i}"} for i in range(1, 21)
])

samples = {
    "orders": orders,
    "customers": customers
}
