# FILE: data_processing.py

import pandas as pd
import numpy as np

# Sample data for demonstration
def get_data():
    data = pd.DataFrame({
        'Date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        'Value1': np.random.rand(100),
        'Value2': np.random.rand(100),
        'Value3': np.random.rand(100),
        'Value4': np.random.rand(100)
    })
    return data