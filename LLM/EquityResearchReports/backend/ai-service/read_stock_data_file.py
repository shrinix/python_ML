#create a function to read a csv file and return a pandas dataframe
import pandas as pd
import os
from typing import List, Dict, Any
from datetime import datetime
from dateutil import parser



def read_stock_data_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file containing stock data and returns a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the required columns are present
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The column '{col}' is missing from the file.")
    
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    
    return df

if __name__ == "__main__":
    # Example usage
    file_path = "/Users/shriniwasiyengar/git/python_ML/LLM/EquityResearchReports/backend/ai-service/source-files/aapl.us.txt"
    try:
        stock_data = read_stock_data_file(file_path)
        print(stock_data.head())
        print(stock_data.info())
        print(stock_data.describe())

        #get the start and end date of the data
        start_date = stock_data.index.min()
        end_date = stock_data.index.max()
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")

        #download remaining data using yfiance
        data = yf.download('AAPL', start=end_date, end=end_date, progress=False, auto_adjust=False)

    except Exception as e:
        print(f"Error: {e}")