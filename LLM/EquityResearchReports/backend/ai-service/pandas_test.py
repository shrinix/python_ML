import pandas as pd
import numpy as np

def calc_bollinger_bands(df, window_size=20):
    """
    Calculate Bollinger Bands for a given DataFrame of closing prices.

    Parameters:
        df (pd.DataFrame): DataFrame containing closing prices.
        window_size (int): Window size for rolling calculations.

    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands added.
    """
    df['period_mean'] = df['close'].mean()
    df['abs_^2_var'] = (np.abs(df['close'] - df['period_mean'])) ** 2
    df['stdev_man'] = np.sqrt(df['abs_^2_var'].rolling(20).mean())
    df['stdev_roll'] = df['close'].rolling(20).std()

    df['BolU_man'] = round(df['period_mean'] + (df['stdev_man'] * 2), 4)
    df['BolL_man'] = round(df['period_mean'] - (df['stdev_man'] * 2), 4)

    df['BolU_roll'] = round(
        df['close'].rolling(20).mean() + (2 * (df['close'].rolling(20).std(ddof=0))), 4)
    df['BolL_roll'] = round(
        df['close'].rolling(20).mean() - (2 * (df['close'].rolling(20).std(ddof=0))), 4)

    print(df)
    return df

if __name__ == "__main__":

    # close_values = {'close': [88.64, 87.02, 88.7, 89.89, 90.93, 90.4, 90.21, 90.16, 90.94, 91.93, 
    #                           90.26, 85.5, 81.89, 82.18, 78.31, 75.65, 79.61, 76.55, 80.41, 77.87]}

    #1st 20 day window from: 2/27/25 to 3/26/25
    close_values = {'close': [237.300003, 241.839996, 238.029999, 235.929993, 235.740005, 235.330002, 239.070007,
    227.479996,  220.839996, 216.979996, 209.679993, 213.490005, 214, 212.690002, 215.240005, 214.100006,
    218.270004, 220.729996, 223.75, 221.529999]}

    df = pd.DataFrame(close_values)

    df = calc_bollinger_bands(df)
    print(df)

    #2nd 20 day window from: 2/26/25 to 3/25/25
    close_values = {'close': [240.360001, 237.300003, 241.839996, 238.029999, 235.929993, 235.740005, 235.330002, 239.070007,
    227.479996,  220.839996, 216.979996, 209.679993, 213.490005, 214, 212.690002, 215.240005, 214.100006,
    218.270004, 220.729996, 223.75]}

    df = pd.DataFrame(close_values)

    df = calc_bollinger_bands(df)
