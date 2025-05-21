from datetime import datetime
import pandas as pd
import yfinance as yf
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import (
	get_max_version, get_timeframes_df, load_configuration_params, prepare_data_for_training,
	fetch_data_into_files, load_and_transform_data_by_ticker, add_engineered_features,
	calc_incremental_nday_average, calc_incremental_rsi, calc_incremental_macd,
	calc_incremental_bollinger_bands, plot_averages, plot_bollinger_bands
)

# column_name=7_day_avg or 30_day_avg
# target_column_name = 'Adj Close'
def test_incremental_nday_average(n, stock_data, column_name, target_column_name):

	print(f"Calculating and testing {n}-day moving average incrementally:")

	# Initialize the DataFrame with the required number of rows (e.g., 7 for 7-day moving average)
	new_tech_indicators = pd.DataFrame(index=range(n), columns=['Date', target_column_name, column_name])

	new_tech_indicators[column_name] = pd.Series(dtype='float64') 

	for i in range(n):
		# print(f"i: {i}")
		#copy date from stock_data
		new_tech_indicators.loc[i, 'Date'] = stock_data.index[-n + i]  # Use the index if 'Date' column is missing
		# Calculate the n-day moving average
		if i==0:
			prev_avg = stock_data[column_name].iloc[-n]
			new_value = stock_data[target_column_name].iloc[-n]
			oldest_value = stock_data[target_column_name].iloc[-n]
			updated_value = calc_incremental_nday_average(prev_avg, new_value, oldest_value, n)
		else:
			prev_avg = new_tech_indicators[column_name].iloc[i - 1]
			new_value = stock_data[target_column_name].iloc[-n + i]
			oldest_index = -2*n + i
			oldest_value = stock_data[target_column_name].iloc[oldest_index]
			print(f"prev_avg: {prev_avg}, new_value: {new_value}, oldest_value: {oldest_value}")
			updated_value = calc_incremental_nday_average(prev_avg, new_value, oldest_value, n)
	
		# Update the DataFrame
		print(f"updated_value for {column_name}: {updated_value}")
		new_tech_indicators.loc[i, column_name] = updated_value
		new_tech_indicators.loc[i, target_column_name] = new_value

	print("New technical indicators:")
	print(new_tech_indicators.tail(30)) #print the values of the last few features
	print("Stock data:")
	print(stock_data[['Adj Close', column_name]].tail(7))
	#compare the values in the new features with the stock_data
	print(f"Comparing the incrementally calculated {n}-day moving average values with values from stock_data:")
	for i in range(n):
		print(f"i: {i}")
		# print(f"new_tech_indicators[{column_name}].iloc[{i}]: {new_tech_indicators[column_name].iloc[i]}, stock_data[{column_name}].iloc[{-n + i}]: {stock_data[column_name].iloc[-n + i]}")
		tolerance = 1e-1  # Define a tolerance threshold
		try:
			assert abs(new_tech_indicators[column_name].iloc[i] - stock_data[column_name].iloc[-n + i]) <= tolerance
		except AssertionError:
			print(f"Assertion failed on date: {new_tech_indicators['Date'].iloc[i]}")
			print(f"new_tech_indicators[{column_name}].iloc[{i}]: {new_tech_indicators[column_name].iloc[i]}, stock_data[{column_name}].iloc[{-n + i}]: {stock_data[column_name].iloc[-n + i]}")
			raise
		assert abs(new_tech_indicators[column_name].iloc[i] - stock_data[column_name].iloc[-n + i]) <= tolerance

def test_incremental_macd(stock_data, target_column_name, column_name, start_date, end_date, n_short=12, n_long=26, n_signal=9):
	"""
	Incrementally update MACD and Signal Line using previous EMA values and current price.

	Parameters:
		prev_ema_short (float): Previous short-term EMA.
		prev_ema_long (float): Previous long-term EMA.
		prev_signal_line (float): Previous Signal Line value.
		curr_price (float): Current day's price.
		n_short (int): Period for short-term EMA (default is 12).
		n_long (int): Period for long-term EMA (default is 26).
		n_signal (int): Period for Signal Line EMA (default is 9).

	Returns:
		tuple: (ema_short, ema_long, macd, signal_line)
	"""

	# Calculate and test MACD incrementally
	print("Calculating MACD incrementally:")
	z = len(stock_data)
	n = 26
	end_date = stock_data.index[z-1]
	start_date = stock_data.index[z-n-1]

	# set n to the number of days between start_date and end_date
	print(f"start_date: {start_date}, end_date: {end_date}")
	n = (end_date - start_date).days+1
	print(f"n: {n}")

	  # Initialize the DataFrame with the required number of rows (e.g., 26 for 26-day MACD)
	new_tech_indicators = pd.DataFrame(index=range(n), columns=['Date', target_column_name, 'ema_short', 'ema_long', 'Sig_L', column_name])

	#add a date column to new_tech_indicators
	#format date as yyyy-mm-dd
	new_tech_indicators['Date'] = pd.Series(dtype='datetime64[ns]')
	#format date as yyyy-mm-dd
	new_tech_indicators['Date'] = pd.to_datetime(new_tech_indicators['Date']).dt.strftime('%Y-%m-%d')
	new_tech_indicators[target_column_name] = pd.Series(dtype='float64')
	#add columns ema_short, ema_long, signal_line, macd to new_tech_indicators dataframe
	#ema stands for exponential moving average
	new_tech_indicators['ema_short'] = pd.Series(dtype='float64')
	new_tech_indicators['ema_long'] = pd.Series(dtype='float64')
	new_tech_indicators['Sig_L'] = pd.Series(dtype='float64')
	new_tech_indicators[column_name] = pd.Series(dtype='float64') #column_name is MACD
	
	#get all the valid dates in stock_data by checking if the date is between start_date and end_date
	valid_dates = stock_data.index[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
	
	for i, curr_date in enumerate(valid_dates):
		date_index = stock_data.index.get_loc(curr_date)
		prev_ema_short = stock_data['ema_12'].iloc[date_index-1]
		prev_ema_long = stock_data['ema_26'].iloc[date_index-1]
		prev_signal_line = stock_data['Sig_L'].iloc[date_index-1]
		curr_price = stock_data['Adj Close'].iloc[date_index]
		print(f"i: {i}, curr_date: {curr_date}")
		print(f"prev_ema_short: {prev_ema_short}, prev_ema_long: {prev_ema_long}, prev_signal_line: {prev_signal_line}")
		print(f"curr_price: {curr_price}")

		ema_short, ema_long, macd, signal_line = calc_incremental_macd(curr_price, prev_ema_short, prev_ema_long, prev_signal_line, n_short, n_long, n_signal)

		# Update the DataFrame
		new_tech_indicators.at[new_tech_indicators.index[i], 'Date'] = curr_date
		new_tech_indicators.at[new_tech_indicators.index[i], column_name] = macd
		new_tech_indicators.at[new_tech_indicators.index[i], target_column_name] = stock_data[target_column_name].iloc[date_index]
		new_tech_indicators.at[new_tech_indicators.index[i], 'ema_short'] = ema_short
		new_tech_indicators.at[new_tech_indicators.index[i], 'ema_long'] = ema_long
		new_tech_indicators.at[new_tech_indicators.index[i], 'Sig_L'] = signal_line

	# remove the rows with NaN values. NaN values maybe present on dates that are weekends or holidays when there is no
	# market data available as the stock market is closed
	new_tech_indicators.dropna(inplace=True)

	#set the date column as the index
	new_tech_indicators.set_index('Date', inplace=True)

	# compare backwards from  the last value in the new_tech_indicators
	print("Comparing the incrementally calculated MACD values with values from stock_data:")
	for i in range(n):
		comparison_date = new_tech_indicators.index[len(new_tech_indicators)-1-i]
		print(f"i: {i}, date: {comparison_date.strftime('%Y-%m-%d')}")
		#get the index of the date in stock_data and new_tech_indicators
		stock_data_index = stock_data.index.get_loc(comparison_date)
		new_tech_indicators_index = new_tech_indicators.index.get_loc(comparison_date)
		# Print the values of the new_tech_indicators and stock_data for debugging
		print(f"new_tech_indicators['MACD'].iloc[{new_tech_indicators_index}]: {new_tech_indicators['MACD'].iloc[new_tech_indicators_index]}")
		print(f"stock_data['MACD'].iloc[{stock_data_index}]: {stock_data['MACD'].iloc[stock_data_index]}")
		# Assert that the values are equal upto d decimal places
		d = 6
		assert round(new_tech_indicators['MACD'].iloc[new_tech_indicators_index], d) == round(stock_data['MACD'].iloc[stock_data_index], d)

#stock_data already contains the pre-calculated  delta, avg_gain and avg_loss values for all the days in stock_data
#this function calculates rsi for the last 14 days in stock_data using update_rsi function
def test_incremental_rsi(stock_data, target_column, column_name, start_date, end_date):

	print("Calculating RSI incrementally:")
	z = len(stock_data)
	n = 14
	end_date = stock_data.index[z-1]
	start_date = stock_data.index[z-n-1]

	# Initialize the DataFrame with the required number of rows (e.g., 14 for 14-day RSI)
	# set n to the number of days between start_date and end_date
	print(f"start_date: {start_date}, end_date: {end_date}")
	n = (end_date - start_date).days+1
	print(f"n: {n}")

	# Initialize the DataFrame with the required number of rows (e.g., 14 for 14-day RSI)
	new_tech_indicators = pd.DataFrame(index=range(n), columns=['Date', column_name, 'delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'])

	#add a date column to new_tech_indicators
	#format date as yyyy-mm-dd
	new_tech_indicators['Date'] = pd.Series(dtype='datetime64[ns]')
	#format date as yyyy-mm-dd
	new_tech_indicators['Date'] = pd.to_datetime(new_tech_indicators['Date']).dt.strftime('%Y-%m-%d')
	new_tech_indicators[target_column] = pd.Series(dtype='float64')
	#add columns delta, gain, loss, avg_gain, avg_loss, rs, rsi to new_tech_indicators dataframe
	new_tech_indicators['delta'] = pd.Series(dtype='float64')
	new_tech_indicators['gain'] = pd.Series(dtype='float64')
	new_tech_indicators['loss'] = pd.Series(dtype='float64')
	new_tech_indicators['avg_gain'] = pd.Series(dtype='float64')
	new_tech_indicators['avg_loss'] = pd.Series(dtype='float64')
	new_tech_indicators['rs'] = pd.Series(dtype='float64')
	new_tech_indicators[column_name] = pd.Series(dtype='float64')
	
	#get all the valid dates in stock_data by checking if the date is between start_date and end_date
	valid_dates = stock_data.index[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
	
	for i, curr_date in enumerate(valid_dates):
		print(f"i: {i}, curr_date: {curr_date}")
		new_tech_indicators.at[new_tech_indicators.index[i], 'Date'] = curr_date
		date_index = stock_data.index.get_loc(curr_date)
		# get the oldest negative delta value from the last 14 days so that it can be removed from the rolling average loss
		# and make way for the new delta value to calculate the new average loss
		idx = date_index - 14 
		if stock_data['delta'].iloc[date_index - 14] < 0:
			oldest_negative_delta = stock_data['delta'].iloc[idx]
		else:
			oldest_negative_delta = 0
		oldest_negative_delta_date = stock_data.index[idx]
		
		#get prev_avg_loss from the row corresponding to start date -1
		#get index of the row corresponding to start date
		prev_avg_loss = -stock_data['avg_loss'].iloc[date_index-1]
		prev_avg_loss_date = stock_data.index[date_index-1]
		
		print(f"oldest_negative_delta_date: {oldest_negative_delta_date}, oldest_negative_delta: {oldest_negative_delta}")
		print(f"prev_avg_loss_date: {prev_avg_loss_date}, prev_avg_loss: {prev_avg_loss}")

		# curr_date = stock_data.index[z-1]
		curr_delta = stock_data['delta'].iloc[date_index]

		# #for avg_loss, we only take into account the negative delta values and ignore the positive delta values
		# if curr_delta > 0: 
		#     curr_delta = 0

		# new_avg_loss = -((prev_avg_loss * 14) - oldest_negative_delta + curr_delta)/14
		# print(f"curr_date: {curr_date}, new_avg_loss: {new_avg_loss}, curr_delta: {curr_delta}")

		# get the oldest positive delta value from the last 14 days so that it can be removed from the rolling average gain
		# and make way for the new delta value to calculate the new average gain
		if stock_data['delta'].iloc[date_index - 14] > 0:
			oldest_positive_delta = stock_data['delta'].iloc[idx]
		else:
			oldest_positive_delta = 0
		oldest_positive_delta_date = stock_data.index[idx]
		
		#get prev_avg_gain from the row corresponding to start date -1
		#get index of the row corresponding to start date
		prev_avg_gain = stock_data['avg_gain'].iloc[date_index-1]
		prev_avg_gain_date = stock_data.index[date_index-1]
		
		print(f"oldest_positive_delta_date: {oldest_positive_delta_date}, oldest_positive_delta: {oldest_positive_delta}")
		print(f"prev_avg_gain_date: {prev_avg_gain_date}, prev_avg_gain: {prev_avg_gain}")

		# # curr_date = stock_data.index[z-1]
		# curr_delta = stock_data['delta'].iloc[date_index]
		# #for avg_gain, we only take into account the positive delta values and ignore the negative delta values
		# if curr_delta < 0: 
		#     curr_delta = 0

		# new_avg_gain = ((prev_avg_gain * 14) - oldest_positive_delta + curr_delta)/14
		# print(f"curr_date: {curr_date}, new_avg_gain: {new_avg_gain}, curr_delta: {curr_delta}")
		
		# Calculate RS
		# rs = new_avg_gain / new_avg_loss if new_avg_loss != 0 else 0

		# Calculate RSI
		# rsi = 100 - (100 / (1 + rs)) if rs != 0 else 0

		new_avg_loss, new_avg_gain, rs, rsi = calc_incremental_rsi(oldest_negative_delta, oldest_positive_delta, prev_avg_loss, prev_avg_gain, curr_delta)
		
		new_tech_indicators.at[new_tech_indicators.index[i], 'avg_loss'] = new_avg_loss
		new_tech_indicators.at[new_tech_indicators.index[i], 'avg_gain'] = new_avg_gain
		new_tech_indicators.at[new_tech_indicators.index[i], 'rs'] = rs
		new_tech_indicators.at[new_tech_indicators.index[i], column_name] = rsi
		new_tech_indicators.at[new_tech_indicators.index[i], target_column] = stock_data[target_column].iloc[date_index]
		# Calculate delta as the difference between the current and previous adjusted close prices
		new_tech_indicators.at[new_tech_indicators.index[i], 'delta'] = curr_delta
		new_tech_indicators.at[new_tech_indicators.index[i], 'gain'] = max(new_tech_indicators.at[new_tech_indicators.index[i], 'delta'], 0)
		new_tech_indicators.at[new_tech_indicators.index[i], 'loss'] = min(new_tech_indicators.at[new_tech_indicators.index[i], 'delta'], 0)

	#rearrange the columns in the order of Date, Adj Close, RSI, delta, gain, loss, avg_gain, avg_loss, rs
	new_tech_indicators = new_tech_indicators[['Date', target_column, column_name, 'delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs']]

	# remove the rows with NaN values. NaN values maybe present on dates that are weekends or holidays when there is no
	# market data available as the stock market is closed
	new_tech_indicators.dropna(inplace=True)

	#set the date column as the index
	new_tech_indicators.set_index('Date', inplace=True)

	print("New technical indicators:")
	print(new_tech_indicators.tail(14)) #print the values of the last few features
	print("Stock data:")
	print(stock_data[[target_column, 'delta', 'avg_gain', 'avg_loss', 'rs', column_name]].tail(40))

	# compare the values in the new features with the stock_data
	# compare backwards from  the last value in the new_tech_indicators
	print("Comparing the incrementally calculated RSI values with values from stock_data:")

	for i in range(n):
		comparison_date = new_tech_indicators.index[len(new_tech_indicators)-1-i]
		print(f"i: {i}, date: {comparison_date.strftime('%Y-%m-%d')}")
		#get the index of the date in stock_data and new_tech_indicators
		stock_data_index = stock_data.index.get_loc(comparison_date)
		new_tech_indicators_index = new_tech_indicators.index.get_loc(comparison_date)
		try:
			tolerance = 1e-6  # Define a tolerance threshold
			assert abs(new_tech_indicators[column_name].iloc[new_tech_indicators_index] - stock_data[column_name].iloc[stock_data_index]) <= tolerance, \
				f"Values differ by more than {tolerance} for column {column_name} on date {comparison_date}"
		except AssertionError:
			print(f"Assertion failed for column: {column_name}")
			print(f"new_tech_indicators[{column_name}].iloc[{new_tech_indicators_index}]: {new_tech_indicators[column_name].iloc[new_tech_indicators_index]}")
			print(f"stock_data[{column_name}].iloc[{stock_data_index}]: {stock_data[column_name].iloc[stock_data_index]}")
			raise

def test_incremental_bollinger_bands(stock_data, target_column, start_date, end_date, window_size=20):
	"""
	Calculate Bollinger Bands for a given DataFrame of closing prices.
	The typical technique of using previous mean and standard deviation to calculate new values 
	does not work well for incremental calculation of Bollinger Bands. This is because decimal point
	precision is lost when using the mean and standard deviation of the previous values to calculate
	the new values. Hence, we use the rolling mean and standard deviation to calculate the new values.
	The rolling mean and standard deviation are calculated using the last 20 (window_size) values of the closing prices.

	Parameters:
		df (pd.DataFrame): DataFrame containing closing prices.
		window_size (int): Window size for rolling calculations.

	Returns:
		pd.DataFrame: DataFrame with Bollinger Bands added.
	"""
	# set n to the number of days between start_date and end_date
	print(f"start_date: {start_date}, end_date: {end_date}")

	#get all the valid dates in stock_data by checking if the date is between start_date and end_date
	valid_dates = stock_data.index[(stock_data.index > start_date) & (stock_data.index <= end_date)]

	# Initialize the DataFrame with the required number of rows
	n = len(valid_dates)
	new_tech_indicators = pd.DataFrame(index=range(n), columns=['Date', target_column, 'roll_mean', 'roll_std', 'B_L', 'B_U'])
	#add a date column to new_tech_indicators
	#format date as yyyy-mm-dd
	new_tech_indicators['Date'] = pd.Series(dtype='datetime64[ns]')
	#format date as yyyy-mm-dd
	new_tech_indicators['Date'] = pd.to_datetime(new_tech_indicators['Date']).dt.strftime('%Y-%m-%d')
	new_tech_indicators[target_column] = pd.Series(dtype='float64')
	#add columns ema_short, ema_long, signal_line, macd to new_tech_indicators dataframe
	#ema stands for exponential moving average
	new_tech_indicators['roll_mean'] = pd.Series(dtype='float64')
	new_tech_indicators['roll_std'] = pd.Series(dtype='float64')
	new_tech_indicators['abs_^2_var'] = pd.Series(dtype='float64')
	new_tech_indicators['B_L'] = pd.Series(dtype='float64') 
	new_tech_indicators['B_U'] = pd.Series(dtype='float64') 

	window_size = 20
	#load the stock_data for the given date range into the new_tech_indicators dataframe
	for i, curr_date in enumerate(valid_dates):
		date_index = stock_data.index.get_loc(curr_date)
		print(f"i: {i}, curr_date: {curr_date}")
		new_tech_indicators.at[new_tech_indicators.index[i], 'Date'] = curr_date
		new_tech_indicators.at[new_tech_indicators.index[i], target_column] = stock_data[target_column].iloc[date_index]
		
		rolling_mean, rolling_std, bollinger_lower, bollinger_upper = calc_incremental_bollinger_bands(stock_data, curr_date, target_column, window_size)
		new_tech_indicators.at[new_tech_indicators.index[i],'roll_mean'] = rolling_mean
		new_tech_indicators.at[new_tech_indicators.index[i],'roll_std'] = rolling_std
		# Calculate the absolute squared variance
		# This is a workaround to avoid using the rolling mean and std directly for incremental calculation
		# new_tech_indicators.at[new_tech_indicators.index[i],'abs_^2_var'] = (np.abs(stock_data[target_column] - stock_data['roll_mean'])) ** 2
		# Calculate the absolute squared variance
		# stock_data['stdev_roll'] = stock_data['target_column].rolling(20).std()

		new_tech_indicators.at[new_tech_indicators.index[i],'B_L'] = bollinger_lower
		new_tech_indicators.at[new_tech_indicators.index[i],'B_U'] = bollinger_upper

		# stock_data['BolU_roll'] = round(
		#     stock_data['close'].rolling(20).mean() + (2 * (stock_data['close'].rolling(20).std(ddof=0))), 4)
		# stock_data['BolL_roll'] = round(
		#     stock_data['close'].rolling(20).mean() - (2 * (stock_data['close'].rolling(20).std(ddof=0))), 4)


	#set the date column as the index
	new_tech_indicators.set_index('Date', inplace=True)

	return new_tech_indicators

if __name__ == '__main__':

	config_params = load_configuration_params()
	start_date = config_params['start_date']
	end_date = config_params['end_date']
	increment = config_params['increment']

	timeframes_df = get_timeframes_df(start_date, end_date, increment)

	load_from_files=False
	if load_from_files == False:
		fetch_data_into_files(config_params, timeframes_df)

# Initialize an empty dictionary to store dataframes
	stock_data_dict = {}
	tickers = config_params['tickers']
	source_files_folder = config_params['source_files_folder']
	for ticker in tickers:
		stock_data = load_and_transform_data_by_ticker(ticker, timeframes_df, source_files_folder)
		# Store the dataframe in the dictionary with the ticker as the key
		stock_data_dict[ticker] = stock_data
		
	#only one ticker presently
	#todo need tocheck if same LSTM works for multiple tickers
	stock_data = stock_data_dict[tickers[0]]
	# Calculate rolling averages
	stock_data['Tgt'] = stock_data['Adj Close'].shift(-1) # next day's price

	# Add engineered features
	stock_data, feature_columns = add_engineered_features(stock_data)
	print("After adding engineered features:")
	print(stock_data.info())

	if not feature_columns:
		feature_columns = []  # Ensure feature_columns is defined even if add_engineered_features fails
	stock_data.dropna(inplace=True)
	print(stock_data[['Adj Close'] + feature_columns].tail(14)) #print the values of the last few features

	# Calculate and test 7-day moving average incrementally
	print("Calculating 7-day moving average incrementally:")
	test_incremental_nday_average(7, stock_data,'7d_avg', 'Adj Close')
		
	# Calculate and test 30-day moving average incrementally
	test_incremental_nday_average(30, stock_data, '30d_avg', 'Adj Close')

	#TODO: Vol_7 and Vol_30
	# Calculate and test RSI incrementally
	test_incremental_rsi(stock_data, 'Adj Close', 'RSI', start_date, end_date)

	test_incremental_macd(stock_data, 'Adj Close', 'MACD', start_date, end_date, n_short=12, n_long=26, n_signal=9)
	
	# Calculate and test Bollinger Bands incrementally
	print("Calculating Bollinger Bands incrementally:")
	z = len(stock_data)
	n = 20
	end_date = stock_data.index[z-1]
	start_date = stock_data.index[z-n-1]
	new_tech_indicators = test_incremental_bollinger_bands(stock_data, 'Adj Close', start_date, end_date, n)
	print("New technical indicators:")
	print(new_tech_indicators.tail(26)) #print the values of the last few features
	print("Stock data:")
	print(stock_data[['Adj Close', 'roll_mean', 'roll_std', 'B_L', 'B_U']].tail(52)) #n+1

	# compare the values in the new features with the stock_data
	# compare backwards from  the last value in the new_tech_indicators
	print("Comparing the incrementally calculated Bollinger Band values with values from stock_data:")
	for i in range(n):
		comparison_date = new_tech_indicators.index[len(new_tech_indicators)-1-i]
		print(f"i: {i}, date: {comparison_date.strftime('%Y-%m-%d')}")
		#get the index of the date in stock_data and new_tech_indicators
		stock_data_index = stock_data.index.get_loc(comparison_date)
		new_tech_indicators_index = new_tech_indicators.index.get_loc(comparison_date)
		# Print the values of the new_tech_indicators and stock_data for debugging
		print(f"new_tech_indicators['B_L'].iloc[{new_tech_indicators_index}]: {new_tech_indicators['B_L'].iloc[new_tech_indicators_index]}")
		print(f"stock_data['B_L'].iloc[{stock_data_index}]: {stock_data['B_L'].iloc[stock_data_index]}")
		print(f"new_tech_indicators['B_U'].iloc[{new_tech_indicators_index}]: {new_tech_indicators['B_U'].iloc[new_tech_indicators_index]}")
		print(f"stock_data['B_U'].iloc[{stock_data_index}]: {stock_data['B_U'].iloc[stock_data_index]}")
		# Assert that the values are equal upto d decimal places
		d = 6
		assert round(new_tech_indicators['B_L'].iloc[new_tech_indicators_index],d) == round(stock_data['B_L'].iloc[stock_data_index],d)
		assert round(new_tech_indicators['B_U'].iloc[new_tech_indicators_index],d) == round(stock_data['B_U'].iloc[stock_data_index],d)

	#Clip or remove outliers in the target column:
	stock_data['Tgt'] = stock_data['Tgt'].clip(
		lower=stock_data['Tgt'].quantile(0.01),
		upper=stock_data['Tgt'].quantile(0.99)
	)
	
	#calculate the log returns
	stock_data['log_returns'] = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
	stock_data['log_returns'] = stock_data['log_returns'].fillna(0)

	#create a plot with multiple sections
	fig, ax = plt.subplots(2, 2, figsize=(14, 12)) # 2 rows, 2 columns
	ax = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]] # Flatten the axes for easier indexing

	sns.histplot(stock_data['Adj Close'], kde=True, ax=ax[0], color='blue', kde_kws={}, label='Adj Close')
	ax[0].set_title('Adjusted Close Price')
	ax[0].set_xlabel('Date')
	ax[0].set_ylabel('Price')
	ax[0].legend()
	ax[0].grid()

	#normalize the target column
	#stock_data['N_Tgt'] = np.log1p(stock_data['Tgt'])  # Use log(1 + Tgt) to handle zeros
	# from scipy.stats import boxcox

	# stock_data['Adj Close_BoxCox'], _ = boxcox(stock_data['Adj Close'] + 1)  # Add 1 to avoid log(0)

	# qt = QuantileTransformer(output_distribution='normal', random_state=42)
	# stock_data['Adj Close_Quantile'] = qt.fit_transform(stock_data[['Adj Close']])

	# #plot the normalized Adjusted Close Price
	# sns.histplot(stock_data['Adj Close_Quantile'], kde=True, ax=ax[1], color='orange', label='Normalized Target')
	# ax[1].set_title('Quantile Transformed Target Price')
	# ax[1].set_xlabel('Date')
	# ax[1].set_ylabel('Price')
	# ax[1].legend()
	# ax[1].grid()

	# Apply K-Means clustering
	stock_data['Cluster'] = KMeans(n_clusters=2, random_state=42).fit_predict(stock_data[['Adj Close']])

	# Plot some sample rows in each cluster
	for cluster_label in stock_data['Cluster'].unique():
		print(f"Cluster {cluster_label} sample rows:")
		print(stock_data[stock_data['Cluster'] == cluster_label].head(5))

	# Include the cluster labels as a feature
	feature_columns.append('Cluster')
	sns.scatterplot(data=stock_data, x=stock_data.index, ax=ax[1], y='Adj Close', hue='Cluster', palette='Set1')
	ax[1].set_title('Clusters in Adjusted Close Prices')
	ax[1].set_xlabel('Date')
	ax[1].set_ylabel('Price')
	ax[1].legend()
	ax[1].grid()

	#plot the second subplot with the technical indicators as a line plot
	#plot the following data in the same graph for ticker 0
	#plot Adj Close, Daily Return, 7_day_avg, 30_day_avg, Volatility_7, Volatility_30, RSI, MACD, Signal_Line, Bollinger_Upper, Bollinger_Lower
	#plot the data for the first ticker in the list
	#assume data is already loaded and calculated
	ticker = tickers[0]
	plot_averages(stock_data, ticker, ax[2])

	plot_bollinger_bands(stock_data, ticker, ax[3])

	plt.tight_layout()
	plt.show()
	
	#locate the exact date where the value in the Cluster columntransitions from 0 to 1
	for index, row in stock_data.iterrows():
		if row['Cluster'] == 1:
			print(f"Cluster changed to 1 on {index}")
			break
