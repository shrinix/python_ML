import pandas as pd
from pandas import DataFrame
import os
from datetime import date, timedelta
import yfinance as yf
import configparser
import json
import numpy as np
from pandas import concat
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.cluster import KMeans
from keras.models import Sequential, Model
import glob
import warnings
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Define scalers_dict as a global variable
scalers_dict = {}

# #Combine the data from the different ticker-specific files into a single dataframe for the given time periods
def load_and_transform_data_by_ticker(ticker, timeframes_df, source_files_folder):
	
	stock_data = pd.DataFrame()
	#iterate through the timeframes_df dataframe
	for index, row in timeframes_df.iterrows():
		start_date = row['From Date']
		end_date = row['To Date']

		print(start_date)
		print(end_date)
		
		filename = f'{source_files_folder}/portfolio_data_{ticker}_{start_date}_{end_date}.csv'
		print("Reading file: ", filename)
		#read each file and concatenate to the data dataframe
		df = pd.read_csv(filename)
		df['Ticker'] = ticker
		#Change name of 'Price' column to 'Date'
		df.rename(columns={'Price':'Date'}, inplace=True)
		#change datatype of 'Date' column to datetime
		df['Date'] = pd.to_datetime(df['Date'])
		df['Adj Close'] = pd.to_numeric(df['Adj Close'])
		df['Close'] = pd.to_numeric(df['Close'])
		df['High'] = pd.to_numeric(df['High'])
		df['Low'] = pd.to_numeric(df['Low'])
		df['Open'] = pd.to_numeric(df['Open'])
		# Concatenate to the main DataFrame
		stock_data = pd.concat([stock_data, df])

	print("Stock data date range:", stock_data['Date'].min(), "to", stock_data['Date'].max())

	#sort the data by Date and Ticker
	stock_data.sort_values(by=['Date', 'Ticker'], inplace=True)
	#write the stock_data to a csv file
	stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()
	min_date = stock_data['Date'].min().strftime('%Y-%m-%d')
	max_date = stock_data['Date'].max().strftime('%Y-%m-%d')
	stock_data.to_csv(f'{source_files_folder}/portfolio_data_all_{min_date}_{max_date}.csv')

	stock_data['Date'] = pd.to_datetime(stock_data['Date'])
	stock_data.reset_index(inplace=True)
	stock_data.set_index('Date', inplace=True)

	#drop the column called 'Index'
	stock_data.drop(columns='index', inplace=True)
	#Arrange the remaining columns in the order: Date,Ticker,Adj Close,Close,High,Low,Open,Volume
	stock_data = stock_data[['Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

	# print(stock_data.columns)
	# print(stock_data.info())
	# print(stock_data.head())
	# print(stock_data.tail())
	#print unique values in Ticker column
	# print("Unique values in Ticker column:", stock_data['Ticker'].unique())

	return stock_data

def load_and_transform_data_by_date(tickers, timeframes_df, source_files_folder, group_all_dates=False):
	
	stock_data = pd.DataFrame()
	#iterate through the timeframes_df dataframe
	for index, row in timeframes_df.iterrows():
		start_date = row['From Date']
		end_date = row['To Date']

		print(start_date)
		print(end_date)
		
		for ticker in tickers:
			filename = f'{source_files_folder}/portfolio_data_{ticker}_{start_date}_{end_date}.csv'
			print("Reading file: ", filename)
			#read each file and concatenate to the data dataframe
			df = pd.read_csv(filename)
			#First row is header row with column names
			# Skip the next 2 rows
			df = df.iloc[2:]
			# Add a column for the ticker and set it to the ticker value
			df['Ticker'] = ticker
			#Change name of 'Price' column to 'Date'
			df.rename(columns={'Price':'Date'}, inplace=True)
			#change datatype of 'Date' column to datetime
			df['Date'] = pd.to_datetime(df['Date'])
			df['Adj Close'] = pd.to_numeric(df['Adj Close'])
			df['Close'] = pd.to_numeric(df['Close'])
			df['High'] = pd.to_numeric(df['High'])
			df['Low'] = pd.to_numeric(df['Low'])
			df['Open'] = pd.to_numeric(df['Open'])
			# Concatenate to the main DataFrame
			stock_data = pd.concat([stock_data, df])

		#sort the data by Date and Ticker
		stock_data.sort_values(by=['Date', 'Ticker'], inplace=True)
		#write the stock_data to a csv file
		stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()
		stock_data.to_csv(f'{source_files_folder}/portfolio_data_all_{start_date}_{end_date}.csv')

		stock_data['Date'] = pd.to_datetime(stock_data['Date'])
		stock_data.reset_index(inplace=True)
		stock_data.set_index('Date', inplace=True)

		#drop the column called 'Index'
		stock_data.drop(columns='index', inplace=True)
		#Arrange the remaining columns in the order: Date,Ticker,Adj Close,Close,High,Low,Open,Volume
		stock_data = stock_data[['Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

		# print(stock_data.columns)
		# print(stock_data.info())
		# print(stock_data.head())
		# print(stock_data.tail())
		#print unique values in Ticker column
		# print("Unique values in Ticker column:", stock_data['Ticker'].unique())

		return stock_data
	
def load_configuration_params():

	#get all configuration parameters from the config.properties file as strings as they will be set as environment variables
	config_params = {}

	config = configparser.ConfigParser()
	config.read('./config.properties')

	  # define the time period for the data
	start_date = config.get('Portfolio', 'start_date') #'2020-02-16'
	end_date = config.get('Portfolio', 'end_date')
	# end_date = str(date.today())
	#Backtesting dates: 3rd July 2023 to 29th June 2024
	#uncomment for backtesting
	#--> start_date = date.fromisoformat('2023-07-03') #backtesting start date
	#--> end_date = date.fromisoformat('2024-06-29') #backtesting end date. Use one date more than the actual end date
	config_params['start_date'] = date.fromisoformat(start_date).strftime('%Y-%m-%d')
	config_params['end_date'] = date.fromisoformat(end_date).strftime('%Y-%m-%d')

	increment = config.get('Portfolio', 'increment') #365
	config_params['increment'] = timedelta(days=int(increment))

	transaction_cost_percent = config.get('Portfolio', 'transaction_cost_percent') #0.01 
	config_params['transaction_cost_percent'] = transaction_cost_percent

	# tickers = ['RELIANCE.NS', 'TCS.NS','INFY.NS', 'HDFCBANK.NS']
	# tickers = ['AAPL','BAC', 'BK', 'LIT', 'VTSAX', 'CSCO', 'GIS','SONY','INTC']
	tickers = config.get('Portfolio', 'tickers')#['BAC', 'LIT', 'VTSAX', 'CSCO', 'GIS','SONY','INTC']
	if tickers.find(',') != -1:
		tickers = tickers.split(',')
	else:
		tickers = [tickers]
	config_params['tickers'] = tickers
	
	currency = config.get('Portfolio', 'currency') #'USD'
	config_params['currency'] = currency

	#convert initial_allocation to a string before setting as environment variable
	# initial_allocation_str = json.dumps(initial_allocation)
	# config_params['initial_allocation'] = initial_allocation_str

	config_params['source_files_folder'] = config.get('Portfolio', 'source_files_folder') #'source_files'
	config_params['output_files_folder'] = config.get('Portfolio', 'output_files_folder') #'output_files'
	config_params['ref_data_folder'] = config.get('Portfolio', 'ref_data_folder')
	config_params['local_repository'] = config.get('Portfolio', 'local_repository')
	config_params['split_date'] = config.get('Portfolio', 'split_date')

	print("Configuration parameters:")
	print(config_params)

	#create and set environment variables based on the configuration parameters
	os.environ['start_date'] = config_params['start_date']
	os.environ['end_date'] = config_params['end_date']
	os.environ['increment'] = str(config_params['increment'])
	os.environ['transaction_cost_percent'] = config_params['transaction_cost_percent']
	os.environ['tickers'] = ','.join(map(str, config_params['tickers']))
	os.environ['currency'] = config_params['currency']
	# os.environ['initial_allocation'] = config_params['initial_allocation']
	os.environ['source_files_folder'] = config_params['source_files_folder']
	os.environ['output_files_folder'] = config_params['output_files_folder']
	os.environ['ref_data_folder'] = config_params['ref_data_folder']

	return config_params

def get_max_version(output_files_folder, file_prefix='lstm_model', look_for_older_files=False):
	current_date = datetime.now().strftime('%Y-%m-%d')

	# Find the maximum version number for existing files
	existing_versions = [
		int(file.split('_v')[-1].split('.')[0])
		for file in os.listdir(output_files_folder)
			if file.startswith(f'{file_prefix}_{current_date}_v') and file.endswith('.keras')
	]
	version = max(existing_versions, default=0)
	date = current_date

	if version > 0:
		return version, date
	
	#if version is 0, check if there is any other version from the latest past date
	#check day-wise for the last 7 days
	if version == 0 and look_for_older_files:
		date_filename_dictionary = {}
		#look for files starting with lstm_model.
		#sort the files by date with latest date first and then by version with latest version first
		for file in os.listdir(output_files_folder):
			orig_filename = file
			if file.startswith(file_prefix) and file.endswith('.keras'):
				#extract date and store (date,filename) in a dictionary
				date_key = file.replace(file_prefix+'_', '').split('_v')[0]
				#if date_Filename_dictionary already has the date_key, check if the version is greater than the existing version
				#if it is, update the filename
				#if it is not, skip the file
				if date_key in date_filename_dictionary:
					#if the version is greater than the existing version, update the filename
					if int(file.split('_v')[-1].split('.')[0]) > int(date_filename_dictionary[date_key].split('_v')[-1].split('.')[0]):
						date_filename_dictionary[date_key] = orig_filename
				else:
					date_filename_dictionary[date_key] = orig_filename

		#get the entries with the latest date from the dictionary
		latest_date = max(date_filename_dictionary.keys(), default=current_date)
		#get the files with the latest date
		latest_files = [date_filename_dictionary[date_key] for date_key in date_filename_dictionary.keys() if date_key.startswith(latest_date)]
		#get the version numbers from the files
		existing_versions = [
			int(file.split('_v')[-1].split('.')[0])
			for file in latest_files
		]
	   
		version = max(existing_versions, default=0)
		date = latest_date

	return version, date

def get_ticker_data(tickers, timeframes_df, source_files_folder, local_repository):
	for index, row in timeframes_df.iterrows():
		start_date = row['From Date']
		end_date = row['To Date']
		# Define a custom business day calendar that excludes weekends and US federal holidays
		us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

		num_business_days = len(pd.date_range(start=start_date, end=end_date, freq=us_bd))

		if isinstance(tickers, str):
			tickers = tickers.split(',')

		for ticker in tickers:
			print(f"Processing ticker: {ticker}")
			filename = f"{source_files_folder}/portfolio_data_{ticker}_{start_date}_{end_date}.csv"

			# Step 1: Check if data exists in a CSV file
			data = check_csv_file(filename)
			if data is not None and not data.empty:
				print("Data loaded from local CSV file.")
				continue

			# Step 2: Check cached files in the local source folder
			data = check_cached_files(ticker, start_date, end_date, source_files_folder, num_business_days)
			if data is not None and not data.empty:
				print("Data loaded from cached files.")
				data.to_csv(filename)
				continue

			# Step 3: Check the local repository (TXT file)
			data = check_local_repository(ticker, start_date, end_date, local_repository, source_files_folder, num_business_days)
			if data is not None and not data.empty:
				print("Data loaded from local TXT datastore.")
				data.to_csv(filename)
				continue

			# Step 4: Fetch data from Yahoo Finance API
			data = fetch_data_from_yahoo(ticker, start_date, end_date, filename)
			if data is not None and not data.empty:
				print("Data fetched from Yahoo Finance API.")
				continue

def check_csv_file(filename):
	"""Check if the data exists in a local CSV file."""
	if os.path.exists(filename):
		print(f"File already exists: {filename}")
		return pd.read_csv(filename, index_col='Date')
	return None

def check_cached_files(ticker, start_date, end_date, source_files_folder, num_business_days, threshold=10):
	"""Check if the data exists in cached files."""
	consolidated_data = pd.DataFrame() #use a consolidated dataframe to store the data spread across multiple cached files
	data = pd.DataFrame() #use a data dataframe to store the data for the current cached file
	cached_files = glob.glob(f"{source_files_folder}/portfolio_data_all_*.csv")
	for cached_file in cached_files:
		data = pd.read_csv(cached_file, sep=',', index_col='Date')
		data.index = pd.to_datetime(data.index)  # Ensure index is datetime
		data = data[(data['Ticker'] == ticker) &
					(data.index >= pd.to_datetime(start_date)) &
					(data.index <= pd.to_datetime(end_date))]
		if data.empty:
			print(f"No data found for {ticker} in {cached_file}.")
			continue
		if len(data) < num_business_days:
			print(f"Data for {ticker} in {cached_file} has fewer rows ({len(data)}) than the number of business days ({num_business_days}).")
			consolidated_data = pd.concat([consolidated_data, data])
			continue

	# Check if the data is already in the consolidated DataFrame
	if consolidated_data is not None and not consolidated_data.empty:
		#remove duplicates from consolidated_data
		consolidated_data = consolidated_data.drop_duplicates()
		
		# Check if all the business days are present in the consolidated_data. 
		# check if number of rows in consolidated_data is less than num_business_days
		if len(consolidated_data) < num_business_days:
			print(f"Consolidated data for {ticker} in {cached_file} has fewer rows ({len(consolidated_data)}) than the number of business days ({num_business_days}).")
			#check if data gap is less than threshold
			if (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days < threshold:
				print(f"Data gap for {ticker} in {cached_file} is less than threshold of {threshold} days.")			
				data = consolidated_data
	return data


def check_local_repository(ticker, start_date, end_date, local_repository, source_files_folder, num_business_days):
	"""Check if the data exists in the local repository (TXT file)."""
	data = get_ticker_datafile_from_local_datastore(ticker, start_date, end_date, local_repository, source_files_folder)
	if data is not None:
		if len(data) < num_business_days:
			print(f"Data for {ticker} in local repository has fewer rows ({len(data)}) than the number of business days ({num_business_days}).")
			return None
	return data


def fetch_data_from_yahoo(ticker, start_date, end_date, filename):
	"""Fetch data from Yahoo Finance API."""
	print(f"Fetching data from Yahoo Finance API for {ticker} for date range {start_date} to {end_date}")
	data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
	if not data.empty:
		#Create a temporary filename by changing the name from <filename>.csv to <filename>.original.csv
		#strip the extension from the filename and insert .original before the extension
		original_filename = filename.split('.')[0] + ".original" + ".csv"
		data.to_csv(original_filename)

		# Read the file and handle the first two rows
		with open(original_filename, 'r') as file:
			lines = file.readlines()

		# Remove the first two rows (ticker and empty row)
		lines = lines[2:]

		# Write the cleaned lines back to a temporary file
		cleaned_filename = filename.split('.')[0] + ".cleaned" + ".csv"
		with open(cleaned_filename, 'w') as file:
			file.writelines(lines)

		# Read the cleaned file into a DataFrame
		data = pd.read_csv(cleaned_filename)

		# Set the correct column names
		data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

		# Drop rows where all values are NaN
		data.dropna(how='all', inplace=True)

		# Ensure the 'Date' column is of datetime type
		data['Date'] = pd.to_datetime(data['Date'])

		# Save the cleaned data to the final file
		data.to_csv(filename, index=False)
		print(f"Cleaned data saved to: {filename}")
		return data
	print(f"No data fetched from Yahoo Finance API for {ticker}.")
	return None

def pre_process(all_tickers_file, tickers_list):
#the all_tickers_file is the file that contains data about all the tickers.
#split the data into individual tickers and save them to individual csv files
#the first row of the all_tickers_file should contain the column names
#the column name for the tickers should be 'Ticker'
#The inout CSV file has data in the following format:
#,Date,Adj Close,Close,High,Low,Open,Volume,Ticker,Daily Return
#the function will also remove empty columns and reorder the columns as follows:
#Date,Open,High,Low,Close,Volume,Name
#where Name is the name of the ticker
#The input file name is in the format:<portfolio_data_all>_<start_date>_<end_date>.csv
#The output files will be in the format:<ticker>_<start_date>_to_<end_date>.csv
	all_tickers_df = pd.read_csv(all_tickers_file)
	for ticker in tickers_list:
		ticker_df = all_tickers_df[all_tickers_df['Ticker'] == ticker]
		ticker_df = ticker_df.dropna(axis=1, how='all')
		ticker_df = ticker_df[['Date','Open','High','Low','Close','Volume','Ticker']]
		ticker_df.columns = ['Date','Open','High','Low','Close','Volume','Name']
		ticker_df.to_csv('input/'+ticker + '_' + all_tickers_file.split('_')[3] + '_to_' + all_tickers_file.split('_')[4], index=False)
	return

def get_timeframes_df(start_date, end_date, increment, remove_short_timeframes=False):
	#calculate number of iterations to run using start_date and today's date
	
	#calculate the number of days between start_date and today's date
	# Ensure end_date and start_date are datetime.date objects
	if not isinstance(end_date, date):
		end_date = date.fromisoformat(end_date)
	if not isinstance(start_date, date):
		start_date = date.fromisoformat(start_date)
	
	num_days = (end_date - start_date).days
	#calculate the number of increments to run
	num_increments = num_days // increment.days
	#check if there is a nonzero remainder when dividing num_days by increment.days
	remaining_days = num_days % increment.days
	if remaining_days > 0:
		num_increments += 1

	timeframes_list = []
	for i in range(1, num_increments):
		timeframes_list.append({'From Date': start_date, 'To Date': start_date + increment})
		start_date += (increment+timedelta(days=1))

	max_to_date = end_date
	#if number of entries in the timeframes_list is greater than 1
	if len(timeframes_list) > 1:
		#get max 'To Date' in the timeframes_list
		max_to_date = max(timeframes_list, key=lambda x:x['To Date'])['To Date']

	#add any remainder days to the timeframes_list
	#if remaining_days is greater than 0 and max 'To Date' is less than today's date
	#or if remaining_days is greater than 0 and len(timeframes_list) is 0

	if remaining_days > 0 and (max_to_date < end_date or len(timeframes_list) == 0):
		#if From Date is same as To Date, skip the iteration
		if start_date != end_date:
			timeframes_list.append({'From Date': start_date, 'To Date': end_date})

	#Check if any of the intervals in the timeframes_list is less than increment, and delete it
	#get the timeframes with less than increment days
	if remove_short_timeframes:
		timeframes_to_be_removed = [timeframe for timeframe in timeframes_list if (timeframe['To Date'] - timeframe['From Date']).days < increment.days]
		print("Removed following timeframes with duration less than increment days:")
		print(timeframes_to_be_removed)
		#remove the timeframes with less than increment days
		timeframes_list = [timeframe for timeframe in timeframes_list if (timeframe['To Date'] - timeframe['From Date']).days >= increment.days]
	
	timeframes_df = pd.DataFrame(timeframes_list)
	# Define a custom business day calendar that excludes weekends and US federal holidays
	us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

	#for each row in the timeframes_df dataframe, calculate the number of business days between From Date and To Date
	timeframes_df['Num Business Days'] = timeframes_df.apply(lambda x: len(pd.date_range(start=x['From Date'], end=x['To Date'], freq=us_bd)), axis=1)
	print("Timeframes DataFrame:")
	print(timeframes_df)
	#print the total number of business days in the timeframes_df dataframe
	print("Total number of business days in the timeframes_df dataframe:")
	print(timeframes_df['Num Business Days'].sum())
	return timeframes_df

def merge_data(timeframes_df, source_files_folder):

	merged_stock_data = pd.DataFrame()
	for index, row in timeframes_df.iterrows():
		start_date = row['From Date']
		end_date = row['To Date']
		df = pd.read_csv(f'{source_files_folder}/portfolio_data_all_{start_date}_{end_date}.csv')
		df['Start Date'] = start_date  # Add start_date to each row
		merged_stock_data = pd.concat([merged_stock_data, df])

	#check for NaN values in the merged_stock_data dataframe and drop them
	print("NaN values in the merged_stock_data dataframe:")
	print(merged_stock_data.isnull().sum())
	merged_stock_data.dropna(inplace=True)

	#remove column named 'Unnamed: 0'
	if 'Unnamed: 0' in merged_stock_data.columns:
		merged_stock_data.drop(columns='Unnamed: 0', inplace=True)

	#convert numeric columns to have only 2 decimal places
	numeric_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
	merged_stock_data[numeric_cols] = merged_stock_data[numeric_cols].apply(lambda x: np.round(x,2))

	return merged_stock_data

def compare_dataframes(df1, df2, lookup_cols, remove_cols=None):

	#make a copy of the dataframes to avoid changing the original dataframes
	df1 = df1.copy()
	df2 = df2.copy()
	
	#remove cols from df1 and df2 if remove_cols is not None
	if remove_cols is not None:
		df1.drop(columns=remove_cols, inplace=True, errors='ignore')
		df2.drop(columns=remove_cols, inplace=True, errors='ignore')

	#Reorder the columns in both DataFrames to match:
	df1 = df1[lookup_cols]
	df2 = df2[lookup_cols]

	#print shapes of the dataframes
	print("Shape of df1:", df1.shape)
	print("Shape of df2:", df2.shape)

	# Check for NaN values in lookup columns
	print("NaN values in df1:")
	print(df1[lookup_cols].isna().sum())
	print("\nNaN values in df2:")
	print(df2[lookup_cols].isna().sum())

	#compare two dataframes manually using the lookup columns
	for col in lookup_cols:
		if col not in df1.columns or col not in df2.columns:
			raise ValueError(f"Column {col} not found in one of the DataFrames.")
	
	# Perform the comparison
	comparison = df1[lookup_cols].merge(df2[lookup_cols], on=lookup_cols, how='outer', indicator=True)
	# Check for differences
	differences = comparison[comparison['_merge'] != 'both']
	if not differences.empty:
		print(f"{len(differences)} Differences found:")
		#print count of items which are in df1 but not in df2
		print(f"Items in df1 but not in df2: {len(differences[differences['_merge'] == 'left_only'])}")
		#print count of items which are in df2 but not in df1
		print(f"Items in df2 but not in df1: {len(differences[differences['_merge'] == 'right_only'])}")
		# print(differences)

		# Compare row-by-row
	for i, (row1, row2) in enumerate(zip(df1[lookup_cols].itertuples(index=False), df2[lookup_cols].itertuples(index=False))):
		if row1 != row2:
			print(f"Row {i} mismatch:")
			print(f"df1: {row1}")
			print(f"df2: {row2}")
			input("Press Enter to continue...")

	else:
		print("No differences found.")

	# Return True if no differences found, False otherwise
	return differences.empty

# Perform reverse lookup for each row in original source DataFrame
def reverse_lookup(row, source_lookup, lookup_columns):
	"""
	Perform reverse lookup to find the corresponding date in the train DataFrame,
	retaining non-trading days by propagating feature values.

	Parameters:
		row (pd.Series): A row from X_encoder_train_df.
		source_lookup (pd.DataFrame): The train DataFrame with feature columns and dates.
		lookup_columns (list): List of feature columns to use for matching.

	Returns:
		datetime or None: The corresponding date if a match is found, otherwise None.
	"""
   # Match the row values with the train DataFrame
	match = source_lookup[(source_lookup[lookup_columns] == row[lookup_columns]).all(axis=1)]
	if not match.empty:
		return match['Date'].iloc[0]  # Return the first matching date
	else:
		return None  # Return None if no match is found

def fetch_data_into_files(config_params, timeframes_df):

	tickers = config_params['tickers']
	#tickers is a string, convert it to a list
	source_files_folder = config_params['source_files_folder']
	local_repository = config_params['local_repository']

	if isinstance(tickers, str):
		tickers = tickers.split(',')
	for ticker in tickers:
		print("Fetching data for ticker: ", ticker)
		get_ticker_data(ticker, timeframes_df, source_files_folder, local_repository)

def calculate_donchian_bands(stock_data, period=20):
	"""
	Calculate Donchian Bands for the given stock data.

	Parameters:
		stock_data (pd.DataFrame): DataFrame containing stock data with 'High' and 'Low' columns.
		period (int): The lookback period for calculating the bands (default is 20 days).

	Returns:
		pd.DataFrame: DataFrame with Donchian Bands added as new columns.
	"""
	# Ensure the required columns are present
	if 'High' not in stock_data.columns or 'Low' not in stock_data.columns:
		raise ValueError("Stock data must contain 'High' and 'Low' columns.")

	# Calculate the upper and lower bands
	stock_data['Donchian_Upper'] = stock_data['High'].rolling(window=period).max()
	stock_data['Donchian_Lower'] = stock_data['Low'].rolling(window=period).min()

	# Calculate the middle band (optional)
	stock_data['Donchian_Middle'] = (stock_data['Donchian_Upper'] + stock_data['Donchian_Lower']) / 2

	return stock_data

def add_technical_features(stock_data):
# Number of samples, timesteps, and features
	# n_samples, n_timesteps, n_features = X.shape

	# features = []
	# # 1. Statistical Features
	# mean_feature = np.mean(stock_data, axis=1, keepdims=True)  # Shape: (samples, 1, features)
	# std_feature = np.std(stock_data, axis=1, keepdims=True)    # Shape: (samples, 1, features)
	# min_feature = np.min(stock_data, axis=1, keepdims=True)    # Shape: (samples, 1, features)
	# max_feature = np.max(stock_data, axis=1, keepdims=True)    # Shape: (samples, 1, features)
	# range_feature = max_feature - min_feature         # Shape: (samples, 1, features)
	
	# # Broadcast statistical features to match the timesteps
	# mean_feature = np.repeat(mean_feature, n_timesteps, axis=1)
	# std_feature = np.repeat(std_feature, n_timesteps, axis=1)
	# min_feature = np.repeat(min_feature, n_timesteps, axis=1)
	# max_feature = np.repeat(max_feature, n_timesteps, axis=1)
	# range_feature = np.repeat(range_feature, n_timesteps, axis=1)

	# # Add statistical features
	# X_augmented = np.concatenate([X_augmented, mean_feature, std_feature, min_feature, max_feature, range_feature], axis=2)
	# features.extend(["mean", "std", "min", "max", "range"])

	# # 2. Temporal Features
	# time_index = np.linspace(0, 1, n_timesteps).reshape(1, n_timesteps, 1)  # Normalized time index
	# time_index = np.repeat(time_index, n_samples, axis=0)  # Repeat for all samples
	# X_augmented = np.concatenate([X_augmented, time_index], axis=2)
	# features.append("time_index")

	# 3. Frequency Domain Features
	fft_features = np.abs(fft(X, axis=1))  # Compute FFT along the time axis
	fft_features = fft_features[:, :, :n_features]  # Keep only the first `n_features` FFT components
	X_augmented = np.concatenate([X_augmented, fft_features], axis=2)
	features.extend(["fft_real", "fft_imag"])  # Ensure this is correct

	# # 4. Trend and Seasonality (using Savitzky-Golay filter for smoothing)
	# smoothed = savgol_filter(X, window_length=5, polyorder=2, axis=1)  # Apply smoothing
	# X_augmented = np.concatenate([X_augmented, smoothed], axis=2)
	# features.append("smoothed")

	# # 5. Derived Mathematical Features
	# cumulative_sum = np.cumsum(X, axis=1)  # Cumulative sum
	# cumulative_product = np.cumprod(X + 1e-6, axis=1)  # Cumulative product (add small value to avoid zeros)
	# X_augmented = np.concatenate([X_augmented, cumulative_sum, cumulative_product], axis=2)
	# features.extend(["cumulative_sum", "cumulative_product"])

	# # 6. Rolling Window Features
	# rolling_mean = np.zeros_like(X)
	# rolling_variance = np.zeros_like(X)
	# window_size = 3
	# for i in range(n_timesteps):
	#     start = max(0, i - window_size + 1)
	#     rolling_mean[:, i, :] = np.mean(X[:, start:i+1, :], axis=1)
	#     rolling_variance[:, i, :] = np.var(X[:, start:i+1, :], axis=1)
	# X_augmented = np.concatenate([X_augmented, rolling_mean, rolling_variance], axis=2)
	# features.extend(["rolling_mean", "rolling_variance"])

	# # 7. Interaction Features
	# if n_features > 1:
	#     interaction_features = []
	#     for i in range(n_features):
	#         for j in range(i + 1, n_features):
	#             interaction_features.append(X[:, :, i] * X[:, :, j])  # Multiply features
	#     interaction_features = np.stack(interaction_features, axis=2)  # Stack along the feature axis
	#     X_augmented = np.concatenate([X_augmented, interaction_features], axis=2)
	#     features.extend([f"interaction_{i}_{j}" for i in range(n_features) for j in range(i + 1, n_features)])

# Add engineered features to the stock data
def add_engineered_features(stock_data):

	feature_columns = []

	# check if the index is a datetime index
	print("Checking for datetime index and gaps in the dataset...")
	if not pd.api.types.is_datetime64_any_dtype(stock_data.index):
		print("Index is not a datetime index. Converting to datetime index.")
		stock_data.index = pd.to_datetime(stock_data.index)
	
	#check that there are no gaps in the dataset using an assert statement
	# Calculate the full range of dates
	full_date_range = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max())
	# Find missing dates
	missing_dates = full_date_range.difference(stock_data.index)
	# Check if there are any missing dates
	if not missing_dates.empty:
		print(f"{len(missing_dates)} Gaps found in the dataset. Missing dates (showing up to 10): {missing_dates[:10]}")

	for date in pd.date_range(start=stock_data.index.min(), end=stock_data.index.max()):
		if date not in stock_data.index:
			print(f"Data not found for {date} in dataframe")
			#implement logic to handle missing data
			# For example, you can carry forward the last known value
			#forward-fill as the missing data is due to weekends and holidays when the stock market is closed
			stock_data.fillna(method='ffill', inplace=True)
			print(f"Data for {date} added to dataframe")

	# Calculate moving averages
	# Since window size is 7 for 7_day_avg, and 30 for 30_day_avg, but min_periods is set to 1 for both,
	# the first 7 rows and the first 30 rows will have non-NaN values for the following columns, respectively:
	# 7_day_avg, 30_day_avg
	stock_data['7d_avg'] = stock_data['Adj Close'].rolling(window=7, min_periods=1).mean()
	# Clip outliers to a reasonable range
	# lower_bound = np.percentile(stock_data['7d_avg'], 1)  # 1st percentile
	# upper_bound = np.percentile(stock_data['7d_avg'], 99)  # 99th percentile
	# stock_data['7d_avg'] = np.clip(stock_data['7d_avg'], lower_bound, upper_bound)

	stock_data['30d_avg'] = stock_data['Adj Close'].rolling(window=30, min_periods=1).mean()
	feature_columns.extend(['7d_avg', '30d_avg'])

	# Calculate daily returns
	stock_data['Daily_Ret'] = stock_data['Adj Close'].pct_change()
	feature_columns.append('Daily_Ret')

	# Calculate rolling standard deviations (volatilities)
	# Since window size is 7 for Volatility_7, and 30 for Volatility_30, the first 7 rows and the first 30 rows 
	# will have NaN values for the following columns, respectively:
	# Volatility_7, Volatility_30
	stock_data['Vol_7'] = stock_data['Adj Close'].rolling(window=7).std()
	stock_data['Vol_30'] = stock_data['Adj Close'].rolling(window=30).std()
	feature_columns.extend(['Vol_7', 'Vol_30'])

	epsilon = 1e-10 #to avoid division by zero
	# Calculate Relative Strength Index (RSI (6))
	delta_1 = stock_data['Adj Close'].diff()
	rsi_period_1 = 6
	avg_gain_1 = (delta_1.where(delta_1 > 0, 0)).rolling(window=rsi_period_1).mean()
	avg_loss_1 = (-delta_1.where(delta_1 < 0, 0)).rolling(window=rsi_period_1).mean()
	rs_1 = avg_gain_1 / (avg_loss_1+epsilon) # Avoid division by zero
	stock_data['delta_1'] = delta_1
	stock_data['avg_gain_1'] = avg_gain_1
	stock_data['avg_loss_1'] = avg_loss_1
	stock_data['rs_1'] = rs_1
	stock_data['RSI_1'] = 100 - (100 / (1 + rs_1))
	# Add delta, gain, loss, rs to feature columns
	feature_columns.extend(['delta_1', 'avg_gain_1', 'avg_loss_1', 'rs_1'])
	# Add RSI to feature columns
	feature_columns.append('RSI_1')

	# Calculate Relative Strength Index (RSI (12)
	delta_2 = stock_data['Adj Close'].diff()
	rsi_period_2 = 12
	avg_gain_2 = (delta_2.where(delta_2 > 0, 0)).rolling(window=rsi_period_2).mean()
	avg_loss_2 = (-delta_2.where(delta_2 < 0, 0)).rolling(window=rsi_period_2).mean()
	rs_2 = avg_gain_2 / (avg_loss_2+epsilon) # Avoid division by zero
	stock_data['delta_2'] = delta_2
	stock_data['avg_gain_2'] = avg_gain_2
	stock_data['avg_loss_2'] = avg_loss_2
	stock_data['rs_2'] = rs_2
	stock_data['RSI_2'] = 100 - (100 / (1 + rs_2))
	# Add delta, gain, loss, rs to feature columns
	feature_columns.extend(['delta_2', 'avg_gain_2', 'avg_loss_2', 'rs_2'])
	# Add RSI to feature columns
	feature_columns.append('RSI_2')

	# Calculate Moving Average Convergence Divergence (MACD (6,15,6))
	ema_6 = stock_data['Adj Close'].ewm(span=6, adjust=False).mean()
	ema_15 = stock_data['Adj Close'].ewm(span=15, adjust=False).mean()
	stock_data['ema_6'] = ema_6
	stock_data['ema_15'] = ema_15
	stock_data['MACD_1'] = ema_6 - ema_15
	#Signal_Line
	stock_data['Sig_L_1'] = stock_data['MACD_1'].ewm(span=6, adjust=False).mean()
	feature_columns.extend(['ema_6', 'ema_15'])
	feature_columns.extend(['MACD_1', 'Sig_L_1']) 

	# Calculate Moving Average Convergence Divergence (MACD (12,26,9))
	ema_12 = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
	ema_26 = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()
	stock_data['ema_12'] = ema_12
	stock_data['ema_26'] = ema_26
	stock_data['MACD_2'] = ema_12 - ema_26
	#Signal_Line
	stock_data['Sig_L_2'] = stock_data['MACD_2'].ewm(span=9, adjust=False).mean()
	feature_columns.extend(['ema_12', 'ema_26'])
	feature_columns.extend(['MACD_2', 'Sig_L_2']) 

	# Calculate Bollinger Bands
	# Since window size is 20, the first 20 rows will have NaN values for the columns:
	# rolling_mean, rolling_std, Bollinger_Upper, and Bollinger_Lower
	rolling_mean = stock_data['Adj Close'].rolling(window=20).mean()
	rolling_std = stock_data['Adj Close'].rolling(window=20).std(ddof=0)
	stock_data['B_U'] = rolling_mean + (rolling_std * 2)
	stock_data['B_L'] = rolling_mean - (rolling_std * 2)
	stock_data['roll_mean'] = rolling_mean
	stock_data['roll_std'] = rolling_std
	feature_columns.extend(['B_U', 'B_L'])
	feature_columns.extend(['roll_mean', 'roll_std'])

	#----START---- Remove for forecasting to work. Need to add incremental versions.
	# Apply K-Means clustering
	# stock_data['Cluster'] = KMeans(n_clusters=2, random_state=42).fit_predict(stock_data[['Adj Close']])
	# feature_columns.append('Cluster')

	# # Donchian Bands
	# donchian_period = 20  # Rolling window size
	# stock_data['Don_U'] = stock_data['High'].rolling(window=donchian_period).max()  # Upper Band
	# stock_data['Don_L'] = stock_data['Low'].rolling(window=donchian_period).min()   # Lower Band
	# stock_data['Don_M'] = (stock_data['Don_U'] + stock_data['Don_L']) / 2           # Middle Band (optional)
	# feature_columns.extend(['Don_U', 'Don_L', 'Don_M'])

	# # Keltner Bands
	# keltner_period = 20  # Rolling window size
	# multiplier = 2  # Multiplier for ATR
	# # Calculate EMA of the closing price
	# stock_data['Kelt_EMA'] = stock_data['Adj Close'].ewm(span=keltner_period, adjust=False).mean()
	# # Calculate True Range (TR)
	# stock_data['TR'] = stock_data[['High', 'Low', 'Adj Close']].apply(
	# 	lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Adj Close']), abs(row['Adj Close'] - row['Low'])),
	# 	axis=1
	# )
	# # Calculate Average True Range (ATR)
	# stock_data['ATR'] = stock_data['TR'].rolling(window=keltner_period).mean()
	# # Calculate Keltner Bands
	# stock_data['Kelt_U'] = stock_data['Kelt_EMA'] + (multiplier * stock_data['ATR'])
	# stock_data['Kelt_L'] = stock_data['Kelt_EMA'] - (multiplier * stock_data['ATR'])
	# # Add Keltner Bands to feature columns
	# feature_columns.extend(['TR','ATR','Kelt_EMA', 'Kelt_U', 'Kelt_L'])
	#----END---- Remove for forecasting to work. Need to add incremental versions.

	if stock_data.isnull().sum().sum() > 0:
		print(stock_data.isnull().sum())
		print("NaN values found in the stock_data dataframe. Implementing logic to handle missing data.")            
		# # Fill NaN values using forward fill first, then backward fill
		# stock_data.fillna(method='ffill', inplace=True)  # Forward fill
		# stock_data.fillna(method='bfill', inplace=True)  # Backward fill
		# # Drop rows with NaN values
		stock_data.dropna(inplace=True)

	print("After removing NaN values:")
	print(stock_data.isnull().sum())

	#START - Commenting out for debugging: Removing outliers seems to change the data, especially on 03/28/25
	#find outliers in the stock_data dataframe
	# outlier_columns = ['Adj Close', '7d_avg', '30d_avg', 'Tgt',
	# 						 'Daily_Ret', 'Vol_7', 'Vol_30', 
	# 						 'delta', 'avg_gain', 'avg_loss', 'rs', 'RSI', 
	# 						 'ema_12', 'ema_26', 'MACD', 'Sig_L', 
	# 						 'roll_mean', 'roll_std', 'B_U', 'B_L',
	# 						# Removed for forecaster to work. 'Cluster','Don_U', 'Don_L', 'Don_M', 'Kelt_EMA', 'Kelt_U', 'Kelt_L']
	# 						 ]
	# outliers = detect_outliers_iqr(stock_data, outlier_columns)
	# if any(not outlier_data.empty for outlier_data in outliers.values()):
	# 	print("Outliers found in the following columns:")
	# 	for col, outlier_indices in outliers.items():
	# 		if not outlier_indices.empty:
	# 			print(f"Column: {col}, Outlier Indices: {outlier_indices}")
	# 		else:
	# 			print(f"Column: {col}, No outliers detected.")
	# 	#Replace with Rolling Mean:
	# 	for col, outlier_indices in outliers.items():
	# 		if not outlier_indices.empty:
	# 			stock_data.loc[outlier_indices.index, col] = stock_data[col].rolling(window=20).mean().loc[outlier_indices.index]
	# 		else:
	# 			print(f"Column: {col}, No outliers detected.")
	# #check again to confirm that the outliers have been removed
	# print("After removing outliers:")
	#END - Commenting out for debugging: Removing outliers seems to change the data, especially on 03/28/25

	# #print the row in stock_data for the date 3/28/2025
	# print(stock_data.loc['2025-03-28'])
	# input("Press Enter to continue...")

	#create a dataframe of all the columns including features and target alongwith their min and max values
	feature_metrics_columns = ['Adj Close', '7d_avg', '30d_avg', 'Tgt',
							 'Daily_Ret', 'Vol_7', 'Vol_30', 
							 'delta_1', 'avg_gain_1', 'avg_loss_1', 'rs_1', 'RSI_1', 
							 'delta_2', 'avg_gain_2', 'avg_loss_2', 'rs_2', 'RSI_2', 
							 'ema_6', 'ema_15', 'MACD_1', 'Sig_L_1', 
							 'ema_12', 'ema_26', 'MACD_2', 'Sig_L_2', 
							 'B_U', 'B_L','roll_mean', 'roll_std'
							# #  Removed for forecaster to work 'Cluster', 'Don_U', 'Don_L', 'Don_M', 'Kelt_EMA', 'Kelt_U', 'Kelt_L'
							 ]

	#Check if feature_metrics_columns are in the same order as the columns in stock_data
	for col in feature_metrics_columns:
		if col not in stock_data.columns:
			print(f"Column {col} not found in stock_data dataframe.")
			#raise ValueError(f"Column {col} not found in stock_data dataframe.")
		else:
			print(f"Column {col} found in stock_data dataframe.")
	#Analyze skewness of features
	for col in feature_metrics_columns:
		skewness = np.round(skew(stock_data[col].dropna()),2)
		print(f"Feature: {col}, Skewness: {skewness}")

	feature_metrics_df = pd.DataFrame(columns=['Feature', 'Min', 'Max'])
	for col in feature_metrics_columns:
		min_val = stock_data[col].min()
		max_val = stock_data[col].max()
		feature_metrics_df = pd.concat([feature_metrics_df, pd.DataFrame([{'Feature': col, 'Min': min_val, 'Max': max_val}])], ignore_index=True)
	
	print("Feature metrics:")
	print(feature_metrics_df)

	#Commented out for debugging - START
  	#Check for negative values for avg gain and avg loss columns in data dataframe using assert statements
	# if (stock_data['avg_gain'] < 0).any():
	# 	print("Negative values found in avg_gain column. Showing a few violating rows:")
	# 	print(stock_data[stock_data['avg_gain'] < 0].head())
	# 	assert False, "Negative values found in avg_gain column."

	# if (stock_data['avg_loss'] < 0).any():
	# 	print("Negative values found in avg_loss column. Showing a few violating rows:")
	# 	print(stock_data[stock_data['avg_loss'] < 0].head())
	# 	assert False, "Negative values found in avg_loss column."
	#Commented out for debugging - END

	#scale and plot histogram to test scaling for debugging
	# #drop ticker column
	# stock_data_select = stock_data.copy()
	# stock_data_select.drop(columns='Ticker', inplace=True, errors='ignore')
	# split_date = pd.to_datetime('2023-10-01')
	# train = stock_data_select.loc[stock_data_select.index < split_date]
	# test = stock_data_select.loc[stock_data_select.index >= split_date]
	# # transform scale
	# scaler = MinMaxScaler(feature_range=(0, 1))
	# scaler = scaler.fit(train)
	# train_scaled = scaler.transform(train)
	# test_scaled = scaler.transform(test)

	# #draw a histogram+kde plot of the scaled data
	# #draw each feature in a separate subplot. draw 3 features in each row
	# import seaborn as sns
	# import matplotlib.pyplot as plt
	# num_features = len(feature_metrics_columns)
	# num_rows = (num_features + 2) // 3
	# fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
	# for i, feature in enumerate(feature_metrics_columns):
	#     print("Feature:", feature)
	#     ax = axes[i // 3, i % 3]
	#     print("Shape of scaled data:", train_scaled.shape)
	#     sns.histplot(train_scaled[:,i], kde=True, ax=ax)
	#     ax.set_title(f"Scaled Feature: {feature}")
	#     ax.set_xlabel("Value")
	#     ax.set_ylabel("Density")
	#     ax.grid()
	# plt.tight_layout()
	# plt.show()

	#print the first 10 rows of the stock_data dataframe
	print("First 10 rows of stock_data dataframe:")
	print(stock_data.head(10))
	#print the last 10 rows of the stock_data dataframe
	print("Last 10 rows of stock_data dataframe:")
	print(stock_data.tail(10))

	# #print the row in stock_data for the date 3/28/2025
	# print(stock_data.loc['2025-03-28'])
	# input("Press Enter to continue...")
	return stock_data, feature_columns

def detect_outliers_iqr(stock_data, columns):
	outliers = {}
	for col in columns:
		Q1 = stock_data[col].quantile(0.25)
		Q3 = stock_data[col].quantile(0.75)
		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR
		outliers[col] = stock_data[(stock_data[col] < lower_bound) | (stock_data[col] > upper_bound)]
		print(f"Outliers in {col} (IQR Method):")
		print(outliers[col])
	return outliers

def check_data(data, feature_columns):
	print("Checking data for NaN values and gaps in the dataset...")

	#check for NaN values in the dataframe
	if data.isnull().sum().sum() > 0:
		violating_rows = data[data.isnull().any(axis=1)].head()  # Show a few violating rows
		raise ValueError(f"NaN values found in the dataframe: {data.isnull().sum()}\nViolating rows:\n{violating_rows}")
		# #Use forward fill to fill NaN values
		# data.fillna(method='ffill', inplace=True)
			 
	# #check for outliers in the stock_data dataframe
	# print("Checking for outliers...")
	# outlier_columns = ['Tgt'] #Tgt = 'Adj Close'
	# outliers = detect_outliers_iqr(stock_data, outlier_columns)
	# if any(not outlier_data.empty for outlier_data in outliers.values()):
	#     print("Outliers found in the following columns:")
	#     for col, outlier_indices in outliers.items():
	#         if not outlier_indices.empty:
	#             print(f"Column: {col}, Outlier Indices: {outlier_indices}")
	#         else:
	#             print(f"Column: {col}, No outliers detected.")
	#         #Replace with Rolling Mean:
	#         print("Replacing outliers using rolling means with window=5")
	#         stock_data.loc[outlier_indices.index, col] = stock_data[col].rolling(window=5, min_periods=1).mean().reindex(stock_data.index).loc[outlier_indices.index]
	
	#check if all the columns in feature_columns are present in the stock_data dataframe using an assert statement
	assert all(col in data.columns for col in feature_columns), f"Not all feature columns are present in the dataframe. Missing columns: {[col for col in feature_columns if col not in data.columns]}"
	
	# Check for negative values for avg gain and avg loss columns in data dataframe using assert statements
	# TODO: this check will not work for multiple timestepped columns. - START
	# if (data['avg_gain'] < 0).any():
	# 	print("Negative values found in avg_gain column. Showing a few violating rows:")
	# 	print(data[data['avg_gain'] < 0].head())
	# 	assert False, "Negative values found in avg_gain column."

	# if (data['avg_loss'] < 0).any():
	# 	print("Negative values found in avg_loss column. Showing a few violating rows:")
	# 	print(data[data['avg_loss'] < 0].head())
	# 	assert False, "Negative values found in avg_loss column."
	# TODO: this check will not work for multiple timestepped columns. - START
	return True

def get_scaler(column, is_existing=False):

	#dictionary to store scalers by feature or column name
	global scalers_dict  # Use the global scalers_dict to ensure that it is created only once
	
	#TODO: Need to create these lists dynamically using skewness and range of the features
	# Group features based on their ranges and skewness
	# minmax_features = ['Adj Close', 'Cluster', 'Tgt', '7d_avg', '30d_avg', 'ema_12', 'ema_26', 'roll_mean', 'B_U', 'B_L']
	# robust_features = ['Vol_7', 'Vol_30', 'avg_loss', 'rs', 'roll_std', 'avg_gain']
	# standard_features = ['Daily_Ret', 'RSI', 'delta', 'MACD', 'Sig_L']

	robust_features = []
	standard_features  = []
	minmax_features= ['Adj Close', 'Cluster', 'Tgt', '7d_avg', '30d_avg', 'ema_12', 'ema_26', 'roll_mean', 'B_U', 'B_L',
					   'Vol_7', 'Vol_30', 'avg_loss', 'rs', 'roll_std', 'avg_gain',
					   'Daily_Ret', 'RSI', 'delta', 'MACD', 'Sig_L', 'Don_U', 'Don_L', 'Don_M', 'Kelt_EMA', 'Kelt_U', 'Kelt_L']

	# Create a scaler for the specified column
	if is_existing:
		return scalers_dict.get(column)
	else:
		if column in minmax_features:
			scaler = MinMaxScaler(feature_range=(0, 1))
			# if column == 'B_L':
			#     input("Press Enter to continue")

		elif column in robust_features:
			scaler = RobustScaler()
		elif column in standard_features:
			scaler = StandardScaler()
		else:
			raise ValueError(f"Unknown feature: {column}")

		# Fit the scaler on the data
		scalers_dict[column] = scaler
	return scaler

#function to determine split date for training and testing. Given a date range, it will return the date that is 80% of the way through the range.
#it will use business days to determine the split date
def get_split_date(start_date, end_date, percentage=0.8):
	# Convert start_date and end_date to datetime objects
	start_date = pd.to_datetime(start_date)
	end_date = pd.to_datetime(end_date)

	#use the US holiday calendar to determine business days
	us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

	# Generate a date range with business days
	date_range = pd.date_range(start=start_date, end=end_date, freq=us_bd)

	# Calculate the split index
	split_index = int(len(date_range) * percentage)

	# Get the split date
	split_date = date_range[split_index]

	return split_date

def prepare_data_for_training(stock_data, feature_columns, split_date, target_column='Tgt', disable_scaling=False):
	# # Remove unnecessary columns. 'Adj Close' is also removed as it has been replaced by the 'Tgt' column
	# unnecessary_columns = ['Ticker', 'Volume', 'Close', 'High', 'Low', 'Open', 'Adj Close']
	# stock_data.drop(columns=unnecessary_columns, inplace=True, errors='ignore')
	# print("After removing unnecessary columns:")
	
	#print columns of the stock_data dataframe
	print(stock_data.columns)
	
	check_data(stock_data, feature_columns)

	# Print the columns in stock_data
	print(stock_data.columns)
	print("Before train-test split:", stock_data.shape)
	print("Dataset date range:", stock_data.index.min(), "to", stock_data.index.max())

	# Split the data into training and test sets
	# Convert split_date to YYYY-MM-DD format string for consistency
	# Clean split_date to remove any comments or extra text
	if isinstance(split_date, str):
		split_date = split_date.split('#')[0].strip()
	split_date = pd.to_datetime(split_date).strftime('%Y-%m-%d')
	print("Adjusted split date:", split_date)
	train = stock_data.loc[stock_data.index < split_date]
	test = stock_data.loc[stock_data.index >= split_date]

	# Print % of data in train and test
	print("Train data %: ", train.shape[0] / stock_data.shape[0])
	print("Test data %: ", test.shape[0] / stock_data.shape[0])

	print("Shapes after train-test split:")
	print("Train data shape:", train.shape)
	print("Test data shape:", test.shape)
	# Print min and max values of target column in train and test data
	print("Train data target range:", train[target_column].min(), "to", train[target_column].max())
	print("Test data target range:", test[target_column].min(), "to", test[target_column].max())

	print('\n-------Train and Test data before scaling-------')
	print(np.round(train.head(), 4))
	print("\n")
	print(np.round(test.head(), 4))

	# Ensure train and test data are not empty
	if train.empty or test.empty:
		raise ValueError("Train or test data is empty. Check the split date or input data.")

	train_prepared = train.copy()
	test_prepared = test.copy()
	feature_scaler = None
	target_scaler = None

	if not disable_scaling:
		# Scale the feature columns
		feature_scaler = MinMaxScaler(feature_range=(0, 1))
		feature_scaler = feature_scaler.fit(train[feature_columns])
		train_features_scaled = feature_scaler.transform(train[feature_columns])
		test_features_scaled = feature_scaler.transform(test[feature_columns])

		# Scale the target column separately
		target_scaler = MinMaxScaler(feature_range=(0, 1))
		target_scaler = target_scaler.fit(train[[target_column]])
		train_target_scaled = target_scaler.transform(train[[target_column]])
		test_target_scaled = target_scaler.transform(test[[target_column]])

		print("\n-------Train and Test after scaling-------")
		print("Feature columns (scaled):")
		print(np.round(train_features_scaled[:5], 4))
		print("\nTarget column (scaled):")
		print(np.round(train_target_scaled[:5], 4))

		# Convert the scaled data back to DataFrames
		train_prepared = pd.DataFrame(train_features_scaled, columns=feature_columns, index=train.index)
		train_prepared[target_column] = train_target_scaled

		test_prepared = pd.DataFrame(test_features_scaled, columns=feature_columns, index=test.index)
		test_prepared[target_column] = test_target_scaled

	else:
		print("Not scaling the data")

	# Check for data leaks
	assert train_prepared.index.max() < test_prepared.index.min(), "Data leakage detected: Train set overlaps with test set."

	return train_prepared, test_prepared, feature_scaler, target_scaler

def validate_input_data(data, feature_columns, target_column, n_steps_in, n_steps_out):
	"""
	Validate the input data and parameters for creating overlapping arrays.
	"""
	if len(data) < n_steps_in + n_steps_out:
		raise ValueError(f"Not enough data to create sequences with n_steps_in={n_steps_in} and n_steps_out={n_steps_out}.")
	if not all(col in data.columns for col in feature_columns):
		raise ValueError(f"Some feature columns are missing in the data: {feature_columns}")
	if target_column not in data.columns:
		raise ValueError(f"Target column '{target_column}' is missing in the data.")
	if data.isnull().any().any():
		raise ValueError("Input data contains missing values. Please clean the data before generating sequences.")

def generate_sequences(data, feature_columns, target_column, n_steps_in, n_steps_out):
	"""
	Generate overlapping sequences for encoder input, decoder input, and decoder output.

	Parameters:
		data (pd.DataFrame): Input data with a DatetimeIndex.
		feature_columns (list): List of feature column names.
		target_column (str): Name of the target column.
		n_steps_in (int): Number of input timesteps.
		n_steps_out (int): Number of output timesteps.

	Returns:
		tuple: (X_encoder, decoder_input, y_decoder, X_encoder_dates, decoder_input_dates, y_decoder_dates)
	"""
	X_encoder, decoder_input, y_decoder = [], [], []
	X_encoder_dates, decoder_input_dates, y_decoder_dates = [], [], []

	# Ensure the input DataFrame has unique and sorted indices
	if not data.index.is_unique:
		raise ValueError("Input data index contains duplicate values. Please ensure the index is unique.")
	if not data.index.is_monotonic_increasing:
		print("Input data index is not sorted. Sorting the index.")
		data = data.sort_index()

	# Total number of samples that can be generated
	total_samples = len(data) - n_steps_in - n_steps_out + 1

	# Ensure there are enough rows to generate at least one sample
	if total_samples <= 0:
		raise ValueError(f"Not enough data to generate sequences with n_steps_in={n_steps_in} and n_steps_out={n_steps_out}.")

	# Loop through the data to create overlapping sequences
	for i in range(total_samples):
		# Encoder input
		encoder_seq = data[feature_columns].iloc[i:i + n_steps_in]
		X_encoder.append(encoder_seq.values)
		X_encoder_dates.append(encoder_seq.index)

		# Decoder input
		decoder_seq = data[feature_columns].iloc[i + n_steps_in:i + n_steps_in + n_steps_out]
		decoder_input.append(decoder_seq.values)
		decoder_input_dates.append(decoder_seq.index)

		# Decoder output
		target_seq = data[target_column].iloc[i + n_steps_in:i + n_steps_in + n_steps_out]
		y_decoder.append(target_seq.values.reshape(-1, 1))
		y_decoder_dates.append(target_seq.index)

	# Ensure unique dates in X_encoder_dates
	X_encoder_dates = [list(dict.fromkeys(dates)) for dates in X_encoder_dates]

	return X_encoder, decoder_input, y_decoder, X_encoder_dates, decoder_input_dates, y_decoder_dates

# Check for duplicate dates in each date list
def convert_to_dataframes(X_encoder, decoder_input, y_decoder, 
						  X_encoder_dates, decoder_input_dates, y_decoder_dates, 
						  feature_columns, target_column):
	"""
	Convert overlapping sequences into DataFrames with MultiIndex.

	Parameters:
		X_encoder (list): Encoder input sequences.
		decoder_input (list): Decoder input sequences.
		y_decoder (list): Decoder output sequences.
		X_encoder_dates (list): Dates for encoder input sequences.
		decoder_input_dates (list): Dates for decoder input sequences.
		y_decoder_dates (list): Dates for decoder output sequences.
		feature_columns (list): List of feature column names.
		target_column (str): Name of the target column.

	Returns:
		tuple: (X_encoder_df, decoder_input_df, y_decoder_df)
	"""
	# Construct DataFrames with proper MultiIndex
	X_encoder_df = pd.concat(
		[pd.DataFrame(data=seq, index=pd.MultiIndex.from_tuples(
			[(date, step) for step, date in enumerate(dates)],  # Use all dates in the sequence
			names=["Date", "Step"]
		), columns=feature_columns) for seq, dates in zip(X_encoder, X_encoder_dates)]
	)

	# Drop duplicate rows based on the "Date" level
	X_encoder_df = X_encoder_df.reset_index().drop_duplicates(subset=["Date"]).set_index(["Date", "Step"])

	decoder_input_df = pd.concat(
		[pd.DataFrame(data=seq, index=pd.MultiIndex.from_tuples(
			[(date, step) for step, date in enumerate(dates)],
			names=["Date", "Step"]
		), columns=feature_columns) for seq, dates in zip(decoder_input, decoder_input_dates)]
	)

	y_decoder_df = pd.concat(
		[pd.DataFrame(data=seq, index=pd.MultiIndex.from_tuples(
			[(date, step) for step, date in enumerate(dates)],
			names=["Date", "Step"]
		), columns=[target_column]) for seq, dates in zip(y_decoder, y_decoder_dates)]
	)

	return X_encoder_df, decoder_input_df, y_decoder_df

def convert_to_dataframes(X_encoder, decoder_input, y_decoder, 
						  X_encoder_dates, decoder_input_dates, y_decoder_dates, 
						  feature_columns, target_column):
	"""
	Convert overlapping sequences into DataFrames with MultiIndex.

	Parameters:
		X_encoder (list): Encoder input sequences.
		decoder_input (list): Decoder input sequences.
		y_decoder (list): Decoder output sequences.
		X_encoder_dates (list): Dates for encoder input sequences.
		decoder_input_dates (list): Dates for decoder input sequences.
		y_decoder_dates (list): Dates for decoder output sequences.
		feature_columns (list): List of feature column names.
		target_column (str): Name of the target column.

	Returns:
		tuple: (X_encoder_df, decoder_input_df, y_decoder_df)
	"""
	# Add a sequence ID to ensure uniqueness in the MultiIndex
	sequence_ids = range(len(X_encoder))

	# Construct DataFrames with proper MultiIndex for X_encoder
	X_encoder_df = pd.concat(
		[pd.DataFrame(data=seq, index=pd.MultiIndex.from_tuples(
			[(date, step, seq_id) for step, date in enumerate(dates)],  # Add sequence ID to the MultiIndex
			names=["Date", "Step", "SequenceID"]
		), columns=feature_columns) for seq, dates, seq_id in zip(X_encoder, X_encoder_dates, sequence_ids)]
	)
	
	# Sort the DataFrame to ensure proper order
	X_encoder_df = X_encoder_df.sort_index()

	# Debugging: Check for duplicate rows in X_encoder_df
	print("X_encoder_df shape before duplicate check:", X_encoder_df.shape)
	if X_encoder_df.index.duplicated().any():
		print("Duplicate rows found in X_encoder_df. Dropping duplicates...")
		X_encoder_df = X_encoder_df[~X_encoder_df.index.duplicated(keep='first')]

	# Reset index to ensure uniqueness
	X_encoder_df = X_encoder_df.reset_index().drop_duplicates(subset=["Date", "Step"]).set_index(["Date", "Step", "SequenceID"])
	print("X_encoder_df shape after duplicate check:", X_encoder_df.shape)

	# Construct DataFrames for decoder_input and y_decoder
	decoder_input_df = pd.concat(
		[pd.DataFrame(data=seq, index=pd.MultiIndex.from_tuples(
			[(date, step, seq_id) for step, date in enumerate(dates)],
			names=["Date", "Step", "SequenceID"]
		), columns=feature_columns) for seq, dates, seq_id in zip(decoder_input, decoder_input_dates, sequence_ids)]
	)

	y_decoder_df = pd.concat(
		[pd.DataFrame(data=seq, index=pd.MultiIndex.from_tuples(
			[(date, step, seq_id) for step, date in enumerate(dates)],
			names=["Date", "Step", "SequenceID"]
		), columns=[target_column]) for seq, dates, seq_id in zip(y_decoder, y_decoder_dates, sequence_ids)]
	)

	return X_encoder_df, decoder_input_df, y_decoder_df

def create_overlapping_arrays(data, feature_columns, target_column, n_steps_in, n_steps_out):
	"""
	Create overlapping arrays for Seq2Seq model training while preserving date information.

	Parameters:
		data (pd.DataFrame): Input DataFrame with a DatetimeIndex.
		feature_columns (list): List of feature column names.
		target_column (str): Name of the target column.
		n_steps_in (int): Number of input timesteps.
		n_steps_out (int): Number of output timesteps.

	Returns:
		tuple: (X_encoder_df, decoder_input_df, y_decoder_df)
	"""
	# Step 1: Validate input data
	validate_input_data(data, feature_columns, target_column, n_steps_in, n_steps_out)

	# Step 2: Generate overlapping sequences
	X_encoder, decoder_input, y_decoder, X_encoder_dates, decoder_input_dates, y_decoder_dates = generate_sequences(
		data, feature_columns, target_column, n_steps_in, n_steps_out
	)

	# Step 3: Convert sequences to DataFrames
	X_encoder_df, decoder_input_df, y_decoder_df = convert_to_dataframes(
		X_encoder, decoder_input, y_decoder, 
		X_encoder_dates, decoder_input_dates, y_decoder_dates, 
		feature_columns, target_column
	)

	return X_encoder_df, decoder_input_df, y_decoder_df

def convert_dataframe_to_3d_array(df, n_timesteps, n_features):
	"""
	Convert a DataFrame with MultiIndex (Date, Step) to a 3D NumPy array.

	Parameters:
		df (pd.DataFrame): Input DataFrame with MultiIndex (Date, Step).
		n_timesteps (int): Number of timesteps.
		n_features (int): Number of features.

	Returns:
		np.ndarray: 3D NumPy array of shape (samples, timesteps, features).
	"""
	# Debugging: Print the initial structure of the DataFrame
	print("\n--- Debugging Input DataFrame ---")
	print("DataFrame Shape:", df.shape)
	print("Index Names:", df.index.names)
	print("Columns:", df.columns)
	print("First 5 Rows:\n", df.head())
	print("Last 5 Rows:\n", df.tail())

	# Validation: Check if 'Date' and 'Step' are part of the MultiIndex
	if 'Date' not in df.index.names or 'Step' not in df.index.names:
		print("Resetting index to include 'Date' and 'Step'.")
		df = df.reset_index().set_index(['Date', 'Step'])

	# Debugging: Print the index names after resetting
	print("\nIndex Names After Reset:", df.index.names)

	# Exclude non-feature columns (e.g., 'Date') from the DataFrame
	if 'Date' in df.columns:
		print("Excluding 'Date' column from the DataFrame.")
		df = df.drop(columns=['Date'])

	# Validation: Check for missing values
	if df.isnull().any().any():
		print("\nNaN values found in the DataFrame:")
		print(df.isnull().sum())
		raise ValueError("Input DataFrame contains NaN values. Please clean the data before proceeding.")

	# Validation: Check if the DataFrame has enough columns
	if len(df.columns) < n_features:
		raise ValueError(f"DataFrame has {len(df.columns)} columns, but expected at least {n_features}.")

	# Ensure the DataFrame is sorted by the MultiIndex
	df = df.sort_index()

	# Ensure all groups have the same columns
	expected_columns = df.columns[:n_features]  # Use the first `n_features` columns as the expected set
	df = df[expected_columns]  # Align all groups to the expected columns

	# Debugging: Print the columns after alignment
	print("\nColumns After Alignment:", df.columns)

	# Group by the "Date" level of the MultiIndex
	grouped = df.groupby(level="Date")

	# UNCOMMENT FOR Debugging: Check the number of rows in each group
	print("\n--- Group Information ---")
	for date, group in grouped:
		# print(f"Date: {date}, Group Shape: {group.shape}")
		if group.shape[1] != n_features:
			raise ValueError(f"Group for date {date} has {group.shape[1]} features, but expected {n_features}.")

	# Initialize an empty list to store arrays
	arrays = []

	# Process each group
	for date, group in grouped:
		if len(group) == n_timesteps:
			# Use the group as is if it matches n_timesteps
			arrays.append(group.values)
		elif len(group) < n_timesteps:
			# Pad groups with fewer rows than n_timesteps
			# UNCOMMENT FOR DEBUGGING - print(f"Padding group for date {date} with {len(group)} rows (less than {n_timesteps}).")
			padding = np.zeros((n_timesteps - len(group), n_features))
			padded_group = np.vstack([group.values, padding])
			arrays.append(padded_group)
		else:
			# Truncate groups with more rows than n_timesteps
			print(f"Truncating group for date {date} with {len(group)} rows (more than {n_timesteps}).")
			truncated_group = group.values[:n_timesteps]
			arrays.append(truncated_group)

	# UNCOMMENT FOR DEBUGGING - Validate the shapes of all arrays 
	print("\n--- Validating Array Shapes ---")
	for i, array in enumerate(arrays):
		# print(f"Array {i} Shape: {array.shape}")
		if array.shape[1] != n_features:
			raise ValueError(f"Array {i} has inconsistent shape: {array.shape}")

	# Convert the list of arrays to a 3D NumPy array
	if len(arrays) == 0:
		raise ValueError("No valid groups found. Check the input DataFrame and parameters.")
	array_3d = np.stack(arrays)

	# Debugging: Print the shape of the final 3D array
	print("\nFinal 3D Array Shape:", array_3d.shape)

	return array_3d

def prepare_seq2seq_data(data, feature_columns, target_column, n_steps_in, n_steps_out):
	"""
	Prepare data for Seq2Seq model training.

	Parameters:
		data (pd.DataFrame): Input data.
		feature_columns (list): List of feature column names.
		target_column (str): Name of the target column.
		n_steps_in (int): Number of input timesteps.
		n_steps_out (int): Number of output timesteps.

	Returns:
		X_encoder (np.ndarray): Encoder input data.
		decoder_input (np.ndarray): Decoder input data.
		y_decoder (np.ndarray): Decoder output data.
	"""
	X_encoder, decoder_input, y_decoder = [], [], []

	# Debugging: Check for missing values
	print("Checking for missing values in feature columns:")
	print(data[feature_columns].isna().sum())

	print("Checking for missing values in target column:")
	print(data[target_column].isna().sum())

	# Handle missing values if any
	if data[feature_columns].isna().sum().sum() > 0:
		print("NaN values found in the feature columns. Filling missing values with forward and backward fill.")
		data[feature_columns] = data[feature_columns].fillna(method='ffill').fillna(method='bfill')

	if data[target_column].isna().sum() > 0:
		print("NaN values found in the target column. Filling missing values with forward and backward fill.")
		data[target_column] = data[target_column].fillna(method='ffill').fillna(method='bfill')

	# #Check if avg_gain, avg_loss, rs columns have negative values
	# if (data['avg_gain'] < 0).any():
	#     print("avg_gain has negative values")
	#     input("Press Enter to continue...")
	# if (data['avg_loss'] < 0).any():
	#     print("avg_loss has negative values")
	#     input("Press Enter to continue...")
	# # Debugging: Check data length and loop range
	# print(f"Data length: {len(data)}")
	# print(f"n_steps_in: {n_steps_in}, n_steps_out: {n_steps_out}")
	# print(f"Loop range: {len(data) - n_steps_in}")

	# Check if there is enough data
	if len(data) < n_steps_in + n_steps_out:
		raise ValueError(f"Not enough data for the specified n_steps_in ({n_steps_in}) and n_steps_out ({n_steps_out}). Data length: {len(data)}")

	# Loop through the data
	for i in range(len(data) - n_steps_in + 1):
		# print(f"\nIteration {i}:")
		# print(f"Data length: {len(data)}")
		# print(f"Loop range: {len(data) - n_steps_in}")

		# Encoder input: n_steps_in timesteps of feature columns
		encoder_seq = data[feature_columns].iloc[i:i + n_steps_in].values
		# print("X_encoder sequence:")
		# print(encoder_seq)
		X_encoder.append(encoder_seq)

		# Decoder input and output
		if i + n_steps_in + n_steps_out <= len(data):
			decoder_seq_features = data[feature_columns].iloc[i + n_steps_in:i + n_steps_in + n_steps_out].values
			# print("Decoder input (decoder_input):")
			# print(decoder_seq_features)
			decoder_input.append(decoder_seq_features)

			target_seq = data[target_column].iloc[i + n_steps_in:i + n_steps_in + n_steps_out].values.reshape(-1, 1)
			# print("Decoder output (y_decoder):")
			# print(target_seq)
			y_decoder.append(target_seq)
		else:
			# Handle cases where there are fewer than n_steps_out timesteps remaining
			remaining_steps = len(data) - (i + n_steps_in)
			decoder_seq_features = data[feature_columns].iloc[i + n_steps_in:].values
			# print(f"Decoder input sequence (remaining {remaining_steps} steps):")
			# print(decoder_seq_features)

			# Pad the decoder input sequence to n_steps_out
			padded_decoder_seq = np.zeros((n_steps_out, len(feature_columns)))
			padded_decoder_seq[:remaining_steps] = decoder_seq_features
			decoder_input.append(padded_decoder_seq)

			target_seq = data[target_column].iloc[i + n_steps_in:].values.reshape(-1, 1)
			# print(f"Decoder output sequence (remaining {remaining_steps} steps):")
			# print(target_seq)

			# Pad the decoder output sequence to n_steps_out
			padded_target_seq = np.zeros((n_steps_out, 1))
			padded_target_seq[:remaining_steps] = target_seq
			y_decoder.append(padded_target_seq)

	# Convert lists to NumPy arrays
	X_encoder = np.array(X_encoder)
	decoder_input = np.array(decoder_input)
	y_decoder = np.array(y_decoder)

	# Debugging: Print final shapes of outputs
	print("\nFinal Shapes:")
	print(f"X_encoder shape: {X_encoder.shape}")
	print(f"decoder_input shape: {decoder_input.shape}")
	print(f"y_decoder shape: {y_decoder.shape}")

	return X_encoder, decoder_input, y_decoder

# def to_supervised(train, n_input, n_out=7):
# 	# flatten data
# 	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
# 	X, y = list(), list()
# 	in_start = 0
# 	# step over the entire history one time step at a time
# 	for _ in range(len(data)):
# 		# define the end of the input sequence
# 		in_end = in_start + n_input
# 		out_end = in_end + n_out
# 		# ensure we have enough data for this instance
# 		if out_end <= len(data):
# 			x_input = data[in_start:in_end, 0]
# 			x_input = x_input.reshape((len(x_input), 1))
# 			X.append(x_input)
# 			y.append(data[in_end:out_end, 0])
# 		# move along one time step
# 		in_start += 1
# 	return array(X), array(y)

# convert series to supervised learning
#data is a numpy array corresponding to the dataframe used to create the array. So it does not have any indices.
#data_frame_columns is a list of column names in the dataframe
#target_column is the name of the target column in the dataframe
#n_in is the number of input time steps
#n_out is the number of output time steps
def series_to_supervised(data, data_frame_columns,  target_column, n_in=1, n_out=1, dropnan=True, index=None):
	print(f"n_steps_in: {n_in}, n_steps_out: {n_out}")
	print(f"Number of input dataframe columns: {len(data_frame_columns)}")
	print("Shape of the input data:", data.shape)

	df = DataFrame(data, columns=data_frame_columns)
	#split the dataframe into feature columns and target column
	#remove the target column from the feature columns
	feature_columns = data_frame_columns.copy()
	feature_columns.remove(target_column)
	print("Feature columns after removing target column:", feature_columns)
	df_features = df[feature_columns]
	df_target = df[[target_column]]
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		shifted = df_features.shift(i)
		# shifted = shifted.reindex(df.index)  # Align the index with the original DataFrame
		cols.append(shifted)
		#use the feature column names for the 1st time step
		if i == 1:
			# names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
			names += [(f'{col}') for col in feature_columns]
		else:
			# names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
			names += [(f'{col}(t-{i})') for col in feature_columns]
		
	print("# of Column names after shifting features:")
	print(len(names))

	# Forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		shifted = df_target[[target_column]].shift(-i)  # Use only the target column
		# shifted = shifted.reindex(df.index)  # Align the index with the original DataFrame
		cols.append(shifted)
		if i == 0:
			names.append(f'{target_column}(t)')
		else:
			names.append(f'{target_column}(t+{i})')
			
	print("# of Column names after shifting target:")
	print(len(names))

	#print the names of the columns in names starting with Target column
	print("Column names after shifting:")
	for name in names:
		if name.startswith(target_column):
			print(name)

	#print shapes of cols
	print("Shapes of the columns:")
	for col in cols:
		print(col.shape)
	# Concatenate all DataFrames in cols along the columns axis
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		#check if there are any NaN values in the DataFrame
		if agg.isnull().sum().sum() > 0:
			count_of_nan_values = agg.isnull().sum().sum()
			print(f"{count_of_nan_values} NaN values found in the DataFrame.")
			#print a few violating rows
			violating_rows = agg[agg.isnull().any(axis=1)].head()  # Show a few violating rows
			# print(f"Violating rows:\n{violating_rows}")
			#show columns with NaN values in the violating rows
			columns_with_nan = agg.columns[agg.isnull().any()].tolist()
			# print(f"Columns with NaN values: {columns_with_nan}")
			print("Dropping NaN values...")
			agg.dropna(inplace=True)
		else:
			print("No NaN values found in the DataFrame.")	
	else:
		print("Ignoring NaN values in the DataFrame.")

	 # Retain the provided index if available
	if index is not None:
		agg.index = index[-agg.shape[0]:]  # Align the index with the resulting DataFrame

	return agg

def validate_nstep_folding_of_data_with_dataframes(n_steps_in, n_steps_out, n_features,
												   X_encoder_train_df, decoder_input_train_df, y_decoder_train_df,
												   X_encoder_test_df, decoder_input_test_df, y_decoder_test_df):
	"""
	Validate the shapes of DataFrame-based overlapping arrays for Seq2Seq models.

	Parameters:
		n_steps_in (int): Number of input timesteps.
		n_steps_out (int): Number of output timesteps.
		n_features (int): Number of features.
		X_encoder_train_df (pd.DataFrame): Encoder input training data.
		decoder_input_train_df (pd.DataFrame): Decoder input training data.
		y_decoder_train_df (pd.DataFrame): Decoder output training data.
		X_encoder_test_df (pd.DataFrame): Encoder input testing data.
		decoder_input_test_df (pd.DataFrame): Decoder input testing data.
		y_decoder_test_df (pd.DataFrame): Decoder output testing data.

	Returns:
		None
	"""
	# Validate the number of rows in X_encoder DataFrames
	train_rows = X_encoder_train_df.shape[0]
	test_rows = X_encoder_test_df.shape[0]
	unique_train_dates = len(X_encoder_train_df.index.get_level_values("Date").unique())
	unique_test_dates = len(X_encoder_test_df.index.get_level_values("Date").unique())
	expected_train_rows = unique_train_dates * n_steps_in
	expected_test_rows = unique_test_dates * n_steps_in

	# Debugging: Print details
	print(f"Unique train dates: {unique_train_dates}, Expected train rows: {expected_train_rows}, Actual train rows: {train_rows}")
	print(f"Unique test dates: {unique_test_dates}, Expected test rows: {expected_test_rows}, Actual test rows: {test_rows}")

	# Adjust validation to allow for missing dates
	if train_rows != expected_train_rows:
		print(f"Warning: Mismatch in number of rows for X_encoder_train_df. Expected {expected_train_rows}, got {train_rows}.")
	if test_rows != expected_test_rows:
		print(f"Warning: Mismatch in number of rows for X_encoder_test_df. Expected {expected_test_rows}, got {test_rows}.")

	# Validate the number of unique dates in decoder_input and y_decoder DataFrames
	train_samples = len(decoder_input_train_df.index.get_level_values("Date").unique())
	test_samples = len(decoder_input_test_df.index.get_level_values("Date").unique())
	assert train_samples == len(y_decoder_train_df.index.get_level_values("Date").unique()), \
		"Mismatch in sample size between decoder_input_train_df and y_decoder_train_df."
	assert test_samples == len(y_decoder_test_df.index.get_level_values("Date").unique()), \
		"Mismatch in sample size between decoder_input_test_df and y_decoder_test_df."

	# Validate the number of timesteps
	train_timesteps = len(X_encoder_train_df.index.get_level_values("Step").unique())
	test_timesteps = len(X_encoder_test_df.index.get_level_values("Step").unique())
	assert train_timesteps == n_steps_in, "Mismatch in number of input timesteps (train)."
	assert test_timesteps == n_steps_in, "Mismatch in number of input timesteps (test)."

	# Validate the number of features
	assert X_encoder_train_df.shape[1] == n_features, "Mismatch in number of features (encoder input train)."
	assert decoder_input_train_df.shape[1] == n_features, "Mismatch in number of features (decoder input train)."
	assert y_decoder_train_df.shape[1] == 1, "Mismatch in number of features (decoder output train)."
	assert X_encoder_test_df.shape[1] == n_features, "Mismatch in number of features (encoder input test)."
	assert decoder_input_test_df.shape[1] == n_features, "Mismatch in number of features (decoder input test)."
	assert y_decoder_test_df.shape[1] == 1, "Mismatch in number of features (decoder output test)."

	print("Validation passed for DataFrame-based overlapping arrays.")
   
def validate_nstep_folding_of_data(n_steps_in, n_steps_out, n_features,
								   X_encoder_train, decoder_input_train, y_decoder_train, 
								   X_encoder_test, decoder_input_test, y_decoder_test):
	# Debugging: Print shapes
	print(f"Expected n_steps_in: {n_steps_in}, n_steps_out: {n_steps_out}, n_features: {n_features}")
	print(f"X_encoder_train shape: {X_encoder_train.shape}")
	print(f"decoder_input_train shape: {decoder_input_train.shape}")
	print(f"y_decoder_train shape: {y_decoder_train.shape}")
	print(f"X_encoder_test shape: {X_encoder_test.shape}")
	print(f"decoder_input_test shape: {decoder_input_test.shape}")
	print(f"y_decoder_test shape: {y_decoder_test.shape}")

	# Ensure the shapes of the data are as expected
	assert X_encoder_train.shape[0] == decoder_input_train.shape[0], "Mismatch in sample size between encoder input and decoder input."
	assert X_encoder_train.shape[0] == y_decoder_train.shape[0], "Mismatch in sample size between encoder input and decoder output."
	assert X_encoder_test.shape[0] == decoder_input_test.shape[0], "Mismatch in sample size between encoder input and decoder input."
	assert X_encoder_test.shape[0] == y_decoder_test.shape[0], "Mismatch in sample size between encoder input and decoder output."
	assert X_encoder_train.shape[1] == n_steps_in, "Mismatch in number of input time steps."
	assert X_encoder_test.shape[1] == n_steps_in, "Mismatch in number of input time steps."
	assert X_encoder_train.shape[2] == n_features, "Mismatch in number of features."
	assert X_encoder_test.shape[2] == n_features, "Mismatch in number of features."
	assert decoder_input_train.shape[1] == n_steps_out, "Mismatch in number of output time steps."
	assert decoder_input_test.shape[1] == n_steps_out, "Mismatch in number of output time steps."
	assert y_decoder_train.shape[1] == n_steps_out, "Mismatch in number of output time steps."
	assert y_decoder_test.shape[1] == n_steps_out, "Mismatch in number of output time steps."
	assert X_encoder_train.shape[2] == n_features, "Mismatch in number of features."
	assert X_encoder_test.shape[2] == n_features, "Mismatch in number of features."
	assert decoder_input_train.shape[2] == n_features, "Mismatch in number of features."
	assert decoder_input_test.shape[2] == n_features, "Mismatch in number of features."
	assert y_decoder_train.shape[2] == 1, "Mismatch in number of features."
	assert y_decoder_test.shape[2] == 1, "Mismatch in number of features."

# Example: Incremental update for 7-day moving average
# prev_avg: The moving average up to the previous day
# new_value: The new value to be included in the moving average
# oldest_value: The value to be removed from the moving average
# window_size: The number of values to consider in the moving average
def calc_incremental_nday_average(prev_avg, new_value, oldest_value, window_size):
	return (prev_avg * window_size - oldest_value + new_value) / window_size

def calc_incremental_stddev(X_test_scaled_subset, new_value, oldest_value, n):
	"""
	Incrementally calculate the standard deviation for a rolling window.

	Parameters:
		X_test_scaled_subset (list or np.ndarray): Current rolling window of values.
		new_value (float): New value to add to the rolling window.
		oldest_value (float): Oldest value to remove from the rolling window.
		n (int): Size of the rolling window.

	Returns:
		tuple: (new_std, updated_subset)
	"""
	# Ensure X_test_scaled_subset is a list
	X_test_scaled_subset = list(X_test_scaled_subset)

	# Remove the oldest value if it matches the first value in the subset
	if len(X_test_scaled_subset) > 0 and X_test_scaled_subset[0] == oldest_value:
		X_test_scaled_subset.pop(0)

	# Append the new value to the rolling window
	X_test_scaled_subset.append(new_value)

	# Calculate the standard deviation
	new_std = np.std(X_test_scaled_subset, ddof=0)

	return new_std, X_test_scaled_subset

def calc_incremental_rsi(oldest_negative_delta, oldest_positive_delta, prev_avg_loss, prev_avg_gain, curr_delta, window_size, is_scaled=False):

	print("Curr delta: ", curr_delta)
	new_avg_gain = 0
	new_avg_loss = 0
	# Calculate new average loss
	if curr_delta > 0: #no changes to avg_loss but changes to avg_gain
		act_curr_delta = 0
		new_avg_loss = -((prev_avg_loss * window_size) - oldest_negative_delta + act_curr_delta) / window_size #window_size=14
		new_avg_gain = ((prev_avg_gain * window_size) - oldest_positive_delta + curr_delta) / window_size

	# Calculate new average gain
	if curr_delta <= 0: #no changes to avg_gain but changes to avg_loss
		act_curr_delta = 0
		new_avg_gain = ((prev_avg_gain * window_size) - oldest_positive_delta + act_curr_delta) / window_size
		new_avg_loss = -((prev_avg_loss * window_size) - oldest_negative_delta + curr_delta) / window_size

	#return absolute values of avg_loss and avg_gain
	new_avg_loss = abs(new_avg_loss)
	new_avg_gain = abs(new_avg_gain)

	# Calculate RS and RSI
	rs = new_avg_gain / abs(new_avg_loss)
	if is_scaled==False:
		rsi = 100 - (100 / (1 + rs))
	else:
		rsi = 1 - (1 / (1 + rs))

	print(f"New Avg Loss: {new_avg_loss}, New Avg Gain: {new_avg_gain}, RS: {rs}, RSI: {rsi}")
	return new_avg_loss, new_avg_gain, rs, rsi

def calc_incremental_macd(curr_price, prev_ema_short, prev_ema_long, prev_signal_line, n_short=12, n_long=26, n_signal=9):
	 # Calculate smoothing factors
	k_short = 2 / (n_short + 1)
	k_long = 2 / (n_long + 1)
	k_signal = 2 / (n_signal + 1)

	# Update short-term and long-term EMAs
	ema_short = curr_price * k_short + prev_ema_short * (1 - k_short)
	ema_long = curr_price * k_long + prev_ema_long * (1 - k_long)

	# Calculate MACD
	macd = ema_short - ema_long

	# Update Signal Line
	signal_line = macd * k_signal + prev_signal_line * (1 - k_signal)

	# Return the updated values
	return ema_short, ema_long, macd, signal_line

def calc_incremental_bollinger_bands(stock_data, curr_date, target_column, window_size = 20):

		date_index = stock_data.index.get_loc(curr_date)

		# # Calculate rolling mean and standard deviation using pandas, backwards from the row at date_index.
		rolling_mean = stock_data[target_column].rolling(window=window_size).mean().iloc[date_index]
		rolling_std = stock_data[target_column].rolling(window=window_size).std(ddof=0).iloc[date_index]

		# Calculate Bollinger Bands
		bollinger_upper = rolling_mean + (rolling_std * 2)
		bollinger_lower = rolling_mean - (rolling_std * 2)

		return rolling_mean, rolling_std, bollinger_lower, bollinger_upper

def plot_averages(stock_data, ticker, axes):
	
	# Plot the main columns
	for i, column in enumerate(['log_returns']): #, 'Vol_7', 'Vol_30', 'Adj Close', '7d_avg', '30d_avg'
		if column in stock_data.columns:
			color = 'red' if column == 'Adj Close' else 'grey'  # Triple the thickness for 'Adj Close'
			axes.plot(stock_data.index, stock_data[column], label=column, color = color)
			# Add y-label next to the last point of the line
			y_pos = stock_data[column].iloc[-1]
			# Spread the y-labels by adding an offset based on the index
			y_offset = 0.15 * y_pos * ((i % 4) - 1.5)  # Dynamic offsets for better spacing
			axes.annotate(
				column,
				xy=(stock_data.index[-1], y_pos),
				xytext=(stock_data.index[-1] + pd.Timedelta(days=15), y_pos + y_offset),  # Adjusted horizontal and vertical gap
				arrowprops=dict(arrowstyle="->", color='black'),
				fontsize=9,
				verticalalignment='center'
			)

	axes.set_xlabel('Date')
	axes.set_ylabel('Value')
	axes.set_title(f'{ticker} Stock Data')
	axes.legend(loc='upper left')
	axes.grid()

def plot_bollinger_bands(stock_data, ticker, axes):
	# Plot Bollinger Bands as shaded areas
	column = 'Adj Close'
	if column in stock_data.columns:
		axes.plot(stock_data.index, stock_data[column], label=column, linewidth=1, color='yellow')

	column = 'Sig_L'
	if column in stock_data.columns:
		axes.plot(stock_data.index, stock_data[column], label=column, linewidth=1, color='blue')

	if 'B_U' in stock_data.columns and 'B_L' in stock_data.columns:
		axes.fill_between(stock_data.index, stock_data['B_U'], stock_data['B_L'], color='gray', alpha=0.3, label='Bollinger Bands')

	axes.set_xlabel('Date')
	axes.set_ylabel('Value')
	axes.set_title(f'{ticker} Stock Data')
	axes.legend(loc='upper left')
	axes.grid()

def get_rolling_subset(step, window_size, data, predicted_data):
	num_predicted_values = max(1, min(step, window_size))  # Ensure at least 1 predicted value
	num_historical_values = window_size - num_predicted_values  # Remaining elements to take from data
	
	# num_historical_values = len(data) 
	# Take the required number of historical values from data
	if data.ndim == 1:
		historical_values = data[-num_historical_values:] if num_historical_values > 0 else np.array([])
	else:
		historical_values = data[-num_historical_values:, 0] if num_historical_values > 0 else np.array([])
	#drop the oldest values from the historical values
	# if step > 0:
	# 	historical_values = historical_values[step:]
	# Take the required number of predicted values
	predicted_values = predicted_data[:num_predicted_values] if num_predicted_values > 0 else np.array([])
	# #drop the oldest values from the historical values
	# if num_historical_values_dropped > 0:
	# 	historical_values = historical_values[num_historical_values_dropped:]

	# Combine historical and predicted values
	# Ensure both arrays have the same number of dimensions
	if historical_values.ndim == 1:
		historical_values = np.expand_dims(historical_values, axis=-1)
	predicted_values = np.array(predicted_values)  # Ensure it's a NumPy array
	if predicted_values.ndim == 1:
		predicted_values = np.expand_dims(predicted_values, axis=-1)

	rolling_subset = np.concatenate([historical_values, predicted_values], axis=0)

	# Flatten rolling_subset_20 to 1D
	rolling_subset = rolling_subset.flatten()

	return rolling_subset

# Update features after prediction for Seq2Seq model - uses 3D array instead of 2D array
def update_features_after_prediction_seq2seq_old(predicted_value, last_row_unscaled, n_steps_in, previous_adj_close, step, X_test_unscaled, y_test_unscaled, predicted_unscaled,  column_names):
		print("Predicted value:", predicted_value)
		print("Shape of last_row:", last_row_unscaled.shape)
		print("Last row scaled:", last_row_unscaled[0, n_steps_in-1, :])
		# Calculate daily returns
		new_daily_return = (predicted_value - previous_adj_close) / previous_adj_close
		print(f"New daily return: {new_daily_return}, Previous Adj Close: {previous_adj_close}, Predicted Value: {predicted_value}")

		window_size = 7
		rolling_subset_7 = get_rolling_subset(step, window_size, y_test_unscaled, predicted_unscaled)
		new_volatility_7 = np.std(rolling_subset_7, ddof=0)
		print("New 7-day volatility:", new_volatility_7)
		# Calculate 7-day average
		new_7_day_avg = np.mean(rolling_subset_7)
		print("New 7-day average:", new_7_day_avg)

		# Calculate rolling standard deviation (volatility): Volatility_30
		# idx = column_names.index('Vol_30')
		# Extract the last 29 values from X_test_unscaled and append the predicted value
		# rolling_subset_30 = np.append(y_test_unscaled[-30 + step:-1 + step, 0], predicted_value)
		window_size = 30
		rolling_subset_30 = get_rolling_subset(step, window_size, y_test_unscaled, predicted_unscaled)
		new_volatility_30 = np.std(rolling_subset_30, ddof=0)
		print("New 30-day volatility:", new_volatility_30)
		# Calculate 30-day average
		new_30_day_avg = np.mean(rolling_subset_30)
		print("New 30-day average:", new_30_day_avg)

		# Calculate Relative Strength Index (RSI)
		# oldest_negative_delta, oldest_positive_delta, prev_avg_loss, prev_avg_gain, curr_delta
		idx = column_names.index('delta')
		oldest_delta = X_test_unscaled[-14+step,  n_steps_in-1, idx]
		if oldest_delta < 0:
			oldest_positive_delta = 0
			oldest_negative_delta = oldest_delta
		else:
			oldest_positive_delta = oldest_delta
			oldest_negative_delta = 0

		idx = column_names.index('avg_gain')
		prev_avg_loss = last_row_unscaled[0,n_steps_in-1,idx] 
		idx = column_names.index('avg_loss')
		prev_avg_gain = last_row_unscaled[0,n_steps_in-1,idx]  
		curr_delta = predicted_value - previous_adj_close  # Current delta
		is_scaled = False
		window_size = 12
		new_avg_loss, new_avg_gain, new_rs, new_rsi = calc_incremental_rsi(oldest_negative_delta, oldest_positive_delta, prev_avg_loss, prev_avg_gain, curr_delta, window_size, is_scaled)
		print("New Avg Loss:", new_avg_loss)
		print("New Avg Gain:", new_avg_gain)
		print("New RS:", new_rs)
		print("New RSI:", new_rsi)
		print("Shape of new_rsi:", new_rsi.shape)
		print("Shape of new_rs:", new_rs.shape)
		print("Shape of new_avg_loss:", new_avg_loss.shape)
		print("Shape of new_avg_gain:", new_avg_gain.shape)

		# Calculate the Moving Average Convergence Divergence (MACD)
		# ema_short, ema_long, macd, signal_line
		#curr_price, prev_ema_short, prev_ema_long, prev_signal_line, n_short=12, n_long=26, n_signal=9
		curr_price = predicted_value
		idx = column_names.index('ema_12')
		prev_ema_short = last_row_unscaled[0,n_steps_in-1,idx]  
		idx = column_names.index('ema_26')
		prev_ema_long = last_row_unscaled[0,n_steps_in-1,idx] 
		idx = column_names.index('Sig_L')
		prev_signal_line = last_row_unscaled[0,n_steps_in-1,idx]
		ema_short, ema_long, new_macd, new_signal = calc_incremental_macd(curr_price, prev_ema_short, prev_ema_long, prev_signal_line, 12, 26, 9)
		print("New EMA Short:", ema_short)
		print("New EMA Long:", ema_long)
		print("New MACD:", new_macd)
		print("New Signal Line:", new_signal)
		print("Shape of new_macd:", new_macd.shape)
		print("Shape of new_signal:", new_signal.shape)
		print("Shape of new_ema_short:", ema_short.shape)
		print("Shape of new_ema_long:", ema_long.shape)

		# Calculate Bollinger Bands
		# rolling_mean, rolling_std, bollinger_lower, bollinger_upper = calc_incremental_bollinger_bands(train['Tgt'].values[-20:], predicted_value, last_row[0], 20)
		# # Calculate rolling mean and standard deviation using pandas, backwards from the row at date_index.
		window_size = 20
		rolling_subset_20 = get_rolling_subset(step, window_size, y_test_unscaled, predicted_unscaled)

		# Calculate rolling mean and standard deviation
		rolling_mean = pd.Series(rolling_subset_20).rolling(window=window_size).mean()
		rolling_std = pd.Series(rolling_subset_20).rolling(window=window_size).std(ddof=0)
		
		# Calculate Bollinger Bands
		bollinger_upper = rolling_mean + (rolling_std * 2)
		bollinger_lower = rolling_mean - (rolling_std * 2)

		# Extract the last rolling mean and standard deviation values
		last_rolling_mean = rolling_mean.iloc[-1].item()
		last_rolling_std = rolling_std.iloc[-1].item()
		last_bollinger_upper = bollinger_upper.iloc[-1].item()
		last_bollinger_lower = bollinger_lower.iloc[-1].item()

		# Print the results for debugging
		print(f"Last Rolling Mean: {last_rolling_mean}")
		print(f"Last Rolling Std: {last_rolling_std}")
		print(f"Last Bollinger Upper: {last_bollinger_upper}")
		print(f"Last Bollinger Lower: {last_bollinger_lower}")

		 # Update `last_row` with the new features
		 # note that these are all scaled values since they are calculated from the scaled data
		last_row_unscaled = np.array(
			[new_7_day_avg, new_30_day_avg, new_daily_return, new_volatility_7, new_volatility_30,
			 curr_delta, new_avg_gain, new_avg_loss, new_rs, new_rsi, 
			 ema_short, ema_long, new_macd, new_signal,
			 last_bollinger_upper, last_bollinger_lower, last_rolling_mean,last_rolling_std])
	
		#create a 3D array with the last_row_scaled as the first row and the rest of the rows as 0
		last_row_scaled_3D = np.zeros((1, n_steps_in, len(column_names)))
		#copy the last_row_scaled to the last row of the last_row_scaled_3D
		last_row_scaled_3D[0, n_steps_in-1, :] = last_row_unscaled
		return last_row_scaled_3D

#lstm version of the function - uses 2D array instead of 3D
def update_features_after_prediction_lstm_old(predicted_value_unscaled, last_row_unscaled, previous_adj_close, step, X_test_unscaled, y_test_unscaled, predicted_unscaled,  column_names):
		print("Shape of last_row_unscaled:", last_row_unscaled.shape)
		# Calculate daily returns
		new_daily_return = (predicted_value_unscaled - previous_adj_close) / previous_adj_close

		window_size = 7
		rolling_subset_7 = get_rolling_subset(step, window_size, y_test_unscaled, predicted_unscaled)
		new_volatility_7 = np.std(rolling_subset_7, ddof=0)
		print("New 7-day volatility:", new_volatility_7)
		# Calculate 7-day average
		new_7_day_avg = np.mean(rolling_subset_7)
		print("New 7-day average:", new_7_day_avg)

		# Calculate rolling standard deviation (volatility): Volatility_30
		# idx = column_names.index('Vol_30')
		# Extract the last 29 values from X_test_unscaled and append the predicted value
		# rolling_subset_30 = np.append(y_test_unscaled[-30 + step:-1 + step, 0], predicted_value)
		window_size = 30
		rolling_subset_30 = get_rolling_subset(step, window_size, y_test_unscaled, predicted_unscaled)
		new_volatility_30 = np.std(rolling_subset_30, ddof=0)
		print("New 30-day volatility:", new_volatility_30)
		# Calculate 30-day average
		new_30_day_avg = np.mean(rolling_subset_30)
		print("New 30-day average:", new_30_day_avg)

		# # Calculate Relative Strength Index (RSI(6))
		window_size = 6
		idx = column_names.index('delta_1')
		oldest_delta_1 = X_test_unscaled[-window_size+step, idx]
		if oldest_delta_1 < 0:
			oldest_positive_delta_1 = 0
			oldest_negative_delta_1 = oldest_delta_1
		else:
			oldest_positive_delta_1 = oldest_delta_1
			oldest_negative_delta_1 = 0

		idx = column_names.index('avg_gain_1')
		prev_avg_loss_1 = last_row_unscaled[idx] 
		idx = column_names.index('avg_loss_1')
		prev_avg_gain_1 = last_row_unscaled[idx]  
		curr_delta_1 = predicted_value_unscaled - previous_adj_close  # Current delta
		is_scaled = False
		new_avg_loss_1, new_avg_gain_1, new_rs_1, new_rsi_1 = calc_incremental_rsi(oldest_negative_delta_1, oldest_positive_delta_1, prev_avg_loss_1, prev_avg_gain_1, curr_delta_1, window_size, is_scaled)

		# Calculate Relative Strength Index (RSI(12))
		window_size = 12
		idx = column_names.index('delta_2')
		oldest_delta_2 = X_test_unscaled[-window_size+step, idx] #14
		if oldest_delta_2 < 0:
			oldest_positive_delta_2 = 0
			oldest_negative_delta_2 = oldest_delta_2
		else:
			oldest_positive_delta_2 = oldest_delta_2
			oldest_negative_delta_2 = 0

		idx = column_names.index('avg_gain_2')
		prev_avg_loss_2 = last_row_unscaled[idx] 
		idx = column_names.index('avg_loss_2')
		prev_avg_gain_2 = last_row_unscaled[idx]  
		curr_delta_2 = predicted_value_unscaled - previous_adj_close  # Current delta
		is_scaled = False
		new_avg_loss_2, new_avg_gain_2, new_rs_2, new_rsi_2 = calc_incremental_rsi(oldest_negative_delta_2, oldest_positive_delta_2, prev_avg_loss_2, prev_avg_gain_2, curr_delta_2, window_size, is_scaled)

		# Calculate the Moving Average Convergence Divergence (MACD(6,15,6))
		# ema_short, ema_long, macd, signal_line
		#curr_price, prev_ema_short, prev_ema_long, prev_signal_line, n_short=6, n_long=15, n_signal=6
		curr_price = predicted_value_unscaled
		idx = column_names.index('ema_6')
		prev_ema_short_6 = last_row_unscaled[idx]  
		idx = column_names.index('ema_15')
		prev_ema_long_15 = last_row_unscaled[idx] 
		idx = column_names.index('Sig_L_1')
		prev_signal_line_1 = last_row_unscaled[idx]
		ema_short_6, ema_long_15, new_macd_1, new_signal_1 = calc_incremental_macd(curr_price, prev_ema_short_6, prev_ema_long_15, prev_signal_line_1, 6, 15, 6)

		# Calculate the Moving Average Convergence Divergence (MACD(12,26,9))
		# ema_short, ema_long, macd, signal_line
		#curr_price, prev_ema_short, prev_ema_long, prev_signal_line, n_short=12, n_long=26, n_signal=9
		curr_price = predicted_value_unscaled
		idx = column_names.index('ema_12')
		prev_ema_short_12 = last_row_unscaled[idx]  
		idx = column_names.index('ema_26')
		prev_ema_long_26 = last_row_unscaled[idx] 
		idx = column_names.index('Sig_L_2')
		prev_signal_line_2 = last_row_unscaled[idx]
		ema_short_12, ema_long_26, new_macd_2, new_signal_2 = calc_incremental_macd(curr_price, prev_ema_short_12, prev_ema_long_26, prev_signal_line_2, 12, 26, 9)

		window_size = 20
		rolling_subset_20 = get_rolling_subset(step, window_size, y_test_unscaled, predicted_unscaled)

		# Calculate rolling mean and standard deviation
		rolling_mean = pd.Series(rolling_subset_20).rolling(window=window_size).mean()
		rolling_std = pd.Series(rolling_subset_20).rolling(window=window_size).std(ddof=0)
		
		# Calculate Bollinger Bands
		bollinger_upper = rolling_mean + (rolling_std * 2)
		bollinger_lower = rolling_mean - (rolling_std * 2)

		# Extract the last rolling mean and standard deviation values
		last_rolling_mean = rolling_mean.iloc[-1].item()
		last_rolling_std = rolling_std.iloc[-1].item()
		last_bollinger_upper = bollinger_upper.iloc[-1].item()
		last_bollinger_lower = bollinger_lower.iloc[-1].item()

		# Print the results for debugging
		print(f"Last Rolling Mean: {last_rolling_mean}")
		print(f"Last Rolling Std: {last_rolling_std}")
		print(f"Last Bollinger Upper: {last_bollinger_upper}")
		print(f"Last Bollinger Lower: {last_bollinger_lower}")

		#debugging
		# print("Shape of new_7_day_avg:", new_7_day_avg.shape)
		# print("Shape of new_30_day_avg:", new_30_day_avg.shape)
		# print("Shape of new_daily_return:", new_daily_return.shape)
		# print("Shape of new_volatility_7:", new_volatility_7.shape)
		# print("Shape of new_volatility_30:", new_volatility_30.shape)
		# print("Shape of curr_delta:", curr_delta.shape)
		# print("Shape of new_avg_gain:", new_avg_gain.shape)
		# print("Shape of new_avg_loss:", new_avg_loss.shape)
		# print("Shape of new_rs:", new_rs.shape)
		# print("Shape of new_rsi:", new_rsi.shape)
		# print("Shape of ema_short:", ema_short.shape)
		# print("Shape of ema_long:", ema_long.shape)
		# print("Shape of new_macd:", new_macd.shape)
		# print("Shape of new_signal:", new_signal.shape)
		# print("Shape of last_bollinger_upper:", last_bollinger_upper.shape)
		# print("Shape of last_bollinger_lower:", last_bollinger_lower.shape)
		# print("Shape of last_rolling_mean:", last_rolling_mean.shape)
		# print("Shape of last_rolling_std:", last_rolling_std.shape)

		 # Update `last_row` with the new features
		 # note that these are all scaled values since they are calculated from the scaled data
		last_row_unscaled = np.array(
			[new_7_day_avg, new_30_day_avg, new_daily_return, new_volatility_7, new_volatility_30,
			 curr_delta_1, new_avg_gain_1, new_avg_loss_1, new_rs_1, new_rsi_1,
			 curr_delta_2, new_avg_gain_2, new_avg_loss_2, new_rs_2, new_rsi_2, 
			 ema_short_12, ema_long_26, new_macd_1, new_signal_1,
			 ema_short_6, ema_long_15, new_macd_2, new_signal_2,
			 last_bollinger_upper, last_bollinger_lower, last_rolling_mean,last_rolling_std
			])
	
		return last_row_unscaled

def get_scalar(val):
	if isinstance(val, np.ndarray):
		return val.item() if val.size == 1 else float(val.flatten()[0])
	return float(val)
# Helper to extract the correct value for both 2D and 3D arrays
def get_feature_value(arr, row_idx, timestep_idx, feature_idx):
	if arr.ndim == 3:
		return get_scalar(arr[row_idx, timestep_idx, feature_idx])
	elif arr.ndim == 2:
		return get_scalar(arr[row_idx, feature_idx])
	else:
		raise ValueError(f"Unsupported array shape: {arr.shape}")
# Helper for last_row_unscaled:
def get_last_row_feature(arr, timestep_idx, feature_idx):
	if arr.ndim == 3:
		return get_scalar(arr[0, timestep_idx, feature_idx])
	elif arr.ndim == 2:
		return get_scalar(arr[feature_idx])
	elif arr.ndim == 1:
		return get_scalar(arr[feature_idx])
	else:
		raise ValueError(f"Unsupported array shape: {arr.shape}")

def compute_updated_features(
	predicted_value,
	last_row_unscaled,
	previous_adj_close,
	step,
	X_test_unscaled,
	y_test_unscaled,
	predicted_unscaled,
	column_names,
	n_steps_in=1,
	rsi_windows=[('delta_1', 'avg_gain_1', 'avg_loss_1', 6), ('delta_2', 'avg_gain_2', 'avg_loss_2', 12)],
	macd_windows=[('ema_6', 'ema_15', 'Sig_L_1', 6, 15, 6), ('ema_12', 'ema_26', 'Sig_L_2', 12, 26, 9)],
	bollinger_window=20,
	return_shape='2d'
):
	# Calculate daily return
	new_daily_return = (predicted_value - previous_adj_close) / previous_adj_close

	# Rolling averages and volatilities
	rolling_subset_7 = get_rolling_subset(step, 7, y_test_unscaled, predicted_unscaled)
	new_volatility_7 = np.std(rolling_subset_7, ddof=0)
	new_7_day_avg = np.mean(rolling_subset_7)

	rolling_subset_30 = get_rolling_subset(step, 30, y_test_unscaled, predicted_unscaled)
	new_volatility_30 = np.std(rolling_subset_30, ddof=0)
	new_30_day_avg = np.mean(rolling_subset_30)

	# RSI calculations
	rsi_results = []
	for delta_col, avg_gain_col, avg_loss_col, window_size in rsi_windows:
		idx_delta = column_names.index(delta_col)
		# oldest_delta = X_test_unscaled[-window_size+step, idx_delta]
		row_idx = -window_size + step
		timestep_idx = n_steps_in - 1
		try:
			oldest_delta = get_feature_value(X_test_unscaled, row_idx, timestep_idx, idx_delta)
		except IndexError:
			# Fallback: use first available row or 0.0
			oldest_delta = 0.0

		if oldest_delta < 0:
			oldest_positive_delta = 0
			oldest_negative_delta = oldest_delta
		else:
			oldest_positive_delta = oldest_delta
			oldest_negative_delta = 0

		idx_gain = column_names.index(avg_gain_col)
		prev_avg_loss = get_last_row_feature(last_row_unscaled, timestep_idx, idx_gain)
		idx_loss = column_names.index(avg_loss_col)
		prev_avg_gain = get_last_row_feature(last_row_unscaled, timestep_idx, idx_loss)
		curr_delta = predicted_value - previous_adj_close
		is_scaled = False
		new_avg_loss, new_avg_gain, new_rs, new_rsi = calc_incremental_rsi(
			oldest_negative_delta, oldest_positive_delta, prev_avg_loss, prev_avg_gain, curr_delta, window_size, is_scaled
		)
		rsi_results.append((curr_delta, new_avg_gain, new_avg_loss, new_rs, new_rsi))

	# MACD calculations
	macd_results = []
	for ema_short_col, ema_long_col, sig_col, n_short, n_long, n_signal in macd_windows:
		idx_short = column_names.index(ema_short_col)
		prev_ema_short = get_last_row_feature(last_row_unscaled, timestep_idx, idx_loss)
		# prev_ema_short = last_row_unscaled[idx_short]
		idx_long = column_names.index(ema_long_col)
		# prev_ema_long = last_row_unscaled[idx_long]
		prev_ema_long = get_last_row_feature(last_row_unscaled, timestep_idx, idx_loss)
		idx_sig = column_names.index(sig_col)
		# prev_signal_line = last_row_unscaled[idx_sig]
		prev_signal_line = get_last_row_feature(last_row_unscaled, timestep_idx, idx_loss)
		ema_short, ema_long, new_macd, new_signal = calc_incremental_macd(
			predicted_value, prev_ema_short, prev_ema_long, prev_signal_line, n_short, n_long, n_signal
		)
		macd_results.append((ema_short, ema_long, new_macd, new_signal))

	# Bollinger Bands
	rolling_subset_20 = get_rolling_subset(step, bollinger_window, y_test_unscaled, predicted_unscaled)
	rolling_mean = pd.Series(rolling_subset_20).rolling(window=bollinger_window).mean()
	rolling_std = pd.Series(rolling_subset_20).rolling(window=bollinger_window).std(ddof=0)
	bollinger_upper = rolling_mean + (rolling_std * 2)
	bollinger_lower = rolling_mean - (rolling_std * 2)
	last_rolling_mean = rolling_mean.iloc[-1].item()
	last_rolling_std = rolling_std.iloc[-1].item()
	last_bollinger_upper = bollinger_upper.iloc[-1].item()
	last_bollinger_lower = bollinger_lower.iloc[-1].item()

	# Compose the feature vector
	features = [
		new_7_day_avg, new_30_day_avg, new_daily_return, new_volatility_7, new_volatility_30,
	]
	# Add RSI features
	for rsi in rsi_results:
		features.extend(rsi)
	# Add MACD features
	for macd in macd_results:
		features.extend(macd)
	# Add Bollinger Bands
	features.extend([last_bollinger_upper, last_bollinger_lower, last_rolling_mean, last_rolling_std])

	if return_shape == '3d':
		arr = np.zeros((1, n_steps_in, len(features)))
		arr[0, n_steps_in-1, :] = features
		return arr
	else:
		return np.array(features)

# --- Refactor the two update functions to use this common function ---

def update_features_after_prediction_lstm(
	predicted_value_unscaled, last_row_unscaled, previous_adj_close, step,
	X_test_unscaled, y_test_unscaled, predicted_unscaled, column_names
):
	return compute_updated_features(
		predicted_value_unscaled, last_row_unscaled, previous_adj_close, step,
		X_test_unscaled, y_test_unscaled, predicted_unscaled, column_names,
		n_steps_in=1, return_shape='2d'
	)

def update_features_after_prediction_seq2seq(
	predicted_value, last_row_unscaled, n_steps_in, previous_adj_close, step,
	X_test_unscaled, y_test_unscaled, predicted_unscaled, column_names
):
	return compute_updated_features(
		predicted_value, last_row_unscaled, previous_adj_close, step,
		X_test_unscaled, y_test_unscaled, predicted_unscaled, column_names,
		n_steps_in=n_steps_in, return_shape='3d'
	)

def validate_data_processing(X_train, X_test, y_train, y_test, feature_columns, target_column):
	"""
	Validate the integrity of processed data.

	Parameters:
		X_train (numpy.ndarray): Training data for features.
		X_test (numpy.ndarray): Testing data for features.
		y_train (numpy.ndarray): Training data for the target variable.
		y_test (numpy.ndarray): Testing data for the target variable.
		feature_columns (list): List of feature column names.
		target_column (str): Name of the target column.

	Returns:
		None
	"""
	import numpy as np

	print("\n--- Validating Data Processing ---")

	# 1. Check for missing values
	if np.isnan(X_train).any() or np.isnan(X_test).any():
		print("Error: NaN values found in feature data.")
	else:
		print("No NaN values found in feature data.")

	if np.isnan(y_train).any() or np.isnan(y_test).any():
		print("Error: NaN values found in target data.")
	else:
		print("No NaN values found in target data.")

	# 2. Check shapes of the data
	print(f"X_train shape: {X_train.shape}")
	print(f"X_test shape: {X_test.shape}")
	print(f"y_train shape: {y_train.shape}")
	print(f"y_test shape: {y_test.shape}")

	# 3. Check feature and target column consistency
	if X_train.shape[2] != len(feature_columns):
		print(f"Error: Number of features in X_train ({X_train.shape[2]}) does not match feature_columns ({len(feature_columns)}).")
	else:
		print("Feature columns match the expected number of features.")

	if y_train.shape[1] != 1:
		print(f"Error: Target variable y_train should have a single column, but found {y_train.shape[1]}.")
	else:
		print("Target column shape is valid.")

	print("--- Data Processing Validation Complete ---\n")

def validate_feature_scalers(scaled_data, feature_columns, target_column, scalers_dict, feature_axis=2):
	"""
	Validate the application of scalers to multiple feature columns and the target column.

	Parameters:
		scaled_data (np.ndarray): Scaled data for the features and target variable.
		feature_columns (list): List of feature column names.
		target_column (str): Name of the target column.
		scalers_dict (dict): Dictionary of scalers for each feature and the target variable.
		feature_axis (int): Axis of the array that corresponds to the features (default is 2).

	Returns:
		None
	"""
	print("\n--- Validating Scalers for Features and Target ---")

	# Validate feature columns
	for i, feature in enumerate(feature_columns):
		scaler = scalers_dict.get(feature)
		if scaler is None:
			print(f"Error: No scaler found for feature '{feature}'.")
			continue

		# Extract the feature data along the specified axis
		feature_data = np.take(scaled_data, indices=i, axis=feature_axis)
		feature_min, feature_max = feature_data.min(), feature_data.max()
		print(f"Feature '{feature}': Min={feature_min:.4f}, Max={feature_max:.4f}")

		# # Deeper Validations based on scaler type
		# if isinstance(scaler, MinMaxScaler):
		#     assert 0 <= feature_min <= 1 and 0 <= feature_max <= 1, f"Feature '{feature}' out of range [0, 1]."
		# elif isinstance(scaler, StandardScaler):
		#     feature_mean, feature_std = feature_data.mean(), feature_data.std()
		#     print(f"Feature '{feature}': Mean={feature_mean:.4f}, Std={feature_std:.4f}")
		#     assert abs(feature_mean) < 1e-2, f"Feature '{feature}' mean not close to 0."
		#     assert abs(feature_std - 1) < 1e-3, f"Feature '{feature}' std not close to 1."
		# elif isinstance(scaler, RobustScaler):
		#     feature_median = np.median(feature_data)
		#     print(f"Feature '{feature}': Median={feature_median:.4f}")
		#     # Relaxed assertion to account for small numerical deviations
		#     assert -1.1 <= feature_median <= 1.1, f"Feature '{feature}' median not centered around 0."

	# Validate target column
	target_scaler = scalers_dict.get(target_column)
	if target_scaler is None:
		print(f"Error: No scaler found for target '{target_column}'.")
	else:
		# Assuming the target column is the last axis
		target_data = scaled_data[:, :, -1] if feature_axis == 2 else scaled_data[:, -1]
		target_min, target_max = target_data.min(), target_data.max()
		print(f"Target '{target_column}': Min={target_min:.4f}, Max={target_max:.4f}")

		# # Deeper Validations based on scaler type
		# if isinstance(target_scaler, MinMaxScaler):
		#     assert 0 <= target_min <= 1 and 0 <= target_max <= 1, f"Target '{target_column}' out of range [0, 1]."
		# elif isinstance(target_scaler, StandardScaler):
		#     target_mean, target_std = target_data.mean(), target_data.std()
		#     print(f"Target '{target_column}': Mean={target_mean:.4f}, Std={target_std:.4f}")
		#     assert abs(target_mean) < 1e-3, f"Target '{target_column}' mean not close to 0."
		#     assert abs(target_std - 1) < 1e-3, f"Target '{target_column}' std not close to 1."
		# elif isinstance(target_scaler, RobustScaler):
		#     target_median = np.median(target_data)
		#     print(f"Target '{target_column}': Median={target_median:.4f}")
		#     assert -1.01 <= target_median <= 1.01, f"Target '{target_column}' median not centered around 0."

	#draw a histogram+kde plot of the scaled data
	#draw each feature in a separate subplot. draw 3 features in each row
	import seaborn as sns
	import matplotlib.pyplot as plt
	num_features = len(feature_columns)
	num_rows = (num_features + 2) // 3
	fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
	for i, feature in enumerate(feature_columns):
		ax = axes[i // 3, i % 3]
		sns.histplot(scaled_data[:, :, i], kde=True, ax=ax)
		ax.set_title(f"Scaled Feature: {feature}")
		ax.set_xlabel("Value")
		ax.set_ylabel("Density")
		ax.grid()
	plt.tight_layout()
	plt.show()
	print("--- Validation Complete ---\n")

def validate_scalers_entire_dataset(scaled_data, scaler, feature_columns, feature_axis=2):
	"""
	Validate the scaling of the entire dataset using a single scaler.

	Parameters:
		scaled_data (np.ndarray): Scaled dataset (features and target combined).
		scaler (object): The scaler used for scaling the dataset (e.g., MinMaxScaler, StandardScaler, RobustScaler).
		feature_columns (list): List of feature column names.

	Returns:
		None
	"""
	print("\n--- Validating Scaling of the Entire Dataset ---")

	# Validate the entire dataset
	dataset_min = scaled_data.min()
	dataset_max = scaled_data.max()
	dataset_mean = scaled_data.mean()
	dataset_std = scaled_data.std()

	print(f"Dataset Min: {dataset_min:.4f}")
	print(f"Dataset Max: {dataset_max:.4f}")
	print(f"Dataset Mean: {dataset_mean:.4f}")
	print(f"Dataset Std: {dataset_std:.4f}")

	# Check if the dataset is scaled correctly based on the scaler type
	if isinstance(scaler, MinMaxScaler):
		assert 0 <= dataset_min <= 1 + 1e-8 and 0 <= dataset_max <= 1 + 1e-8, \
			f"Dataset out of range [0, 1] for MinMaxScaler. Min={dataset_min}, Max={dataset_max}"
	elif isinstance(scaler, StandardScaler):
		assert abs(dataset_mean) < 1e-2, f"Dataset mean not close to 0 for StandardScaler. Mean={dataset_mean}"
		assert abs(dataset_std - 1) < 1e-2, f"Dataset std not close to 1 for StandardScaler. Std={dataset_std}"
	elif isinstance(scaler, RobustScaler):
		dataset_median = np.median(scaled_data)
		print(f"Dataset Median: {dataset_median:.4f}")
		assert -1.1 <= dataset_median <= 1.1, \
			f"Dataset median not centered around 0 for RobustScaler. Median={dataset_median}"

#    #draw a histogram+kde plot of the scaled data for debugging
#     #draw each feature in a separate subplot. draw 3 features in each row
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     num_features = len(feature_columns)
#     num_rows = (num_features + 2) // 3
#     fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
#     for i, feature in enumerate(feature_columns):
#         print("Feature:", feature)
#         ax = axes[i // 3, i % 3]
#         print("Shape of scaled data:", scaled_data.shape)
#         sns.histplot(np.take(scaled_data, indices=i, axis=feature_axis).flatten(), kde=True, ax=ax)
#         ax.set_title(f"Scaled Feature: {feature}")
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Density")
#         ax.grid()
#     plt.tight_layout()
#     plt.show()

	print("--- Dataset Scaling Validation Complete ---\n")

def plot_attention_weights(model, X_encoder_test):
	# Extract attention weights after training
	attention_layer = model.get_layer("attention_layer")  # Replace with the actual name of your attention layer
	attention_model = Model(inputs=model.input, outputs=attention_layer.output)

	# Pass a sample input to the model
	sample_input = X_encoder_test[:1]  # Use the first sample from the test set
	attention_weights = attention_model.predict([sample_input])

	# Visualize attention weights
	import matplotlib.pyplot as plt
	plt.imshow(attention_weights[0], cmap='viridis', aspect='auto')
	plt.colorbar()
	plt.title("Attention Weights")
	plt.xlabel("Input Timesteps")
	plt.ylabel("Output Timesteps")
	plt.show()

def generate_signals(stock_data):
	"""
	Generate buy/hold/sell signals based on engineered features and create a separate DataFrame for signals.

	Parameters:
		stock_data (pd.DataFrame): DataFrame containing stock data with engineered features.

	Returns:
		pd.DataFrame: DataFrame with buy/hold/sell signals.
	"""
	# Initialize a column for signals
	stock_data['Signal'] = 'Hold'  # Default signal is 'Hold'

	# 1. Bollinger Bands
	stock_data.loc[stock_data['Adj Close'] < stock_data['B_L'], 'Signal'] = 'Buy'
	stock_data.loc[stock_data['Adj Close'] > stock_data['B_U'], 'Signal'] = 'Sell'

	# 2. RSI (Relative Strength Index)
	stock_data.loc[stock_data['RSI'] < 30, 'Signal'] = 'Buy'
	stock_data.loc[stock_data['RSI'] > 70, 'Signal'] = 'Sell'

	# 3. MACD (Moving Average Convergence Divergence)
	stock_data.loc[(stock_data['MACD'] > stock_data['Sig_L']) & (stock_data['MACD'].shift(1) <= stock_data['Sig_L'].shift(1)), 'Signal'] = 'Buy'
	stock_data.loc[(stock_data['MACD'] < stock_data['Sig_L']) & (stock_data['MACD'].shift(1) >= stock_data['Sig_L'].shift(1)), 'Signal'] = 'Sell'

	# 4. Donchian Bands
	stock_data.loc[stock_data['Adj Close'] > stock_data['Don_U'], 'Signal'] = 'Buy'
	stock_data.loc[stock_data['Adj Close'] < stock_data['Don_L'], 'Signal'] = 'Sell'

	# 5. Keltner Channels
	stock_data.loc[stock_data['Adj Close'] > stock_data['Kelt_U'], 'Signal'] = 'Buy'
	stock_data.loc[stock_data['Adj Close'] < stock_data['Kelt_L'], 'Signal'] = 'Sell'

	# 6. Combined Signal (Optional)
	stock_data['Signal'] = stock_data['Signal'].replace({'Buy': 1, 'Sell': -1, 'Hold': 0})
	stock_data['Combined_Signal'] = stock_data[['Signal']].sum(axis=1)
	stock_data['Combined_Signal'] = stock_data['Combined_Signal'].apply(lambda x: 'Buy' if x > 0 else ('Sell' if x < 0 else 'Hold'))

	# Create a separate DataFrame for signals
	signals_df = stock_data[['Signal', 'Combined_Signal']].copy()
	signals_df['Date'] = stock_data.index  # Add the date index for plotting

	return signals_df

def get_ticker_datafile_from_local_datastore(ticker, start_date, end_date, local_repo_folder, source_files_folder):
#check local datastore/archive file to see if the ticker data for the date range provided is available in the local datastore.
#if available, read the data from the local datastore and return it as a dataframe. The local datastore is a folder in the local file system,
#containing excel files with the ticker data. The name of the file is the ticker symbol, and the file may contain the data for the date range provided.
	#check if the folder exists
	if not os.path.exists(local_repo_folder):
		print(f"Folder {local_repo_folder} does not exist.")
		return None
	#check if the folder is empty
	if not os.listdir(local_repo_folder):
		print(f"Folder {local_repo_folder} is empty.")
		return None
	#check if the folder contains any files
	if not glob.glob(os.path.join(local_repo_folder, '*.txt')):
		print(f"Folder {local_repo_folder} does not contain any txt files.")
		return None
	#check if the file exists
	if not os.path.exists(os.path.join(local_repo_folder, ticker + '.us.txt')):
		print(f"File {ticker}.us.txt does not exist in folder {local_repo_folder}.")
		return None
	#read the file into a dataframe
	file_path = os.path.join(local_repo_folder, ticker + '.us.txt')
	#read the file into a dataframe
	df = pd.read_csv(file_path, sep=',', parse_dates=['Date'], index_col='Date')
	#check if the dataframe is empty
	if df.empty:
		print(f"File {ticker}.txt is empty.")
		return None
	#check if the dataframe contains any NaN values
	if df.isnull().values.any():
		print(f"File {ticker}.txt contains NaN values.")
		return None
	#the file contains comma-separated values for Date,Open,High,Low,Close,Volume,OpenInt
	#check if the dataframe contains the required columns
	required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
	#drop the OpenInt column
	df.drop(columns=['OpenInt'], inplace=True)

	if 'Adj Close' not in df.columns:
		print(f"File {ticker}.txt does not contain Adj Close column.")
		#copy the Close column to Adj Close column
		df['Adj Close'] = df['Close']
		
	#filter the dataframe for the date range provided
	
	df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
	#check if the dataframe is empty after filtering
	if df.empty:
		print(f"File {ticker}.txt does not contain any data for the date range {start_date} to {end_date}.")
		return None
	
	ticker_filename = f"{source_files_folder}/portfolio_data_{ticker}_{start_date}_{end_date}.csv"
	#save the dataframe to a csv file
	df.to_csv(ticker_filename, index=True)
	print(f"File {ticker_filename} created.")
	return df

def check_local_store_for_ticker_data(ticker, start_date, end_date, local_store_folder):
	#check if the folder exists
	if not os.path.exists(local_store_folder):
		print(f"Folder {local_store_folder} does not exist.")
		return False
	#check if the folder is empty
	if not os.listdir(local_store_folder):
		print(f"Folder {local_store_folder} is empty.")
		return False
	#check if the folder contains any files
	if not glob.glob(os.path.join(local_store_folder, '*.txt')):
		print(f"Folder {local_store_folder} does not contain any txt files.")
		return False
	#check if the file exists
	if not os.path.exists(os.path.join(local_store_folder, ticker + '.us.txt')):
		print(f"File {ticker}.us.txt does not exist in folder {local_store_folder}.")
		return False

	#read the file into a dataframe and confirm that the data is available for the date range provided
	df = pd.read_csv(os.path.join(local_store_folder, ticker + '.us.txt'), sep=',', parse_dates=['Date'], index_col='Date')
	#check if the dataframe is empty
	if df.empty:
		print(f"File {ticker}.txt is empty.")
		return False
	#check if the dataframe contains any NaN values
	if df.isnull().values.any():
		print(f"File {ticker}.txt contains NaN values.")
		return False
	
	#filter the dataframe for the date range provided
	df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
	#check if the dataframe is empty after filtering
	if df.empty:
		print(f"File {ticker}.txt does not contain any data for the date range {start_date} to {end_date}.")
		return False
	
	return True

#function to compare the data in two files. The files are assumed to be in the same format and contain the same columns.
def compare_files(source_file_path_and_name, target_file_path_and_name):
	#columns to be compared:
	columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
	#read the files into dataframes
	source_data = pd.read_csv(source_file_path_and_name, sep=',', index_col='Date')
	#drop the columns that are not to be compared
	source_data.drop(columns=[col for col in source_data.columns if col not in columns], inplace=True)
	#if Adj Close column is not present in the target file, copy the Close column to Adj Close column
	if 'Adj Close' not in source_data.columns:
		source_data['Adj Close'] = source_data['Close']
	target_data = pd.read_csv(target_file_path_and_name, sep=',', index_col='Date')
	#drop the columns that are not to be compared
	target_data.drop(columns=[col for col in target_data.columns if col not in columns], inplace=True)
	#if Adj Close column is not present in the target file, copy the Close column to Adj Close column
	if 'Adj Close' not in target_data.columns:
		target_data['Adj Close'] = target_data['Close']

	#check if the dataframes are empty
	if source_data.empty or target_data.empty:
		print(f"One of the files is empty.")
		return False
	#check if the dataframes contain any NaN values
	if source_data.isnull().values.any():
		print(f"Source file {source_file_path_and_name} contains NaN values.")
		#show the rows with NaN values
		print(source_data[source_data.isnull().any(axis=1)])
		return False
	if target_data.isnull().values.any():
		print(f"Target file {target_file_path_and_name} contains NaN values.")
		return False
	
	# Check if the columns are the same
	source_columns = set(source_data.columns)
	target_columns = set(target_data.columns)

	if source_columns != target_columns:
		print("Columns are not the same between source and target:")
		print("Columns in source but not in target:", source_columns - target_columns)
		print("Columns in target but not in source:", target_columns - source_columns)
	else:
		print("Columns are the same between source and target.")

	#compare the data in the dataframes in specific date ranges. The filenames contain the date ranges.
	#Assume the left file is the source file and the right file is the target file.
	#the source filename is in the format portfolio_data_all_<ticker>_<start_date>_<end_date>.csv
	#the target filename is in the format portfolio_data_<ticker>_<start_date>_<end_date>.csv
	#extract the date ranges from the filenames
	start_date = target_file_path_and_name.split('_')[-2]
	end_date = target_file_path_and_name.split('_')[-1].split('.')[0]
	print(f"Start date: {start_date}, End date: {end_date}")
	#filter the dataframes for the date range provided
	source_data.index = pd.to_datetime(source_data.index)  # Ensure index is a DatetimeIndex
	source_data = source_data[(source_data.index >= pd.to_datetime(start_date)) & (source_data.index <= pd.to_datetime(end_date))]
	target_data.index = pd.to_datetime(target_data.index)  # Ensure index is a DatetimeIndex
	target_data = target_data[(target_data.index >= pd.to_datetime(start_date)) & (target_data.index <= pd.to_datetime(end_date))]
	#check if the dataframes are empty after filtering
	if source_data.empty or target_data.empty:
		print(f"One of the files does not contain any data for the date range {start_date} to {end_date}.")
		return False
	#compare the data in the dataframes
	if source_data.equals(target_data):
		print(f"The data in the files is the same.")
		return True
	else:
		print(f"The data in the files is different.")
		#compare the number of rows in the dataframes
		if source_data.shape[0] != target_data.shape[0]:
			print(f"The number of rows in the files is different.")
			print(f"Source file has {source_data.shape[0]} rows and target file has {target_data.shape[0]} rows.")
			#compare the two dataframes row by row and print the rows that are different
			#use the dataframe with the most rows as the base dataframe
			if source_data.shape[0] > target_data.shape[0]:
				print(f"Source file has more rows than target file.")
				different_rows = source_data[~source_data.index.isin(target_data.index)]
				print(f"Rows in source file that are not in target file:")
				print(different_rows)
			else:
				print(f"Target file has more rows than source file.")
				different_rows = target_data[~target_data.index.isin(source_data.index)]
				print(f"Rows in target file that are not in source file:")
				print(different_rows)
				return False
		else:
			print(f"The number of rows in the files is the same.")
			#show the rows that are different
			different_rows = source_data.compare(target_data)
			print(different_rows)
			#show the rows that are different in the source file
			different_rows_source = source_data[~source_data.index.isin(target_data.index)]
			print(f"Rows in source file that are not in target file:")
			print(different_rows_source)
			#show the rows that are different in the target file
			different_rows_target = target_data[~target_data.index.isin(source_data.index)]
			print(f"Rows in target file that are not in source file:")
			print(different_rows_target)
			#show the rows that are different in both files
			different_rows_both = source_data[~source_data.index.isin(target_data.index)].append(target_data[~target_data.index.isin(source_data.index)])
			print(f"Rows in both files that are different:")
			print(different_rows_both)
		return False
	
def test_combined_excel(start_date, end_date, combined_file_path_and_name):
	#read the combined file into a dataframe
	combined_data = pd.read_csv(combined_file_path_and_name, sep=',', index_col='Date')
	#check if the dataframe is empty
	if combined_data.empty:
		print(f"Combined file {combined_file_path_and_name} is empty.")
		return False
	#check if the dataframe contains any NaN values
	if combined_data.isnull().values.any():
		print(f"Combined file {combined_file_path_and_name} contains NaN values.")
		#show the rows with NaN values
		print(combined_data[combined_data.isnull().any(axis=1)])
		return False
	#filter the dataframe for the date range provided
	# Ensure the index is a DatetimeIndex
	combined_data.index = pd.to_datetime(combined_data.index)

	# Filter the data for the specified date range
	combined_data = combined_data[(combined_data.index >= pd.to_datetime(start_date)) & (combined_data.index <= pd.to_datetime(end_date))]
	#check if the dataframe is empty after filtering
	if combined_data.empty:
		print(f"Combined file {combined_file_path_and_name} does not contain any data for the date range {start_date} to {end_date}.")
		return False
	
	 # Detect steep drops in stock prices between 2018 and 2019
	steep_drop_threshold = -0.5  # Define a threshold for a steep drop (e.g., -50%)
	filtered_data = combined_data[(combined_data.index >= '2017-01-01') & (combined_data.index <= '2019-12-31')]
	filtered_data['Pct_Change'] = filtered_data['Adj Close'].pct_change()

	# Identify rows with steep drops
	steep_drops = filtered_data[filtered_data['Pct_Change'] <= steep_drop_threshold]
	if not steep_drops.empty:
		print(f"Steep drops detected between 2018 and 2019:")
		print(steep_drops[['Adj Close', 'Pct_Change']])
	else:
		print("No steep drops detected between 2018 and 2019.")
		
	#calculate the number of business days between the start and end date
	#use the US custom calendar to calculate the number of business days
	us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
	# Create a date range for the business days
	start_date = pd.to_datetime(start_date)
	end_date = pd.to_datetime(end_date)	
	business_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
	
	#Check if the number of business days in the combined file is equal to the number of business days calculated
	if len(combined_data) != len(business_days):
		print(f"The number of business days in the combined file is not equal to the number of business days calculated.")
		print(f"Combined file has {len(combined_data)} rows and calculated business days has {len(business_days)} rows.")
	else:
		print(f"The number of business days in the combined file is equal to the number of business days calculated.")

	status = True
	missing_dates = []
	#check each date in the list of business days against the dates in the combined file
	for date in business_days:
		if date not in combined_data.index:
			missing_dates.append(date)
			status = False
		# else:
		# 	print(f"Date {date.strftime('%Y-%m-%d')} is present in the combined file.")
	
	print(f"Number of missing dates: {len(missing_dates)}")
	if len(missing_dates) > 0:
		# Scan the target folder for all files and check if the missing dates are present in the files
		for file in os.listdir(config_params['source_files_folder']):
			if file.endswith('.csv') and file != os.path.basename(combined_file_path_and_name):
				print(f"Checking file {file} for missing dates.")
				# Read the file into a dataframe
				file_data = pd.read_csv(os.path.join(config_params['source_files_folder'], file), sep=',', index_col='Date')
				# Check if the dataframe is empty
				if file_data.empty:
					print(f"File {file} is empty.")
					continue
				# Check if the dataframe contains any NaN values
				if file_data.isnull().values.any():
					print(f"File {file} contains NaN values.")
					continue
				# Filter the dataframe for the date range provided
				file_data.index = pd.to_datetime(file_data.index)
				file_data = file_data[(file_data.index >= pd.to_datetime(start_date)) & (file_data.index <= pd.to_datetime(end_date))]
				# Check if the dataframe is empty after filtering
				if file_data.empty:
					print(f"File {file} does not contain any data for the date range {start_date} to {end_date}.")
					continue

				# Check for missing dates in the current file
				found_dates = set(file_data.index) & set(missing_dates)
				if found_dates:
					print(f"Found {len(found_dates)} missing dates in file {file}.")
					# Remove found dates from the missing_dates list
					missing_dates = [date for date in missing_dates if date not in found_dates]

		# After scanning all files, print any remaining missing dates
		if missing_dates:
			print(f"{len(missing_dates)} dates are still missing from the combined file:")
			for date in missing_dates:
				print(f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
		else:
			print("All missing dates have been found.")


	
	return status

def check_scaler_files_exist(scaler_files_folder, file_prefix, date, version):
	#check if the scaler files exist in the local store
	if not os.path.exists(scaler_files_folder):
		print(f"Folder {scaler_files_folder} does not exist.")
		return False
	#check if the folder is empty
	if not os.listdir(scaler_files_folder):
		print(f"Folder {scaler_files_folder} is empty.")
		return False
	#check if the folder contains any files
	if not glob.glob(os.path.join(scaler_files_folder, '*.save')):
		print(f"Folder {scaler_files_folder} does not contain any joblib files.")
		return False
	
	feature_scaler = os.path.join(scaler_files_folder, f"{file_prefix}_feature_scaler_{date}_v{version}.save")
	target_scaler = os.path.join(scaler_files_folder, f"{file_prefix}_target_scaler_{date}_v{version}.save")
	#check if the scaler files exist
	if not os.path.exists(feature_scaler):
		print(f"File {feature_scaler} does not exist.")
		return False
	if not os.path.exists(target_scaler):
		print(f"File {target_scaler} does not exist.")
		return False
	
	return True

#main function
if __name__ == '__main__':

	# start_date = date.fromisoformat('2006-01-01')
	# end_date = date.fromisoformat('2018-01-01')

	# days = (end_date - start_date).days
	# print(days)
	
	#get all configuration parameters from the config.properties file as strings as they will be set as environment variables
	config_params = {}

	config = configparser.ConfigParser()
	config.read('/Users/shriniwasiyengar/git/python_ML/LLM/EquityResearchReports/backend/ai-service/config.properties')
	# define the time period for the data
	start_date = config.get('Portfolio', 'start_date') #'2020-02-16'
	end_date = config.get('Portfolio', 'end_date') #'2020-02-16'

	config_params = load_configuration_params()
	# tickers = ['BAC','LIT','VTSAX','CSCO','GIS','SONY','INTC']
	start_date = config_params['start_date'] #date.fromisoformat(os.environ.get('start_date')) #'2006-01-01'
	# increment = timedelta(days=int(os.environ.get('increment'))) #timedelta(days=30)
	# end_date = date.fromisoformat(os.environ.get('end_date'))
	end_date = config_params['end_date'] #date.fromisoformat(os.environ.get('end_date')) #'2018-01-01'
	increment = config_params['increment']

	# timeframes_df = get_timeframes_df(start_date, end_date, increment)
	# print(timeframes_df)
	# fetch_data_into_files(config_params, timeframes_df)
	# for index, row in timeframes_df.iterrows():
	# 	start_date = row['From Date']
	# 	end_date = row['To Date']
	# 	print(f"Start date: {start_date}, End date: {end_date}")
	# 	#check if the ticker data is available in the local store
	# 	tickers = config_params['tickers']
	# 	if isinstance(tickers, str):
	# 		tickers = [ticker.strip() for ticker in tickers.split(',')]

	# 	for ticker in tickers:
	# 		print(f"Ticker: {ticker}")
	# 		#check if the ticker data is available in the local store
	# 		if not check_local_store_for_ticker_data(ticker, start_date, end_date, config_params['source_files_folder']):
	# 			print(f"Ticker data for {ticker} is not available in the local store.")
	# 			#fetch the data from the source and save it to the local store
	# 			# get_ticker_datafile_from_local_datastore(ticker, start_date, end_date, config_params['source_files_folder'])
	# 		else:
	# 			print(f"Ticker data for {ticker} is available in the local store.")

	# source_file = config_params['source_files_folder']+'-old' + '/portfolio_data_all_1984-09-11_2025-03-27.csv'
	# target_file = config_params['source_files_folder'] + '/portfolio_data_AAPL_2005-01-01_2006-01-01.csv'
	# compare_files(source_file,target_file)
	#scan the target folder for all files and compare them with the source file
	# for file in os.listdir(config_params['source_files_folder']):
	# 	if file.endswith('.csv'):
	# 		print(f"Comparing file {file} with source file {source_file}")
	# 		compare_files(source_file, os.path.join(config_params['source_files_folder'], file))
	# 	else:
	# 		print(f"File {file} is not a csv file.")

	# target_combined_file = config_params['source_files_folder'] + '/portfolio_data_all_2005-01-03_2025-05-02.csv'		
	# start_date = '2005-01-03'
	# end_date = '2025-05-02'
	# test_combined_excel(start_date, end_date, target_combined_file)

	#download data from yahoo finance for a specific date range and save it to the local store
	tickers= config_params['tickers']
	# start_date = '2019-01-01'
	# end_date = '2020-01-01'
	# new_source_file = config_params['source_files_folder'] + '/portfolio_data_AAPL_2019-01-01_2020-01-01.csv'	
	# start_date = '2020-01-02'
	# end_date = '2021-01-01'
	# new_source_file = config_params['source_files_folder'] + '/portfolio_data_AAPL_2020-01-02_2021-01-01.csv'
	# start_date = '2021-01-02'
	# end_date = '2022-01-02'
	# new_source_file = config_params['source_files_folder'] + '/portfolio_data_AAPL_2021-01-02_2022-01-02.csv'
	# start_date = '2022-01-03'
	# end_date = '2023-01-03'
	# new_source_file = config_params['source_files_folder'] + '/portfolio_data_AAPL_2022-01-03_2023-01-03.csv'
	# start_date = '2025-01-06'
	# end_date = '2025-05-20'
	# # new_source_file = config_params['source_files_folder'] + '/portfolio_data_AAPL_2025-01-06_2025-05-19.csv'
		
	# if isinstance(tickers, str):
	# 	tickers = [ticker.strip() for ticker in tickers.split(',')]
	# for ticker in tickers:
	# 	fetch_data_from_yahoo(tickers, start_date, end_date, new_source_file)


	split_date = get_split_date(start_date, end_date, percentage=0.8)
	print(f"Split date: {split_date.strftime('%Y-%m-%d')}")