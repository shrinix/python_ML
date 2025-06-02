import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import yfinance as yf
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Attention, Add, LayerNormalization, Layer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient
from pandas.tseries.offsets import CustomBusinessDay
import pandas_market_calendars as mcal


# Configuration for InfluxDB
bucket = "stock_data"
org = "Telluride Solutions"
token = "T7B_OVPPwqwfASNP_SzR9jbnRbdTI381GsjCn14p1r2geDrPk8UiB7LJ9SEA5eGBbuvZUMOP0hU8lBUJtwk3tA=="
url = "http://localhost:8086"

def query_stock_data(ticker, start_date, end_date):
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	# Ensure start_date and end_date are RFC3339 timestamps (no extra quotes in Flux)
	if hasattr(start_date, 'strftime'):
		start_rfc3339 = start_date.strftime('%Y-%m-%dT00:00:00Z')
	elif "T" in str(start_date):
		start_rfc3339 = str(start_date)
	else:
		start_rfc3339 = f"{start_date}T00:00:00Z"

	if hasattr(end_date, 'strftime'):
		end_rfc3339 = end_date.strftime('%Y-%m-%dT23:59:59Z')
	elif "T" in str(end_date):
		end_rfc3339 = str(end_date)
	else:
		end_rfc3339 = f"{end_date}T23:59:59Z"

	print(f"Querying data for ticker: {ticker}, from {start_rfc3339} to {end_rfc3339}")  # Debug

	flux = f'''
	from(bucket: "{bucket}")
	|> range(start: {start_rfc3339}, stop: {end_rfc3339})
	|> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
	|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
	|> sort(columns: ["_time"])
	'''
	print("Flux query:\n", flux)  # Debug: See the actual query

	records = []
	tables = query_api.query(flux)
	for table in tables:
		for rec in table.records:
			records.append({
				"Date": pd.to_datetime(rec["_time"]),
				"Open": rec["Open"],
				"High": rec["High"],
				"Low": rec["Low"],
				"Close": rec["Close"],
				"Adj Close": rec["Adj Close"],
				"Volume": int(rec["Volume"])
			})

	client.close()
	df = pd.DataFrame(records)
	numeric_cols = df.select_dtypes(include='number').columns
	df[numeric_cols] = df[numeric_cols].round(2)
	return df

def fetch_stock_data_from_yahoo(ticker, start_date, end_date):
	"""
	Fetch Tesla's historical stock data from Yahoo Finance.

	Returns:
		pd.DataFrame: DataFrame containing adjusted close prices indexed by date.
	"""
	# # Fetch data for Tesla (TSLA) from Yahoo Finance
	# start_date = "2010-01-01"
	# end_date = "2024-11-17"
	stock_data = yf.download(ticker, start=start_date, end=end_date)

	# Return a DataFrame with the adjusted close prices
	# stock_data = ticker_data[['Adj Close']].rename(columns={"Adj Close": "adjClose"})
	stock_data.index.name = "date"
	return stock_data

# Fetch stock data
start_date = "2010-01-01"
end_date = "2024-11-17"
ticker = "AAPL" 
#stock_data = fetch_stock_data_from_yahoo('AAPL', start_date, end_date)
stock_data = query_stock_data(ticker, start_date, end_date)
print("Number of rows in stock data:", len(stock_data)) 
# Ensure the index is a DatetimeIndex before formatting
if not isinstance(stock_data.index, pd.DatetimeIndex):
	stock_data.index = pd.to_datetime(stock_data.index)
# Set index to "Date" column if it exists and not already set
if "Date" in stock_data.columns and stock_data.index.name != "Date":
	stock_data.set_index("Date", inplace=True)
	stock_data.index = pd.to_datetime(stock_data.index)

print("Min date in stock data:", stock_data.index.min().strftime('%Y-%m-%d'))
print("Max date in stock data:", stock_data.index.max().strftime('%Y-%m-%d'))
print("Number of business days between start and end date:", len(pd.date_range(start=start_date, end=end_date, freq='B')))

#if number of rows in stock data is less than number of business days, use yahoo finance to fetch data
if len(stock_data) < len(pd.date_range(start=start_date, end=end_date, freq='B')):
	print("Stock data from InfluxDB is incomplete, fetching from Yahoo Finance...")
	# Ensure the DataFrame index is a DatetimeIndex for correct comparison
	if not isinstance(stock_data.index, pd.DatetimeIndex):
		if "Date" in stock_data.columns:
			stock_data.set_index("Date", inplace=True)
		stock_data.index = pd.to_datetime(stock_data.index)

	# Now calculate missing dates correctly
	#get US holidays calendar
	#use the US holiday calendar to determine business days
	nyse = mcal.get_calendar('NYSE')
	all_business_days = nyse.schedule(start_date=start_date, end_date=end_date).index
	# all_business_days is now set above using NYSE calendar
	present_days = pd.to_datetime(stock_data.index.date).unique()
	missing_dates = pd.DatetimeIndex(all_business_days).difference(present_days)
	missing_dates = all_business_days.difference(present_days)
	#sort the missing dates from oldest to newest
	missing_dates = missing_dates.sort_values()
	print("Number of missing dates:", len(missing_dates))
	#capture the missing dates into yearly buckets using original start and end dates
	if len(missing_dates) == 0:
		print("No missing dates found, using existing stock data.")
		# Use the existing stock_data for analysis
		stock_data = stock_data.sort_index()
		print("Stock data is complete, using it for analysis.")
	else:
		print("Missing dates found, fetching data for these dates from Yahoo Finance.")
		yearly_buckets = {}
		#using start_date and end_date to create keys for yearly buckets as <first_start_date>_<first_end_date>,
		#  <second_start_date>_<second_end_date>, etc. where first_start_date = start_date of original range, and last_end_date = end_date of original range
		#each subsequent start_date should be previous start_date + 1 day, and each subsequent end_date should be previous end_date + 365
		#compute number of years in the range
		# Convert start_date and end_date to pd.Timestamp for date arithmetic
		start_dt = pd.to_datetime(start_date)
		end_dt = pd.to_datetime(end_date)
		num_years = (end_dt - start_dt).days // 365 + 1
		# Ensure start_date and end_date are pd.Timestamp for date arithmetic
		start_date_ts = pd.to_datetime(start_date)
		end_date_ts = pd.to_datetime(end_date)
		for i in range(num_years):
			# Calculate the start and end dates for each year
			year_start = start_date_ts + pd.Timedelta(days=i * 365)
			year_end = year_start + pd.Timedelta(days=364)
			# Ensure the end date does not exceed the original end_date
			if year_end > end_date_ts:
				year_end = end_date_ts
			# Create a key for the yearly bucket
			bucket_key = f"{year_start.strftime('%Y-%m-%d')}_{year_end.strftime('%Y-%m-%d')}"
			# Store the start and end dates in the dictionary
			yearly_buckets[bucket_key] = (year_start, year_end)
		# Print the yearly buckets
		print("Yearly buckets for missing dates:")
		for key, (start, end) in yearly_buckets.items():
			print(f"{key}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
		#scan the missing dates and create a list of missing dates for each year_range determined by the key in yearly_buckets
		#also store this list as a value in the yearly_buckets dictionary
		for key, (start, end) in yearly_buckets.items():
			# Create a mask for missing dates within the current year range
			mask = (missing_dates >= start) & (missing_dates <= end)
			# Get the missing dates for this year range
			missing_dates_for_year = missing_dates[mask]
			# Store the missing dates in the dictionary
			yearly_buckets[key] = missing_dates_for_year.tolist()
		#Print a count of missing dates by year
		print("Count of missing dates by year:")
		for key, missing_dates_list in yearly_buckets.items():
			print(f"{key}: {len(missing_dates_list)} missing dates")
	
	input("Press Enter to continue fetching data for missing dates from Yahoo Finance...")
	# If there are missing dates, fetch data for those dates from Yahoo Finance
	#Get data for the entire yearly range if the number of missing dates is greater than 10
	for key, missing_dates in yearly_buckets.items():
		if len(missing_dates) > 10:
			# Get the start and end dates for the current year range by splitting the key
			start_date, end_date = key.split('_')
			start_date = pd.to_datetime(start_date)
			end_date = pd.to_datetime(end_date)
			print(f"Fetching data for {key} from Yahoo Finance: {start_date} to {end_date}")
			# Fetch data for the missing dates from Yahoo Finance
			stock_data_yf = fetch_stock_data_from_yahoo(ticker, start_date, end_date)
			# Save the fetched data to a CSV file for future use
			folder_path = "/Users/shriniwasiyengar/git/python_ML/LLM/EquityResearchReports/backend/ai-service/source-files"
			file_path = f"{folder_path}/portfolio_data_{ticker}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
			stock_data_yf.to_csv(file_path)
			print(f"Data saved to {file_path}")
			print(f"Fetched {len(stock_data_yf)} items from Yahoo Finance.")
			# Append the fetched data to the existing stock_data DataFrame
			stock_data = pd.concat([stock_data, stock_data_yf])
			# Remove duplicates based on the index (date)
			stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
			print("Stock data after merging with Yahoo Finance:", len(stock_data), "rows.")
else:
	print("Stock data from InfluxDB is complete, using it for analysis.")
# Display the first few rows of data
print(stock_data.head(10))

# Define the window size and prediction time
window_size = 20
prediction_steps = 10
# Function to create sequences
def create_sequences(data, window_size, prediction_steps):
	X = []
	y = []
	for i in range(window_size, len(data) - prediction_steps):
		X.append(data[i-window_size:i, 0]) # input sequence
		y.append(data[i+prediction_steps-1, 0]) # target value (price at the next timestep)
	return np.array(X), np.array(y)
# Fetch Tesla stock data
data = stock_data[['Adj Close']].values
# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
# Create sequences for the model
X, y = create_sequences(scaled_data, window_size, prediction_steps)
# Reshape input data to be in the shape [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Define a custom attention layer
class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		super(AttentionLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(shape=(input_shape[2], input_shape[2]), initializer='random_normal', trainable=True)
		self.b = self.add_weight(shape=(input_shape[1],), initializer='zeros', trainable=True)
		super(AttentionLayer, self).build(input_shape)

	def call(self, inputs):
		q = tf.matmul(inputs, self.W)
		a = tf.matmul(q, inputs, transpose_b=True)
		attention_weights = tf.nn.softmax(a, axis=-1)
		return tf.matmul(attention_weights, inputs)

# LSTM model with attention and early stopping
def build_lstm_model_with_attention(input_shape):
	model = Sequential()
	model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
	model.add(Dropout(0.2))
	
	# Attention layer
	model.add(AttentionLayer())
	model.add(LayerNormalization())
	
	model.add(LSTM(units=50, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(units=1))  # Output layer for prediction
	
	model.compile(optimizer='adam', loss='mean_squared_error')
	return model

# Build the LSTM model with attention
model = build_lstm_model_with_attention(X_train.shape[1:])

# Implement EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with EarlyStopping and 50 epochs
# Set a custom learning rate for the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=200, batch_size=4, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Inverse scale the actual stock prices
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test_scaled, predicted_stock_price)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# Calculate RMSE
rmse = np.sqrt(np.mean((y_test_scaled - predicted_stock_price) ** 2))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} USD")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled, label=f"Actual {ticker} Stock Price", color='blue')
plt.plot(predicted_stock_price, label=f"Predicted {ticker} Stock Price", color='red')
plt.title(f'{ticker} Stock Price Prediction with LSTM', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Scaled Stock Price (USD)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()