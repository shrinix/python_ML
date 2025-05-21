# Import necessary libraries for LSTM model
import builtins
import tensorflow as tf
globals()['tf'] = tf # <-- This ensures tf is available in the Lambda's global scope
builtins.tf = tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Define the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, AdamW, RMSprop
import random
import os
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from utils import (
	get_max_version, get_timeframes_df, load_configuration_params, prepare_data_for_training,
	fetch_data_into_files, load_and_transform_data_by_ticker, add_engineered_features,
	check_data, update_features_after_prediction_lstm, check_scaler_files_exist
)
import keras
import joblib

def forecast_n_steps_ahead_using_lstm(X_test_scaled, y_test_scaled, lstm_model, feature_scaler, target_scaler, n_days, column_names):
	# Initialize a DataFrame to store predictions
	future_predictions_unscaled = pd.DataFrame(columns=column_names+['Trend_Predicted']) #[
	# 	'7_day_avg', '30_day_avg', 'Daily Ret', 'Volatility_7', 'Volatility_30',
	#    'delta', 'avg_gain', 'avg_loss', 'rs', 'RSI', 'ema_12', 'ema_26', 'MACD',
	#    'Signal_Line', 'Bollinger_Upper', 'Bollinger_Lower', 'rolling_mean',
	#    'rolling_std','Trend_Predicted']

	print("X_test_scaled shape: ", X_test_scaled.shape)
	print ("Feature columns: ", column_names)

	# Start with the last row of the test data
	last_row_scaled = X_test_scaled[-1, :]  # Use all columns. last column has already been removed
	last_row_unscaled = feature_scaler.inverse_transform(last_row_scaled.reshape(1, -1)).flatten()	
	#get last value from y_test_scaled
	previous_adj_close_scaled = y_test_scaled[-1]  # The last known adjusted close price
	
	# Generate a range of business days for the next n_days
	last_date = pd.Timestamp.now()  # Replace with the last date in your dataset if available
	business_days = pd.bdate_range(start=last_date, periods=n_days)
	#enumerate all daates including weekends and holidays
	# days = pd.date_range(start=last_date, periods=n_days, freq='D')
	
	n_features = len(column_names)  # Number of features in the last row
	predictions = []
	current_input_df = pd.DataFrame()
	step=0
	for day in business_days:
	# for step in range(n_steps):
		previous_adj_close_unscaled = target_scaler.inverse_transform(previous_adj_close_scaled.reshape(1, -1)).item()
		print("Unscaled previous adjusted close: ", previous_adj_close_unscaled)

		current_input_df_inc = pd.DataFrame([last_row_unscaled], columns=column_names)
		current_input_df_inc['Step'] = step
		current_input_df_inc['Date'] = day
		current_input_df_inc['Previous Adj Close'] = previous_adj_close_unscaled
		current_input_df_inc = np.round(current_input_df_inc, 2)

		# Reshape the input to match LSTM input shape: [samples, timesteps, features]
		input_data = last_row_scaled.reshape(1, 1, -1)
		print(f"Input data shape: {input_data.shape} at step {step}")

		# Predict the next step
		prediction = lstm_model.predict(input_data)
		print(f"Prediction shape: {prediction.shape} at step {step}")	
		if prediction.ndim == 2 and prediction.shape[1] == 1:
			predicted_value = prediction[0, 0]  # Extract the predicted value
		else:
			raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
		print(f"Step {step + 1}: Predicted value = {predicted_value:.4f}")
		
		predicted_value_actual = target_scaler.inverse_transform(np.array([[predicted_value]])).reshape(1, -1)[0, 0]
		# Append the first prediction (next day's value) to the results
		predictions.append(predicted_value_actual)
		#add predicted value to the current_input_df for the next iteration
		current_input_df_inc['Predicted'] = predicted_value_actual
		current_input_df = pd.concat([current_input_df, current_input_df_inc], ignore_index=True)
		print(f"Step {step + 1}: Predicted value (actual) = {predicted_value_actual:.4f}")
		X_test_unscaled = feature_scaler.inverse_transform(X_test_scaled.reshape(-1, n_features)).reshape(X_test_scaled.shape)
		y_test_unscaled = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)
		last_row_unscaled = feature_scaler.inverse_transform(last_row_scaled.reshape(1, -1)).flatten()
		# predicted_value, last_row_unscaled, previous_adj_close, step, X_test_unscaled, y_test_unscaled, predicted_unscaled,  column_names
		last_row_updated = update_features_after_prediction_lstm(predicted_value_actual, last_row_unscaled, previous_adj_close_unscaled, step, X_test_unscaled,  y_test_unscaled, predictions,  column_names)
		print("Updated last row: ", last_row_updated)
		# Append the prediction and updated features to the DataFrame
		future_predictions_step_unscaled = pd.DataFrame(last_row_updated.reshape(1, -1), columns=column_names)
		future_predictions_step_unscaled['Trend_Predicted'] = predicted_value_actual
		future_predictions_step_unscaled['Date'] = day
		future_predictions_unscaled = pd.concat([future_predictions_unscaled, future_predictions_step_unscaled], ignore_index=True)
		print(f"Future Predictions Unscaled (Step {step + 1}):")
		previous_adj_close_scaled = predicted_value
		last_row_scaled = feature_scaler.transform(last_row_updated.reshape(1, -1)).flatten()
		step += 1

	#set Date as index for current_input_df
	current_input_df['Date'] = pd.to_datetime(current_input_df['Date'])
	#Sort the columns to have the order Step, Date, Predicted, Previous Adj Close, followed by the other columns
	current_input_df = current_input_df[['Step', 'Date', 'Previous Adj Close','Predicted'] + [col for col in current_input_df.columns if col not in ['Step', 'Date', 'Predicted', 'Previous Adj Close']]]
	# current_input_df.set_index('Date', inplace=True)
	#save the current_input_df to a csv file
	current_input_df.to_csv(f'{output_files_folder}/lstm_current_input_df.csv', index=False)

	#save x_encoder_test_3d_unscaled to a csv file
	X_encoder_test_unscaled_df = pd.DataFrame(X_test_unscaled.reshape(-1, n_features), columns=column_names)
	X_encoder_test_unscaled_df.to_csv(f'{output_files_folder}/lstm_X_encoder_test_3d_unscaled_df.csv', index=False)

	#save y_decoder_test_3d_unscaled to a csv file
	y_decoder_test_unscaled_df = pd.DataFrame(y_test_unscaled.reshape(-1, 1), columns=['Tgt'])
	y_decoder_test_unscaled_df.to_csv(f'{output_files_folder}/lstm_y_decoder_test_3d_unscaled_df.csv', index=False)

	#set date as index for future_predictions_unscaled
	future_predictions_unscaled['Date'] = pd.to_datetime(future_predictions_unscaled['Date'])
	future_predictions_unscaled.set_index('Date', inplace=True)
	# Print the future predictions
	print("Future Predictions:")
	print(future_predictions_unscaled)
	
	return future_predictions_unscaled

def backtest_lstm(stock_data, lstm_model, scaler, n_steps, window_size):
	"""
	Back-test the LSTM model using a rolling window approach.

	Parameters:
		stock_data (pd.DataFrame): The historical stock data with features and target.
		lstm_model (tf.keras.Model): The trained LSTM model.
		scaler (MinMaxScaler): The scaler used for feature scaling.
		n_steps (int): Number of steps to forecast.
		window_size (int): Size of the rolling window for training.
		ax (matplotlib.axes.Axes): The axis object for plotting.

	Returns:
		pd.DataFrame: A DataFrame containing actual and predicted values for back-testing.
	"""
	# Ensure the dataset has enough rows for back-testing
	if len(stock_data) < window_size + n_steps:
		raise ValueError(
			f"Dataset is too small for back-testing. "
			f"Required: {window_size + n_steps} rows, but got: {len(stock_data)} rows."
		)

	backtest_results = pd.DataFrame(columns=['Actual', 'Predicted'])

	# Iterate over the data with a rolling window
	for start_idx in range(0, len(stock_data) - window_size - n_steps + 1):
		# Define the training and test sets for the current window
		train_window = stock_data.iloc[start_idx:start_idx + window_size]
		test_window = stock_data.iloc[start_idx + window_size:start_idx + window_size + n_steps]

		# Skip if the training or test window is empty
		if train_window.empty or test_window.empty:
			print(f"Skipping window starting at index {start_idx} due to insufficient data.")
			continue

		# Prepare the data for the LSTM model
		try:
			X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, train, test, _ = prepare_data_for_training(train_window)
		except ValueError as e:
			print(f"Skipping window starting at index {start_idx} due to data preparation error: {e}")
			continue

		# Forecast n steps ahead
		predictions = forecast_n_steps_ahead_using_lstm(train, test, X_test_scaled, lstm_model, target_scaler, n_steps)

		# Store the actual and predicted values
		actual_values = test_window['Tgt'].values[:n_steps]
		predicted_values = predictions['Trend_Predicted'].values

		# Append the results to the backtest DataFrame
		for actual, predicted in zip(actual_values, predicted_values):
			backtest_results = pd.concat(
				[backtest_results, pd.DataFrame([[actual, predicted]], columns=['Actual', 'Predicted'])],
				ignore_index=True
			)

	return backtest_results

def analyze_backtest_results(backtest_results, ax):

	# Calculate performance metrics
	mae = mean_absolute_error(backtest_results['Actual'], backtest_results['Predicted'])
	mse = mean_squared_error(backtest_results['Actual'], backtest_results['Predicted'])
	rmse = np.sqrt(mse)

	# Print metrics
	print(f"Mean Absolute Error (MAE): {mae:.2f}")
	print(f"Mean Squared Error (MSE): {mse:.2f}")
	print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

	# Plot actual vs. predicted values
	ax[0,0].plot(backtest_results['Actual'], label='Actual', marker='o')
	ax[0,0].plot(backtest_results['Predicted'], label='Predicted', linestyle='--', marker='o')
	ax[0,0].set_title('Back-Test Results: Actual vs. Predicted')
	ax[0,0].set_xlabel('Time')
	ax[0,0].set_ylabel('Adj Close Price')
	ax[0,0].legend()

	# Calculate residuals
	backtest_results['Residual'] = backtest_results['Actual'] - backtest_results['Predicted']

	# Plot residuals
	ax[0,1].plot(backtest_results['Residual'], label='Residuals', marker='o')
	ax[0,1].axhline(0, color='red', linestyle='--', linewidth=1)
	ax[0,1].set_title('Residuals Over Time')
	ax[0,1].set_xlabel('Time')
	ax[0,1].set_ylabel('Residual')
	ax[0,1].legend()

	# Plot histogram of residuals
	ax[1,0].hist(backtest_results['Residual'], bins=30, edgecolor='black')
	ax[1,0].set_title('Histogram of Residuals')
	ax[1,0].set_xlabel('Residual')
	ax[1,0].set_ylabel('Frequency')
	ax[1,0].legend()

	# Plot the back-test results
	ax[1,1].plot(backtest_results['Actual'], label='Actual', marker='o')
	ax[1,1].plot(backtest_results['Predicted'], label='Predicted', linestyle='--', marker='o')
	ax[1,1].set_title('Back-Test Results')
	ax[1,1].set_xlabel('Time')
	ax[1,1].set_ylabel('Adj Close Price')
	ax[1,1].legend()

	# Identify periods with large residuals
	threshold = 2 * rmse  # Define a threshold for poor performance
	poor_performance = backtest_results[np.abs(backtest_results['Residual']) > threshold]
	print("Periods of Poor Performance:")
	print(poor_performance)

#Adds metrics data to the plot
def plot_metrics(future_predictions, train, test, n_steps, title, ax):
	
	ax.plot(train.index, train['Tgt'], label='Training Data', marker='o')
	ax.plot(test.index, test['Tgt'], label='Test Data', marker='o')
	# ax.plot(future_predictions.index, future_predictions['Trend_Predicted'], label='Predicted Trend', linestyle='--', marker='o')
	ax.plot(
		pd.date_range(start=test.index[-1], periods=n_steps + 1, freq='D')[1:],
		future_predictions['Trend_Predicted'],
		label='Predicted Trend (n steps)',
		linestyle='--',
		marker='o'
	)
	ax.set_title(title)
	ax.set_xlabel('Date')
	ax.set_ylabel('Adj Close Price')
	ax.legend()

	#cannot calculate performance metrics like lstm since we are predicting multiple steps ahead and there is
	#no reference data to compare with

def plot_future_trend(future_predictions, test_prepared_scaled_df,  n_steps, title, ax):
	print("Shape of future predictions: ", future_predictions.shape)
	print("Shape of test prepared scaled data: ", test_prepared_scaled_df.shape)
	ax.plot(
		# pd.date_range(start=test_prepared_scaled_df.index[-1], periods=n_steps + 1, freq='D')[1:],
		future_predictions.index,
		future_predictions['Trend_Predicted'],
		label='Predicted Trend (n steps)',
		linestyle='--',
		marker='o'
	)
	#add annotations
	for i in range(n_steps):
		ax.annotate(
			f"Step {i + 1}: {future_predictions['Trend_Predicted'].iloc[i]:.2f}",
			(future_predictions.index[i], future_predictions['Trend_Predicted'].iloc[i]),
			textcoords="offset points",
			xytext=(0, 10),
			ha='center'
		)
	ax.set_title(title)
	ax.set_xlabel('Date')
	ax.set_ylabel('Adj Close Price')
	ax.legend()

if __name__ == '__main__':

	#create a plot with multiple sections
	fig, ax = plt.subplots(1,1, figsize=(14,12)) #width, height

	config_params = load_configuration_params()
	start_date = config_params['start_date']
	end_date = config_params['end_date']
	increment = config_params['increment']
	output_files_folder = config_params['output_files_folder']
	split_date = config_params['split_date']

	timeframes_df = get_timeframes_df(start_date, end_date, increment)
	print(timeframes_df)

	#Look for most recent models and scalers.
	look_for_older_files=True
	current_date = datetime.now().strftime('%Y-%m-%d')
	file_prefix='lstm_model'
	version,date = get_max_version(output_files_folder, file_prefix, look_for_older_files=True)

	load_from_files=True
	# stock_data = yf.download('AAPL', start='2022-01-01', end='2024-01-01')
	# stock_data = pd.read_csv('../data/stock_data.csv', parse_dates=['Date'], index_col='Date')
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
	stock_data['Tgt'] = stock_data['Adj Close'].shift(-1) # next day's price
	print("Before adding engineered features: ", stock_data.shape)
	# Add engineered features
	stock_data,feature_columns = add_engineered_features(stock_data)
	stock_data.dropna(inplace=True)
	print("Feature columns after adding engineered features: ", feature_columns)
	print("Stock data columns after adding engineered features: ", stock_data.columns)
	print("Shape of stock data after adding engineered features: ", stock_data.shape)

	# Calculate rolling averages
	stock_data['Tgt'] = stock_data['Adj Close'].shift(-1) # next day's price
	stock_data.dropna(inplace=True)

	#check if scaler files exist
	scaler_files_exist = check_scaler_files_exist(output_files_folder, file_prefix, date, version)
	X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = None, None, None, None
	train_prepared_df, test_prepared_df = None, None
	if scaler_files_exist:
		print("Scaler files exist. Loading scalers.")
		feature_scaler = os.path.join(output_files_folder, f"{file_prefix}_feature_scaler_{date}_v{version}.save")
		target_scaler = os.path.join(output_files_folder, f"{file_prefix}_target_scaler_{date}_v{version}.save")
		feature_scaler = joblib.load(feature_scaler)
		target_scaler = joblib.load(target_scaler)
	
		# disable_scaling = True
		# train_prepared_df, test_prepared_df, _, _ = prepare_data_for_training(stock_data, feature_columns, split_date, 'Tgt', disable_scaling)
		#create X_test_scaled, y_test_scaled from stock_data
		test_prepared_df = stock_data[stock_data.index > split_date].tail(30)
		X_test = test_prepared_df[feature_columns].values
		y_test = test_prepared_df['Tgt'].values
		X_test_scaled = feature_scaler.transform(X_test)
		y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
		print("X_test_scaled shape: ", X_test_scaled.shape)
		print("y_test_scaled shape: ", y_test_scaled.shape)
		#check if X_test_scaled and y_test_scaled are scaled
	
	else: #run data pipeline to create scalers		
		disable_scaling = False #When using k-fold, scaling is done after KFold splitting.
		print("Preparing data for training...")
		feature_scaler, target_scaler = None, None

		if disable_scaling:
			print("Scaling is disabled. Data will not be scaled during preparation.")
			#discard scaler outputs
			#These checks will work only if data is not scaled
			train_prepared_df, test_prepared_df, _, _ = prepare_data_for_training(stock_data, feature_columns, split_date, 'Tgt', disable_scaling)
			check_data(train_prepared_df, train_prepared_df.columns)
			check_data(test_prepared_df, test_prepared_df.columns)
		else:
			print("Scaling is enabled. Data will be scaled during preparation.")        
			train_prepared_df, test_prepared_df, feature_scaler, target_scaler = prepare_data_for_training(stock_data, feature_columns, split_date, 'Tgt', disable_scaling)

		print("train_prepared_df shape: ", train_prepared_df.shape)
		print("test_prepared_df shape: ", test_prepared_df.shape)

		#create X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled from train_prepared_df and test_prepared_df
		X_train_scaled = train_prepared_df.drop(columns=['Tgt']).values
		y_train_scaled = train_prepared_df['Tgt'].values
		X_test_scaled = test_prepared_df.drop(columns=['Tgt']).values
		y_test_scaled = test_prepared_df['Tgt'].values
		
		print("X_train_scaled shape: ", X_train_scaled.shape)
		print("y_train_scaled shape: ", y_train_scaled.shape)
		print("X_test_scaled shape: ", X_test_scaled.shape)
		print("y_test_scaled shape: ", y_test_scaled.shape)

	# load model
	model_path = f'{output_files_folder}/{file_prefix}_{date}_v{version}.keras'
	print(f"Model path: {model_path}")
	lstm_model = None
	if os.path.exists(model_path):            
		keras.config.enable_unsafe_deserialization()
		# lstm_model = tf.keras.Sequential([
		# 	tf.keras.models.load_model(model_path)
		# ])
		lstm_model = tf.keras.models.load_model(model_path)
		print(f"Loaded model: {lstm_model.name} from {model_path}")
		lstm_model.summary()
		print(f"Model input shape: {lstm_model.input_shape}")
		print(f"Model output shape: {lstm_model.output_shape}")

	else:
		print(f"Model not found at {model_path}. Creating new model.")
	
	# # forecast_data_using_lstm(X_test_scaled, y_test_scaled, lstm_model, ax)
	n_steps = 10  # Number of steps to forecast

	future_predictions_unscaled = forecast_n_steps_ahead_using_lstm(X_test_scaled, y_test_scaled, lstm_model, feature_scaler, target_scaler, n_steps, feature_columns)
	print("Future Predictions:")
	print(future_predictions_unscaled.head())
	# #save the future predictions to a file
	# future_predictions.to_csv(f'{output_files_folder}/lstm_future_predictions.csv', index=False)

	# plot_metrics(future_predictions, train, test, n_steps, f'Financial Time Series Trend Forecasting ({n_steps} ahead)', ax[0])
	plot_future_trend(future_predictions_unscaled, test_prepared_df, n_steps, f'FutureTrend Forecasting ({n_steps} ahead)', ax)
	plt.tight_layout()
	#save the plot to a pdf
	version=version+1 #increment version
	plot_path = f'{output_files_folder}/lstm_model_plot_forecast_{current_date}_v{version}.pdf'
	plt.savefig(f'{plot_path}')
	
	plt.show()

	# # Create 4 subplots in a 2x2 grid
	# fig, ax = plt.subplots(2, 2, figsize=(14, 12))

	# # # Back-test the model
	# n_steps = 5
	# window_size = 60  # Use 60 days for training in each window
	# backtest_results = backtest_lstm(stock_data, lstm_model, scaler, n_steps, window_size)

	# # # Print back-test results
	# print("Back-Test Results:")
	# print(backtest_results.head())

	# analyze_backtest_results(backtest_results,ax)

	# plt.tight_layout()
	# plt.show()