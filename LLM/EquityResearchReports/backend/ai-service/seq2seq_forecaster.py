# Import necessary libraries for LSTM model
import tensorflow as tf
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
	check_data, update_features_after_prediction
)

def forecast_n_steps_ahead_using_seq2seq(X_test_scaled, y_test_scaled, lstm_model, feature_scaler, target_scaler, n_steps, column_names):
	# Initialize a DataFrame to store predictions
	future_predictions = pd.DataFrame(columns=column_names+['Trend_Predicted']) #[
	# 	'7_day_avg', '30_day_avg', 'Daily Ret', 'Volatility_7', 'Volatility_30', 'RSI',
	#    'delta', 'avg_gain', 'avg_loss', 'rs', 'ema_12', 'ema_26', 'MACD',
	#    'Signal_Line', 'Bollinger_Upper', 'Bollinger_Lower', 'rolling_mean',
	#    'rolling_std','Trend_Predicted']

	print("X_test_scaled shape: ", X_test_scaled.shape)
	print ("Feature columns: ", column_names)

	# Start with the last row of the test data
	last_row_scaled = X_test_scaled[-1, :]  # Use all columns. last column has already been removed
	#get last value from y_test_scaled
	previous_adj_close = y_test_scaled[-1]  # The last known adjusted close price
	#print last_row and previous_adj_close tog
	last_row_unscaled = feature_scaler.inverse_transform(last_row_scaled.reshape(1, -1))
	print("Unscaled last row: ", last_row_unscaled)

	for step in range(n_steps):
		previous_adj_close_unscaled = target_scaler.inverse_transform(previous_adj_close.reshape(1, -1))
		print("Unscaled previous adjusted close: ", previous_adj_close_unscaled)

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
		
		# Inverse transform the prediction to get the actual scale
		last_row_updated = np.append(last_row_scaled[:-1], predicted_value).reshape(1, -1)
		# Ensure the input to target_scaler.inverse_transform has the correct shape
		# input_for_inverse_transform = np.zeros((1, last_row.shape[0] + 1))  # Adjust size to match last_row + predicted_value
		# input_for_inverse_transform[0, :last_row.shape[0]] = last_row.flatten()
		# input_for_inverse_transform[0, -1] = predicted_value
		# #predicted_value_actual is predicted value of ADJ Close in the original scale
		# predicted_value_actual = target_scaler.inverse_transform(input_for_inverse_transform)[0, -1]
		predicted_value_actual = target_scaler.inverse_transform(prediction).reshape(1, -1)[0, 0]
		print(f"Step {step + 1}: Predicted value (actual) = {predicted_value_actual:.4f}")
        
		last_row_scaled = update_features_after_prediction(predicted_value, last_row_scaled, previous_adj_close, step, X_test_scaled, column_names)
		
		last_row_unscaled = feature_scaler.inverse_transform(last_row_scaled.reshape(1, -1))
		# Append the prediction and updated features to the DataFrame
		future_predictions_step = pd.DataFrame(last_row_unscaled, columns=column_names)
		future_predictions_step['Trend_Predicted'] = predicted_value_actual
		future_predictions = pd.concat([future_predictions, future_predictions_step], ignore_index=True)
		print(f"Future Predictions (Step {step + 1}):")
		previous_adj_close = predicted_value

	# Print the future predictions
	print("Future Predictions:")
	print(future_predictions)
	return future_predictions

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
		pd.date_range(start=test_prepared_scaled_df.index[-1], periods=n_steps + 1, freq='D')[1:],
		future_predictions['Trend_Predicted'],
		label='Predicted Trend (n steps)',
		linestyle='--',
		marker='o'
	)
	#add annotations
	for i in range(n_steps):
		ax.annotate(
			f"Step {i + 1}: {future_predictions['Trend_Predicted'].iloc[i]:.2f}",
			(pd.date_range(start=test_prepared_scaled_df.index[-1], periods=n_steps + 1, freq='D')[i + 1], future_predictions['Trend_Predicted'].iloc[i]),
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
	fig, ax = plt.subplots(2,1, figsize=(14,12)) #width, height

	config_params = load_configuration_params()
	start_date = config_params['start_date']
	end_date = config_params['end_date']
	increment = config_params['increment']
	timeframes_df = get_timeframes_df(start_date, end_date, increment)
	print(timeframes_df)

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
	# Calculate rolling averages
	stock_data['Tgt'] = stock_data['Adj Close'].shift(-1) # next day's price
	stock_data.dropna(inplace=True)
	
	print("Before adding engineered features: ", stock_data.shape)
	# Add engineered features
	stock_data,feature_columns = add_engineered_features(stock_data)
	stock_data.dropna(inplace=True)
	print("Feature columns after adding engineered features: ", feature_columns)
	print("Stock data columns after adding engineered features: ", stock_data.columns)
	print("Shape of stock data after adding engineered features: ", stock_data.shape)

	disable_scaling = False #When using k-fold, scaling is done after KFold splitting.
	print("Preparing data for training...")
	feature_scaler, target_scaler = None, None

	if disable_scaling:
		print("Scaling is disabled. Data will not be scaled during preparation.")
		#discard scaler outputs
		#These checks will work only if data is not scaled
		train_prepared_df, test_prepared_df, _, _ = prepare_data_for_training(stock_data, feature_columns, 'Tgt', disable_scaling)
		check_data(train_prepared_df, train_prepared_df.columns)
		check_data(test_prepared_df, test_prepared_df.columns)
	else:
		print("Scaling is enabled. Data will be scaled during preparation.")        
		train_prepared_scaled_df, test_prepared_scaled_df, feature_scaler, target_scaler = prepare_data_for_training(stock_data, feature_columns, 'Tgt', disable_scaling)

	print("train_prepared_df shape: ", train_prepared_scaled_df.shape)
	print("test_prepared_df shape: ", test_prepared_scaled_df.shape)
	output_files_folder = config_params['output_files_folder']

	#create X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled from train_prepared_df and test_prepared_df
	X_train_scaled = train_prepared_scaled_df.drop(columns=['Tgt']).values
	y_train_scaled = train_prepared_scaled_df['Tgt'].values
	X_test_scaled = test_prepared_scaled_df.drop(columns=['Tgt']).values
	y_test_scaled = test_prepared_scaled_df['Tgt'].values
	
	print("X_train_scaled shape: ", X_train_scaled.shape)
	print("y_train_scaled shape: ", y_train_scaled.shape)
	print("X_test_scaled shape: ", X_test_scaled.shape)
	print("y_test_scaled shape: ", y_test_scaled.shape)

	# load model
	#Look for yesterday's models.
	look_for_older_files=True
	current_date = datetime.now().strftime('%Y-%m-%d')
	file_prefix='seq2seq_model_'
	version,date = get_max_version(output_files_folder, file_prefix, look_for_older_files=True)
	model_path = f'{output_files_folder}/{file_prefix}{date}_v{version}.keras'
	print(f"Model path: {model_path}")
	lstm_model = None
	if os.path.exists(model_path):            
		lstm_model = tf.keras.Sequential([
			tf.keras.models.load_model(model_path)
		])
		print(f"Loaded model: {lstm_model.name} from {model_path}")
		lstm_model.summary()
		#print the shape of input layer
		input_layer = lstm_model.layers[0]
		print(f"Input layer shape: {input_layer.input_shape}")
		output_layer = lstm_model.layers[-1]
		print(f"Output layer shape: {output_layer.output_shape}")

	else:
		print(f"Model not found at {model_path}. Creating new model.")
	
	# # forecast_data_using_lstm(X_test_scaled, y_test_scaled, lstm_model, ax)
	n_steps = 10  # Number of steps to forecast

	future_predictions = forecast_n_steps_ahead_using_seq2seq(X_test_scaled, y_test_scaled, lstm_model, feature_scaler, target_scaler, n_steps, feature_columns)
	print("Future Predictions:")
	print(future_predictions.head())
	#save the future predictions to a file
	future_predictions.to_csv(f'{output_files_folder}/future_predictions.csv', index=False)

	# plot_metrics(future_predictions, train, test, n_steps, f'Financial Time Series Trend Forecasting ({n_steps} ahead)', ax[0])
	plot_future_trend(future_predictions, test_prepared_scaled_df, n_steps, f'FutureTrend Forecasting ({n_steps} ahead)', ax[1])
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