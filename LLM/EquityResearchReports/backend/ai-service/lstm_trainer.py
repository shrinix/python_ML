# Import necessary libraries for LSTM model
# Place these lines at the very top of your script, before importing tensorflow/keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to use CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Optional: reduces TensorFlow log verbosity
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
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, Permute, Multiply, Lambda

from keras.optimizers import Adam, AdamW, RMSprop
from keras.optimizers import AdamW
import random
from keras.regularizers import l2, l1, l1_l2
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from utils import (
	get_max_version, get_timeframes_df, load_configuration_params, prepare_data_for_training, prepare_seq2seq_data,
	fetch_data_into_files, load_and_transform_data_by_ticker, add_engineered_features, get_scaler, check_data
)
from joblib import Parallel, delayed
from itertools import product
import tensorflow as tf
import joblib

def create_lstm_model(neurons=50, n_steps=1, n_features=2, dropout=0.2, optimizer='adam', 
					learning_rate=0.001, layers=1, l1_reg=0.005, l2_reg=0.01):
	#update the model to use layers
	
	model = Sequential()
	model.add(LSTM(neurons, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))   
	model.add(Dropout(dropout))
	for i in range(layers-1):
		model.add(LSTM(neurons, activation='relu', return_sequences=True))
		model.add(Dropout(dropout))
	model.add(LSTM(neurons, activation='relu', return_sequences=False))
	model.add(Dropout(dropout))
	model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
	if (optimizer == 'adam'):
		optimizer = Adam(learning_rate=learning_rate)
	elif (optimizer == 'adamw'):
		optimizer = AdamW(learning_rate=learning_rate)
	else:
		optimizer = RMSprop(learning_rate=learning_rate)
		
	model.compile(optimizer=optimizer, loss='mse')
	
	model_name = 'Simple LSTM Model'
	return model,model_name

def create_lstm_model_with_attention(neurons=50, n_steps=1, n_features=2, dropout=0.2, optimizer='adam',
					learning_rate=0.001, layers=1, l1_reg=0.005, l2_reg=0.01):
	"""
	Create a single-layer LSTM model with attention mechanism.
	"""

	# Input layer
	inputs = Input(shape=(n_steps, n_features))

	# Single LSTM layer (return_sequences=True for attention)
	lstm_out = LSTM(neurons, activation='relu', return_sequences=True)(inputs)
	lstm_out = Dropout(dropout)(lstm_out)

	# Attention mechanism
	# Compute attention scores
	attention_scores = Dense(1, activation='tanh')(lstm_out)
	attention_scores = Lambda(
		lambda x: tf.nn.softmax(x, axis=1),
		output_shape=lambda s: s
	)(attention_scores)
	# Multiply attention scores with LSTM output
	# Repeat attention_scores across the feature dimension to match lstm_out's shape
	attention_scores_repeated = Lambda(
		lambda x: tf.repeat(x, repeats=neurons, axis=-1),
		output_shape=lambda s: (s[0], s[1], neurons)
	)(attention_scores)
	context_vector = Multiply()([lstm_out, attention_scores_repeated])
	context_vector = Lambda(
		lambda x: tf.reduce_sum(x, axis=1),
		output_shape=lambda s: (s[0], s[2])
	)(context_vector)

	# Output layer
	output = Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(context_vector)

	model = Model(inputs=inputs, outputs=output)

	# Optimizer selection
	if optimizer == 'adam':
		optimizer_instance = Adam(learning_rate=learning_rate)
	elif optimizer == 'adamw':
		optimizer_instance = AdamW(learning_rate=learning_rate)
	else:
		optimizer_instance = RMSprop(learning_rate=learning_rate)

	model.compile(optimizer=optimizer_instance, loss='mse')
	model_name = 'Single-Layer LSTM with Attention'
	return model, model_name

def train_lstm_model(model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, n_features, n_steps, batch_size,  epochs, early_stopping, ax):
	print(f"# of features: {n_features} #of time-steps: {n_steps}")
	print('\n-------Shape of scaled train and test data after reshaping-------')
	print("X_train_scaled shape: ", X_train_scaled.shape)
	print("y_train_scaled shape: ", y_train_scaled.shape) 
	print("X_test_scaled shape: ", X_test_scaled.shape)
	print("y_test_scaled shape: ", y_test_scaled.shape)

	# Ensure the number of elements matches the shape
	assert X_train_scaled.shape[1] == n_features, f"Number of features does not match: expected {n_features}, got {X_train_scaled.shape[1]}"
	assert X_test_scaled.shape[1] == n_features, f"Number of features does not match: expected {n_features}, got {X_test_scaled.shape[1]}"
	assert y_train_scaled.shape[0] == X_train_scaled.shape[0], f"Number of samples does not match: expected {X_train_scaled.shape[0]}, got {y_train_scaled.shape[0]}"
	assert y_test_scaled.shape[0] == y_test_scaled.shape[0], f"Number of samples does not match: expected {y_test_scaled.shape[0]}, got {y_test_scaled.shape[0]}"

	# Reshape the input data for LSTM
	# LSTM expects the input data to be in a 3D format of [samples, timesteps, features]
	X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0],1,n_features)
	print("X_train_reshaped shape: ", X_train_reshaped.shape)
	print(X_train_reshaped[:5])
	#reshape y_train
	y_train_reshaped = y_train_scaled.reshape(y_train_scaled.shape[0],1)
	print("y_train_reshaped shape: ",y_train_reshaped.shape)
	print(y_train_reshaped[:5])

	# Set random seeds for reproducibility
	seed = 42
	np.random.seed(seed)
	random.seed(seed)
	tf.random.set_seed(seed)

	# Configure TensorFlow for deterministic operations
	# Note: This may impact performance
	tf.config.experimental.enable_op_determinism()
	# Enable XLA
	# Disable XLA compilation to avoid platform-related errors
	tf.config.optimizer.set_jit(False)

	# model_lstm.compile(optimizer=optimizer, loss='mse')
	# #add learning rate reduction
	# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=0.0001)
	# Train the model and capture the training history
	history = model.fit(X_train_reshaped, y_train_reshaped, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping])

	# Get the last epoch number
	last_epoch = len(history.history['loss'])
	print(f"Last Epoch: {last_epoch}")
	ax.plot(history.history['loss'], label='Training Loss')
	ax.plot(history.history['val_loss'], label='Validation Loss')
	ax.set_title('Training and Validation Loss')
	ax.set_xlabel('Epochs')
	ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, colors='black', grid_color='gray', grid_alpha=0.5)
	ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
	ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))  # Format y-axis ticks to 2 decimal places
	ax.set_ylabel('Loss')
	# Add legend to the plot
	ax.legend()

	# Display model training characteristics in a box on the plot
	train_loss = history.history['loss'][-1]
	val_loss = history.history['val_loss'][-1]
	textstr = '\n'.join((
		f'Training Loss: {train_loss:.4f}',
		f'Validation Loss: {val_loss:.4f}',
		f'Batch Size: {batch_size}',
		f'Epochs: {epochs}',
	))

	# Add a box with the text
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	# Adjust the position of the text box to align with the legend and add a slight horizontal gap
	ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=10,
			verticalalignment='bottom', horizontalalignment='left', bbox=props)
	
	return last_epoch

def predict_using_lstm(X_train_scaled, X_test_scaled, y_test_scaled, column_names, lstm_model, ax):
	print("X_test_scaled shape before reshaping: ", X_test_scaled.shape)
	n_steps = 1  #time_steps
	n_features = X_test_scaled.shape[1]

	X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], n_steps, n_features)
	print("Test data shape after reshaping: ", X_test_reshaped.shape)
	# print(X_test_reshaped[:5])

	# Make predictions
	predictions_lstm = lstm_model.predict(X_test_reshaped)
	
	return predictions_lstm

def forecast_n_steps_ahead_using_seq2seq_model(model, X_encoder_test, decoder_input_test, y_decoder_test, n_steps_out):
	"""
	Forecast multiple steps ahead using the Seq2Seq model.

	Parameters:
		model (keras.Model): Trained Seq2Seq model.
		X_encoder_test (numpy.ndarray): Encoder input data for testing.
		decoder_input_test (numpy.ndarray): Decoder input data for testing.
		y_decoder_test (numpy.ndarray): True output data for testing.
		n_steps_out (int): Number of output time steps to forecast.

	Returns:
		predictions (numpy.ndarray): Forecasted values.
	"""
	# Reshape the encoder input data
	X_encoder_test_reshaped = X_encoder_test.reshape(X_encoder_test.shape[0], X_encoder_test.shape[1], X_encoder_test.shape[2])
	
	# Make predictions
	predictions = model.predict([X_encoder_test_reshaped, decoder_input_test])
	print("Shape of predictions before reshaping:", predictions.shape)

	# Ensure predictions have the correct shape
	if predictions.shape[-1] == 1:  # If the last dimension is 1, squeeze it
		predictions_reshaped = predictions.squeeze(axis=-1)
	else:
		# Reshape predictions to match the expected output shape
		predictions_reshaped = predictions.reshape(predictions.shape[0], n_steps_out)
	
	return predictions_reshaped

def plot_predictions(train, test, predictions_lstm, model_name, ax):

	# Align the test DataFrame with the predictions DataFrame
	test_aligned = test.iloc[-len(predictions_lstm):]

	#plot the actual and predicted trends
	# print(train.head())
	ax.plot(train.index, train['Tgt'], label='Training Data', marker='o')
	ax.plot(test_aligned.index, test_aligned['Tgt'], label='Actual Trend', marker='o')
	# ax.plot(test_aligned.index, predictions_lstm['Trend_Predicted'], label='Predicted Trend', linestyle='--', marker='o')

	 # Plot each 'Trend_Predicted_tX' column
	for col in predictions_df.columns:
		if col.startswith('Trend_Predicted_t'):
			ax.plot(test_aligned.index, predictions_df[col], label=f'Predicted {col}', linestyle='--', marker='o')


	ax.set_title(f'Financial Time Series Trend Forecasting using: {model_name}')
	ax.set_xlabel('Date')
	ax.set_ylabel('Adj Close Price')
	ax.legend()

	 # Calculate performance metrics for each prediction column
	for col in predictions_df.columns:
		if col.startswith('Trend_Predicted_t'):
			mae = mean_absolute_error(test_aligned['Tgt'], predictions_df[col])
			mse = mean_squared_error(test_aligned['Tgt'], predictions_df[col])
			rmse = np.sqrt(mse)
			print(f"{col} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

	# Display model forecasting metrics in a box on the plot
	textstr = '\n'.join((
		f'MAE (t1): {mean_absolute_error(test_aligned["Tgt"], predictions_df["Trend_Predicted_t1"]):.4f}',
		f'MSE (t1): {mean_squared_error(test_aligned["Tgt"], predictions_df["Trend_Predicted_t1"]):.4f}',
		f'RMSE (t1): {np.sqrt(mean_squared_error(test_aligned["Tgt"], predictions_df["Trend_Predicted_t1"])):.4f}',
	))

	# Add a box with the text
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

	# # Adjust the position of the text box to align with the legend and add a slight horizontal gap
	ax.text(0.01, 0.85, textstr, transform=ax.transAxes, fontsize=10,
			verticalalignment='bottom', horizontalalignment='left', bbox=props)

# def evaluate_lstm_hyperparams(n, d, lr, opt, l, b, e, l1, l2, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
def evaluate_lstm_hyperparams(n, d, lr, e, l1, l2, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
	model, _ = create_lstm_model(neurons=n, n_steps=1, n_features=X_train_scaled.shape[1], dropout=d, optimizer='adam', 
								 learning_rate=lr, layers=1, l1_reg=l1, l2_reg=l2)
	n_steps = 1
	n_features = X_train_scaled.shape[1]
	X_train_scaled_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], n_steps, n_features))
	X_test_scaled_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], n_steps, n_features))
	model.fit(X_train_scaled_reshaped, y_train_scaled, batch_size=4, epochs=e, verbose=0)
	y_pred = model.predict(X_test_scaled_reshaped)
	mse = mean_squared_error(y_test_scaled, y_pred)
	print(f"Neurons: {n}, Dropout: {d}, Learning Rate: {lr}, Epochs: {e}, L1: {l1}, L2: {l2}, MSE: {mse:.4f}")
	return (n, d, lr, e, l1, l2, mse)

def grid_search_lstm_hyperparameters(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
	neurons = [50, 100]
	dropout = [0.1, 0.2]
	learning_rate = [0.001, 0.0001, 0.00001]
	# optimizer = ['adam', 'adamw']
	# layers = [1, 2]
	# batch_size = [4, 8]
	epochs = [10, 20, 50]
	l1_reg = [0.0001, 0.001]
	l2_reg = [0.0001, 0.001]

	# param_grid = list(product(neurons, dropout, learning_rate, optimizer, layers, batch_size, epochs, l1_reg, l2_reg))
	param_grid = list(product(neurons, dropout, learning_rate, epochs, l1_reg, l2_reg))

	results = Parallel(n_jobs=-1, backend="loky")(
		delayed(evaluate_lstm_hyperparams)(
			n, d, lr, e, l1, l2, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
		) for (n, d, lr, e, l1, l2) in param_grid
	)

	results_df = pd.DataFrame(results, columns=['Neurons', 'Dropout', 'Learning Rate', 'Epochs', 'L1', 'L2', 'MSE'])
	best_hyperparameters = results_df.loc[results_df['MSE'].idxmin()]
	print("Best Hyperparameters:")
	print(best_hyperparameters)
	return best_hyperparameters
if __name__ == '__main__':

	#create a plot with multiple sections
	fig, ax = plt.subplots(2,1, figsize=(14,12)) #width, height

	config_params = load_configuration_params()
	start_date = config_params['start_date']
	end_date = config_params['end_date']
	increment = config_params['increment']
	split_date = config_params['split_date']
	timeframes_df = get_timeframes_df(start_date, end_date, increment)

	load_from_files=False
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
		
	# merged_stock_data = merge_data(timeframes_df, source_files_folder)
	# #save merged_stock_data to a csv file
	# merged_stock_data.to_csv(f'{output_files_folder}/merged_stock_data.csv')

	#only one ticker presently
	#todo need tocheck if same LSTM works for multiple tickers
	stock_data = stock_data_dict[tickers[0]]
	stock_data['Tgt'] = stock_data['Adj Close'].shift(-1) # next day's price

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
		train_prepared_df, test_prepared_df, _, _ = prepare_data_for_training(stock_data, feature_columns, split_date, 'Tgt', disable_scaling)
		check_data(train_prepared_df, train_prepared_df.columns)
		check_data(test_prepared_df, test_prepared_df.columns)
	else:
		print("Scaling is enabled. Data will be scaled during preparation.")        
		train_prepared_scaled_df, test_prepared_scaled_df, feature_scaler, target_scaler = prepare_data_for_training(stock_data, feature_columns, split_date, 'Tgt', disable_scaling)

	print("train_prepared_df shape: ", train_prepared_scaled_df.shape)
	print("test_prepared_df shape: ", test_prepared_scaled_df.shape)

	#create X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled from train_prepared_df and test_prepared_df
	X_train_scaled = train_prepared_scaled_df.drop(columns=['Tgt']).values
	y_train_scaled = train_prepared_scaled_df['Tgt'].values
	X_test_scaled = test_prepared_scaled_df.drop(columns=['Tgt']).values
	y_test_scaled = test_prepared_scaled_df['Tgt'].values
	
	print("X_train_scaled shape: ", X_train_scaled.shape)
	print("y_train_scaled shape: ", y_train_scaled.shape)
	print("X_test_scaled shape: ", X_test_scaled.shape)
	print("y_test_scaled shape: ", y_test_scaled.shape)

	# best_hyperparameters = grid_search_lstm_hyperparameters(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

	n_steps = 1 #X_train_scaled.shape[0]#1 #X_train_reshaped.shape[1]
	n_features = X_train_scaled.shape[1]
	neurons = 100 #int(best_hyperparameters['Neurons']) #100
	dropout = 0.0 #int(best_hyperparameters['Dropout']) #0.1
	learning_rate = 0.0001 #int(best_hyperparameters['Learning_Rate']) #0.00001
	# optimizer = AdamW(learning_rate=learning_rate)
	# optimizer_name = str(best_hyperparameters['Optimizer']) #'adamw'
	optimizer_name = 'adamw'
	if optimizer_name == 'adam':
		optimizer = Adam(learning_rate=learning_rate)
	elif optimizer_name == 'adamw':
		optimizer = AdamW(learning_rate=learning_rate)

	# layers = int(best_hyperparameters['Layers'])#2
	layers = 2
	# batch_size = int(best_hyperparameters['Batch Size']) #4
	batch_size = 4

	epochs = 5 #int(best_hyperparameters['Epochs']) #500
	l1_reg = 0.000 #int(best_hyperparameters['L1']) #0.0005
	l2_reg = 0.0005 #int(best_hyperparameters['L2']) #0.0000

	# lstm_model, model_name = create_lstm_model(neurons=neurons, n_steps = n_steps, n_features = n_features, dropout=dropout, optimizer=optimizer, 
	# 							learning_rate=learning_rate, layers=layers, l1_reg=l1_reg, l2_reg=l2_reg)
	
	lstm_model, model_name = create_lstm_model_with_attention(neurons=neurons, n_steps = n_steps, n_features = n_features, dropout=dropout, optimizer=optimizer,
								learning_rate=learning_rate, layers=layers, l1_reg=l1_reg, l2_reg=l2_reg)
	print('-------Model Summary-------')
	print(lstm_model.summary())
	#check the shape of the input layer
	print("Input shape of the model: ", lstm_model.input_shape)
	print("Output shape of the model: ", lstm_model.output_shape)
	model = lstm_model
	
	#add early stopping
	patience = 20 #epochs
	early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
	training_start_time= datetime.now()
	last_epoch = train_lstm_model(model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, n_features, n_steps, batch_size, epochs, early_stopping, ax[0])

	print("Last epoch: ", last_epoch)

	training_end_time= datetime.now()

	predictions_scaled = predict_using_lstm(X_train_scaled, X_test_scaled, y_test_scaled, feature_columns, lstm_model, ax[0])
	print(predictions_scaled[:5])
	print("Shape of predictions:", predictions_scaled.shape)
	
	predictions_with_Xtest_df = pd.DataFrame(index=test_prepared_scaled_df.index[-len(test_prepared_scaled_df):])
	print("Shape of predictions_with_Xtest_df:", predictions_with_Xtest_df.shape)

	# Perform inverse transformation on predictions
	predictions_unscaled = target_scaler.inverse_transform(predictions_scaled)
	print("Shape of predictions_unscaled:", predictions_unscaled.shape)

	X_test_unscaled = feature_scaler.inverse_transform(X_test_scaled)
	print("Shape of X_test_unscaled:", X_test_unscaled.shape)

	#combine the unscaled predictions with the unscaled X_test and create a new dataframe
	predictions_df = pd.DataFrame(X_test_unscaled, columns=feature_columns)
	#Create a new column for each output time step
	for i in range(predictions_unscaled.shape[1]):
		predictions_df[f'Trend_Predicted_t{i+1}'] = predictions_unscaled[:, i]	
	# Debug: Print the final DataFrame
	print(predictions_df.tail())

	if disable_scaling == False:
		#Convert train_prepared_df_unscaled and test_prepared_df to arrays, unscale them and reconvert them to DataFrames
		train_prepared_index = train_prepared_scaled_df.index
		train_prepared_features_unscaled = feature_scaler.inverse_transform(train_prepared_scaled_df.drop(columns=['Tgt']).values)
		train_prepared_target_unscaled = target_scaler.inverse_transform(train_prepared_scaled_df['Tgt'].values.reshape(-1, 1))
		train_prepared_df_unscaled = pd.DataFrame(train_prepared_features_unscaled, columns=feature_columns)
		train_prepared_df_unscaled['Tgt'] = train_prepared_target_unscaled
		train_prepared_df_unscaled.set_index(train_prepared_index, inplace=True)
		#set index of train_prepared_df to the index of 
		test_prepared_features_unscaled = feature_scaler.inverse_transform(test_prepared_scaled_df.drop(columns=['Tgt']).values)
		test_prepared_target_unscaled = target_scaler.inverse_transform(test_prepared_scaled_df['Tgt'].values.reshape(-1, 1))
		test_prepared_df_unscaled = pd.DataFrame(test_prepared_features_unscaled, columns=feature_columns)
		test_prepared_df_unscaled['Tgt'] = test_prepared_target_unscaled
		test_prepared_df_unscaled.set_index(test_prepared_scaled_df.index, inplace=True)

	# Pass the DataFrame to the plot_predictions function
	plot_predictions(train_prepared_df_unscaled, test_prepared_df_unscaled, predictions_df, model_name, ax[1])

	plt.tight_layout()

	output_files_folder = config_params['output_files_folder']
	duration = training_end_time - training_start_time
	#save training parameters to a dataframe
	training_params = {'model_name': model_name,'neurons': neurons, 'n_features': n_features, 'dropout': dropout, 'optimizer': optimizer_name, 
						'learning_rate': learning_rate, 'l1_reg': l1_reg, 'l2_reg':l2_reg, 'num_layers': layers, 'batch_size': batch_size, 'epochs': epochs,
						'patience': patience, 'training_start_time': training_start_time, 'training_end_time': training_end_time, 'duration':duration, 
						'last_epoch': last_epoch, 'feature_columns': feature_columns}
	# Ensure all values in the dictionary are scalar or convert non-scalar values to strings
	training_params_cleaned = {key: (str(value) if isinstance(value, (list, dict)) else value) for key, value in training_params.items()}
	training_params_df = pd.DataFrame([training_params_cleaned])
	#save the plot to a pdf
	current_date = datetime.now().strftime('%Y-%m-%d')
	
	#do not look for yesterday's models.
	look_for_older_files=True
	file_prefix='lstm_model'
	version,_ = get_max_version(output_files_folder, file_prefix, look_for_older_files=look_for_older_files)
	version += 1 #increment the version

	model_path = f'{output_files_folder}/{file_prefix}_{current_date}_v{version}.keras'
	training_params_path = f'{output_files_folder}/{file_prefix}_{current_date}_training_params_v{version}.csv'

	#save the training parameters to a csv file
	training_params_df.to_csv(training_params_path)

	model.save(model_path)
	print("Model saved to disk as: ", model_path)

	plot_path = f'{output_files_folder}/{file_prefix}_plot_training_{current_date}_v{version}.pdf'
	plt.savefig(f'{plot_path}')

	# Save the scalers
	joblib.dump(feature_scaler, f'{output_files_folder}/{file_prefix}_feature_scaler_{current_date}_v{version}.save')
	joblib.dump(target_scaler, f'{output_files_folder}/{file_prefix}_target_scaler_{current_date}_v{version}.save')
	#------

	plt.show()
