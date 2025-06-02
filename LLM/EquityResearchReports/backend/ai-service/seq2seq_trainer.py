# Import necessary libraries for LSTM model
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
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
from keras.layers import Input, LSTM, Dense, Dropout, Softmax, Dot, Activation
from keras.optimizers import Adam, AdamW, RMSprop
from keras.optimizers import AdamW
import random
import os
from keras.regularizers import l2, l1, l1_l2
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from utils import (
	get_max_version, get_timeframes_df, load_configuration_params, prepare_data_for_training, create_overlapping_arrays,
	fetch_data_into_files, load_and_transform_data_by_ticker, add_engineered_features, get_scaler,
	reverse_lookup,compare_dataframes,validate_nstep_folding_of_data,validate_feature_scalers, validate_data_processing,
	plot_attention_weights,scalers_dict,validate_scalers_entire_dataset,check_data,
	validate_nstep_folding_of_data_with_dataframes, convert_dataframe_to_3d_array,
	series_to_supervised,update_features_after_prediction_seq2seq,test_update_features_after_prediction_seq2seq_all
)
from keras.layers import Attention, Concatenate, Permute
from keras.layers import RepeatVector, TimeDistributed
from keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate
from keras.models import Model
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.layers import Bidirectional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.dates as mdates

def create_seq2seq_model(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=50, dropout=0.2, l1_reg=0.01, l2_reg=0.01, use_teacher_forcing=False):
	"""
	Create a Seq2Seq model with or without teacher forcing.

	Parameters:
		optimizer_name (str): Name of the optimizer.
		learning_rate (float): Learning rate for the optimizer.
		n_steps_in (int): Number of input timesteps.
		n_steps_out (int): Number of output timesteps.
		n_features (int): Number of features.
		neurons (int): Number of LSTM units.
		dropout (float): Dropout rate.
		l1_reg (float): L1 regularization.
		l2_reg (float): L2 regularization.
		use_teacher_forcing (bool): Whether to use teacher forcing.

	Returns:
		model (keras.Model): Compiled Seq2Seq model.
	"""
	# Set random seeds for reproducibility
	seed = 42
	np.random.seed(seed)
	tf.random.set_seed(seed)

	# Enable deterministic operations in TensorFlow
	tf.config.experimental.enable_op_determinism()

	# Encoder
	encoder_inputs = Input(shape=(n_steps_in, n_features), name="encoder_input")
	encoder_lstm = LSTM(neurons, activation='relu', return_state=True, dropout=dropout, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name="encoder_lstm")
	encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
	encoder_states = [state_h, state_c]

	if use_teacher_forcing:
		# Decoder with teacher forcing
		decoder_inputs = Input(shape=(n_steps_out, n_features), name="decoder_input")
		decoder_lstm = LSTM(neurons, activation='relu', return_sequences=True, return_state=True, dropout=dropout, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name="decoder_lstm")
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
		decoder_dense = Dense(1, activation="relu", name="decoder_dense")
		decoder_outputs = decoder_dense(decoder_outputs)

		# Define the model with teacher forcing
		model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name="seq2seq_model_with_teacher_forcing")
	else:
		# Decoder without teacher forcing
		decoder_inputs = RepeatVector(n_steps_out)(state_h)  # Repeat the encoder's final hidden state for n_steps_out timesteps
		decoder_outputs = LSTM(neurons, activation='relu', return_sequences=True, dropout=dropout, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name="decoder_lstm_no_teacher_forcing")(decoder_inputs, initial_state=encoder_states)
		decoder_dense = Dense(1, activation="relu", name="decoder_dense")
		decoder_outputs = decoder_dense(decoder_outputs)

		# Define the model without teacher forcing
		model = Model(inputs=encoder_inputs, outputs=decoder_outputs, name="seq2seq_model_without_teacher_forcing")

	# Compile the model
	optimizer = Adam(learning_rate=learning_rate) if optimizer_name == "adam" else optimizer_name
	model.compile(optimizer=optimizer, loss="mse")

	# Debugging: Print model summary
	print("Model Summary:")
	model.summary()

	return model

# # def create_seq2seq_model_with_attention(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=50, dropout=0.2, l1_reg=0.01, l2_reg=0.01, use_teacher_forcing=False):
# 	"""
# 	Create a Seq2Seq model with Attention, with or without teacher forcing.

# 	Parameters:
# 		optimizer_name (str): Name of the optimizer ('adam', 'rmsprop').
# 		learning_rate (float): Learning rate for the optimizer.
# 		n_steps_in (int): Number of input timesteps.
# 		n_steps_out (int): Number of output timesteps.
# 		n_features (int): Number of features in the input data.
# 		neurons (int): Number of LSTM units in each layer.
# 		dropout (float): Dropout rate for regularization.
# 		l1_reg (float): L1 regularization factor.
# 		l2_reg (float): L2 regularization factor.
# 		use_teacher_forcing (bool): Whether to use teacher forcing.

# 	Returns:
# 		model (keras.Model): Compiled Seq2Seq model with attention.
# 	"""
# 	# Encoder
# 	encoder_inputs = Input(shape=(n_steps_in, n_features), name="encoder_inputs")
# 	encoder_outputs, state_h, state_c = LSTM(
# 		neurons,
# 		activation='relu',
# 		return_sequences=True,
# 		return_state=True,
# 		dropout=dropout,
# 		kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
# 		name="encoder_lstm"
# 	)(encoder_inputs)

# 	encoder_states = [state_h, state_c]

# 	if use_teacher_forcing:
# 		# Decoder with teacher forcing
# 		decoder_inputs = Input(shape=(n_steps_out, n_features), name="decoder_inputs")
# 		decoder_lstm_outputs, _, _ = LSTM(
# 			neurons,
# 			activation='relu',
# 			return_sequences=True,
# 			return_state=True,
# 			dropout=dropout,
# 			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
# 			name="decoder_lstm"
# 		)(decoder_inputs, initial_state=encoder_states)

# 		# print("Shape of encoder_outputs:", encoder_outputs.shape)  # Expected: (batch_size, n_steps_in, neurons)
# 		# print("Shape of decoder_lstm_outputs:", decoder_lstm_outputs.shape)  # Expected: (batch_size, n_steps_out, neurons)

# 		# Custom attention mechanism
# 		# Compute attention scores as the dot product between decoder outputs and encoder outputs
# 		attention_scores = Dot(axes=[2, 2], name="attention_scores")([decoder_lstm_outputs, encoder_outputs])  # Shape: (batch_size, n_steps_out, n_steps_in)
		
# 		# print("Shape of attention_scores:", attention_scores.shape)  # Expected: (batch_size, n_steps_out, n_steps_in)

# 		# Normalize the attention scores to get attention weights
# 		attention_weights = Activation('softmax', name="attention_weights")(attention_scores)  # Shape: (batch_size, n_steps_out, n_steps_in)
# 		# Compute the context vector as a weighted sum of encoder outputs
# 		context_vector = Dot(axes=[2, 1], name="context_vector")([attention_weights, encoder_outputs])  # Shape: (batch_size, n_steps_out, neurons)

# 		# Concatenate the context vector with the decoder outputs
# 		decoder_combined_context = Concatenate(name="decoder_combined_context")([decoder_lstm_outputs, context_vector])

# 		# Dense layer to produce the final output
# 		decoder_dense = TimeDistributed(Dense(
# 			1,  # Output layer (1 feature for the target column)
# 			activation='linear',
# 			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
# 			name="decoder_dense"
# 		))(decoder_combined_context)

# 		# Define the model with teacher forcing
# 		model = Model([encoder_inputs, decoder_inputs], decoder_dense, name="seq2seq_model_with_attention_teacher_forcing")
# 	else:
# 		# Decoder without teacher forcing
# 		decoder_inputs = RepeatVector(n_steps_out)(state_h)  # Repeat the encoder's final hidden state for n_steps_out timesteps
# 		decoder_lstm_outputs = LSTM(
# 			neurons,
# 			activation='relu',
# 			return_sequences=True,
# 			dropout=dropout,
# 			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
# 			name="decoder_lstm_no_teacher_forcing"
# 		)(decoder_inputs, initial_state=encoder_states)

# 		# Custom Attention mechanism
# 		# print("Shape of encoder_outputs:", encoder_outputs.shape)  # Expected: (batch_size, n_steps_in, neurons)
# 		# print("Shape of decoder_lstm_outputs:", decoder_lstm_outputs.shape)  # Expected: (batch_size, n_steps_out, neurons)

# 		# Custom attention mechanism
# 		# Compute attention scores as the dot product between decoder outputs and encoder outputs
# 		attention_scores = Dot(axes=[2, 2], name="attention_scores")([decoder_lstm_outputs, encoder_outputs])  # Shape: (batch_size, n_steps_out, n_steps_in)
		
# 		# print("Shape of attention_scores:", attention_scores.shape)  # Expected: (batch_size, n_steps_out, n_steps_in)
		
# 		# Normalize the attention scores to get attention weights
# 		attention_weights = Activation('softmax', name="attention_weights")(attention_scores)  # Shape: (batch_size, n_steps_out, n_steps_in)
# 		# Compute the context vector as a weighted sum of encoder outputs
# 		context_vector = Dot(axes=[2, 1], name="context_vector")([attention_weights, encoder_outputs])  # Shape: (batch_size, n_steps_out, neurons)

# 		# Concatenate the context vector with the decoder outputs
# 		decoder_combined_context = Concatenate(name="decoder_combined_context")([decoder_lstm_outputs, context_vector])

# 		# Dense layer to produce the final output
# 		decoder_dense = TimeDistributed(Dense(
# 			1,  # Output layer (1 feature for the target column)
# 			activation='linear',
# 			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
# 			name="decoder_dense"
# 		))(decoder_combined_context)

# 		# Define the model without teacher forcing
# 		model = Model(encoder_inputs, decoder_dense, name="seq2seq_model_with_attention_no_teacher_forcing")

# 	# Select the optimizer
# 	if optimizer_name == 'adam':
# 		optimizer = Adam(learning_rate)
# 	else:
# 		optimizer = RMSprop(learning_rate)

# 	# Compile the model
# 	model.compile(optimizer=optimizer, loss='mse')

# 	# Debugging: Print model summary
# 	print("Model Summary:")
# 	model.summary()

# 	return model

#function to create a regular LSTM sequence to sequence model with attention
def create_seq2seq_model_with_attention(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=50, dropout=0.2, l1_reg=0.01, l2_reg=0.01, use_teacher_forcing=False):
	"""
	Create a Seq2Seq model with Attention, with or without teacher forcing.

	Parameters:
		optimizer_name (str): Name of the optimizer ('adam', 'rmsprop').
		learning_rate (float): Learning rate for the optimizer.
		n_steps_in (int): Number of input time steps.
		n_steps_out (int): Number of output time steps (forecast horizon).
		n_features (int): Number of features in the input data.
		neurons (int): Number of LSTM units in each layer.
		dropout (float): Dropout rate for regularization.
		l1_reg (float): L1 regularization factor.
		l2_reg (float): L2 regularization factor.
		use_teacher_forcing (bool): Whether to use teacher forcing.

	Returns:
		model (keras.Model): Compiled Seq2Seq model with attention.
	"""
	# Set random seeds for reproducibility
	seed = 42
	np.random.seed(seed)
	tf.random.set_seed(seed)

	# Enable deterministic operations in TensorFlow
	tf.config.experimental.enable_op_determinism()

	# Encoder
	encoder_inputs = Input(shape=(n_steps_in, n_features), name="encoder_inputs")
	encoder_outputs, state_h, state_c = LSTM(
		neurons,
		activation='relu',
		return_sequences=True,
		return_state=True,
		dropout=dropout,
		kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
		name="encoder_lstm"
	)(encoder_inputs)

	encoder_states = [state_h, state_c]

	if use_teacher_forcing:
		# Decoder with teacher forcing
		decoder_inputs = Input(shape=(n_steps_out, n_features), name="decoder_inputs")
		decoder_lstm_outputs, _, _ = LSTM(
			neurons,
			activation='relu',
			return_sequences=True,
			return_state=True,
			dropout=dropout,
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			name="decoder_lstm"
		)(decoder_inputs, initial_state=encoder_states)

		print("Shape of encoder_outputs:", encoder_outputs.shape)  # Expected: (batch_size, n_steps_in, neurons)
		print("Shape of decoder_lstm_outputs:", decoder_lstm_outputs.shape)  # Expected: (batch_size, n_steps_out, neurons)

		# Custom attention mechanism
		# Compute attention scores as the dot product between decoder outputs and encoder outputs
		attention_scores = Dot(axes=[2, 2], name="attention_scores")([decoder_lstm_outputs, encoder_outputs])  # Shape: (batch_size, n_steps_out, n_steps_in)
		
		print("Shape of attention_scores:", attention_scores.shape)  # Expected: (batch_size, n_steps_out, n_steps_in)

		# Normalize the attention scores to get attention weights
		attention_weights = Activation('softmax', name="attention_weights")(attention_scores)  # Shape: (batch_size, n_steps_out, n_steps_in)
		
		# Compute the context vector as a weighted sum of encoder outputs
		context_vector = Dot(axes=[2, 1], name="context_vector")([attention_weights, encoder_outputs])
		# Shape: (batch_size, n_steps_out, neurons)
		print("Shape of context_vector:", context_vector.shape)  # Expected: (batch_size, n_steps_out, neurons)
		# Concatenate the context vector with the decoder outputs
		decoder_combined_context = Concatenate(name="decoder_combined_context")([decoder_lstm_outputs, context_vector])
		# Shape: (batch_size, n_steps_out, 2 * neurons)
		print("Shape of decoder_combined_context:", decoder_combined_context.shape)  # Expected: (batch_size, n_steps_out, 2 * neurons)
		# Dense layer to produce the final output
		decoder_dense = TimeDistributed(Dense(
			1,
			activation='relu',
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
		), name="decoder_dense")(decoder_combined_context)
		# Define the model with teacher forcing
		model = Model([encoder_inputs, decoder_inputs], decoder_dense, name="seq2seq_model_with_attention_and_teacher_forcing")
	else:
		# Decoder without teacher forcing
		decoder_inputs = RepeatVector(n_steps_out)(state_h)
		# Repeat the encoder's final hidden state for n_steps_out timesteps
		decoder_lstm_outputs = LSTM(
			neurons,
			activation='relu',
			return_sequences=True,
			dropout=dropout,
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			name="decoder_lstm_no_teacher_forcing"
		)(decoder_inputs, initial_state=encoder_states)
		# Custom Attention mechanism
		print("Shape of encoder_outputs:", encoder_outputs.shape)  # Expected: (batch_size, n_steps_in, neurons)
		print("Shape of decoder_lstm_outputs:", decoder_lstm_outputs.shape)  # Expected: (batch_size, n_steps_out, neurons)
		# Custom attention mechanism
		# Compute attention scores as the dot product between decoder outputs and encoder outputs
		attention_scores = Dot(axes=[2, 2], name="attention_scores")([decoder_lstm_outputs, encoder_outputs])  # Shape: (batch_size, n_steps_out, n_steps_in)
		print("Shape of attention_scores:", attention_scores.shape)  # Expected: (batch_size, n_steps_out, n_steps_in)
		# Normalize the attention scores to get attention weights
		attention_weights = Activation('softmax', name="attention_weights")(attention_scores)  # Shape: (batch_size, n_steps_out, n_steps_in)
		# Compute the context vector as a weighted sum of encoder outputs
		context_vector = Dot(axes=[2, 1], name="context_vector")([attention_weights, encoder_outputs])  # Shape: (batch_size, n_steps_out, neurons)
		print("Shape of context_vector:", context_vector.shape)  # Expected: (batch_size, n_steps_out, neurons)
		# Concatenate the context vector with the decoder outputs
		decoder_combined_context = Concatenate(name="decoder_combined_context")([decoder_lstm_outputs, context_vector])
		# Shape: (batch_size, n_steps_out, 2 * neurons)
		print("Shape of decoder_combined_context:", decoder_combined_context.shape)  # Expected: (batch_size, n_steps_out, 2 * neurons)
		# Dense layer to produce the final output
		decoder_dense = TimeDistributed(Dense(
			1,
			activation='relu',
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
		), name="decoder_dense")(decoder_combined_context)
		# Define the model without teacher forcing
		model = Model(encoder_inputs, decoder_dense, name="seq2seq_model_with_attention_and_without_teacher_forcing")
	# Select the optimizer
	if optimizer_name == 'adam':
		optimizer = Adam(learning_rate)
	elif optimizer_name == 'adamw':
		optimizer = AdamW(learning_rate)
	else:
		optimizer = RMSprop(learning_rate)
	# Compile the model
	model.compile(optimizer=optimizer, loss='mse')
	# Debugging: Print model summary
	print("Model Summary:")
	model.summary()
	return model

def create_bidirectional_seq2seq_with_attention(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=50, dropout=0.2, l1_reg=0.01, l2_reg=0.01, use_teacher_forcing=False):
	"""
	Create a Bidirectional Seq2Seq model with Attention, with or without teacher forcing.

	Parameters:
		optimizer_name (str): Name of the optimizer ('adam', 'rmsprop').
		learning_rate (float): Learning rate for the optimizer.
		n_steps_in (int): Number of input time steps.
		n_steps_out (int): Number of output time steps (forecast horizon).
		n_features (int): Number of features in the input data.
		neurons (int): Number of LSTM units in each layer.
		dropout (float): Dropout rate for regularization.
		l1_reg (float): L1 regularization factor.
		l2_reg (float): L2 regularization factor.
		use_teacher_forcing (bool): Whether to use teacher forcing.

	Returns:
		model (keras.Model): Compiled Seq2Seq model.
	"""

	# Set random seeds for reproducibility
	seed = 42
	np.random.seed(seed)
	tf.random.set_seed(seed)

	# Enable deterministic operations in TensorFlow
	tf.config.experimental.enable_op_determinism()

	# Encoder
	encoder_inputs = Input(shape=(n_steps_in, n_features), name="encoder_inputs")
	encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(
		LSTM(
			neurons,
			activation='relu',
			return_sequences=True,
			return_state=True,
			dropout=dropout,
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			name="encoder_lstm"
		),
		name="bidirectional_encoder_lstm"
	)(encoder_inputs)

	# Combine forward and backward states
	state_h = Concatenate(name="encoder_state_h")([forward_h, backward_h])
	state_c = Concatenate(name="encoder_state_c")([forward_c, backward_c])

	if use_teacher_forcing:
		# Decoder with teacher forcing
		decoder_inputs = Input(shape=(n_steps_out, n_features), name="decoder_inputs")
		decoder_lstm, _, _ = LSTM(
			neurons * 2,  # Double the number of neurons to match the concatenated encoder states
			activation='relu',
			return_sequences=True,
			return_state=True,
			dropout=dropout,
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			name="decoder_lstm"
		)(decoder_inputs, initial_state=[state_h, state_c])

		# Attention
		attention = Attention(name="attention_layer")([decoder_lstm, encoder_outputs])
		decoder_combined_context = Concatenate(name="decoder_combined_context")([decoder_lstm, attention])

		# Dense layer to produce the final output
		decoder_dense = TimeDistributed(Dense(
			1,  # Output layer (1 feature for the target column)
			activation='linear',
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			name="decoder_dense"
		))(decoder_combined_context)

		# Define the model with teacher forcing
		model = Model([encoder_inputs, decoder_inputs], decoder_dense)
	else:
		# Decoder without teacher forcing
		decoder_inputs = RepeatVector(n_steps_out)(state_h)  # Repeat the encoder's final hidden state for n_steps_out timesteps
		decoder_outputs = LSTM(
			neurons * 2,  # Double the number of neurons to match the concatenated encoder states
			activation='relu', # Use ReLU to enforce non-negative predictions
			return_sequences=True,
			dropout=dropout,
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			name="decoder_lstm"
		)(decoder_inputs, initial_state=[state_h, state_c])

		# Attention
		attention = Attention(name="attention_layer")([decoder_outputs, encoder_outputs])
		decoder_combined_context = Concatenate(name="decoder_combined_context")([decoder_outputs, attention])

		# Dense layer to produce the final output
		decoder_dense = TimeDistributed(Dense(
			1,  # Output layer (1 feature for the target column)
			activation='relu', # Use ReLU to enforce non-negative predictions
			kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
			name="decoder_dense"
		))(decoder_combined_context)

		# Define the model without teacher forcing
		model = Model(encoder_inputs, decoder_dense)

	# Select the optimizer
	if optimizer_name == 'adam':
		optimizer = Adam(learning_rate)
	else:
		optimizer = RMSprop(learning_rate)

	# Compile the model
	model.compile(optimizer=optimizer, loss='mse')

	return model

# Define the model creation function
def create_model(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=50, dropout=0.2, l1_reg=0.01, l2_reg=0.01, use_teacher_forcing=False):
	# model = create_seq2seq_model_with_attention(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=50, dropout=0.2, l1_reg=0.01, l2_reg=0.01, use_teacher_forcing=False)
	# model_name = "seq2seq_model_with_attention_no_teacher_forcing"
	if use_teacher_forcing:
		# model = create_seq2seq_model(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=neurons, dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg, use_teacher_forcing=True)
		# model_name = "seq2seq_model_with_teacher_forcing"
		# model = create_bidirectional_seq2seq_with_attention(
		#     optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=False)
		# model_name = "bidirectional_seq2seq_model_with_attention_no_teacher_forcing"
		model = create_seq2seq_model_with_attention(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=neurons, dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg, use_teacher_forcing=True)
		model_name = "seq2seq_model_with_attention_and_teacher_forcing"
	else:
		# model = create_seq2seq_model(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=neurons, dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg, use_teacher_forcing=False)
		# model_name = "seq2seq_model_no_teacher_forcing"
		# model = create_bidirectional_seq2seq_with_attention(
		#     optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=False)
		# model_name = "bidirectional_seq2seq_model_with_attention_no_teacher_forcing"
		model = create_seq2seq_model_with_attention(optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=neurons, dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg, use_teacher_forcing=False)
		model_name = "seq2seq_model_with_attention_and_without_teacher_forcing"

	print("Model Summary:")
	print(model.summary())
	return model, model_name

#TODO: Add determinism and remove plotting functions
def train_seq2seq_with_timefolding_teacher_forcing(model_fn, X_encoder_train, decoder_input_train, y_decoder_train, feature_columns,
												   n_splits, batch_size, epochs, patience, lr_scheduler, ax=None):
	"""
	Train the Seq2Seq model using k-fold cross-validation with teacher forcing.

	Parameters:
		model_fn (function): Function to create a new Seq2Seq model.
		X_encoder_train (numpy.ndarray): Encoder input data.
		decoder_input_train (numpy.ndarray): Decoder input data.
		y_decoder_train (numpy.ndarray): Ground truth output data.
		n_splits (int): Number of folds for k-fold cross-validation.
		batch_size (int): Batch size for training.
		epochs (int): Number of epochs for training.
		patience (int): Early stopping patience.
		ax (matplotlib.axes.Axes): Axis for plotting training and validation loss.

	Returns:
		dict: Dictionary containing average RMSE and validation losses for all folds.
	"""

	# Set random seeds for reproducibility
	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

	# Configure TensorFlow for deterministic operations
	tf.config.experimental.enable_op_determinism()

	fold = 1
	rmse_scores = []
	val_losses = []
	tscv = TimeSeriesSplit(n_splits=n_splits)

	print("Shape of X_encoder_train:", X_encoder_train.shape)
	print("n_steps_in:", X_encoder_train.shape[1])
	print("n_steps_out:", y_decoder_train.shape[1])
	scalers = {}  # Cache the scalers for use across folds

	# Step 1: Scale the data
	# Scale the train and validation sets using the same scaler
	X_scaler = MinMaxScaler(feature_range=(0, 1))
	X_scaler.fit(X_encoder_train.reshape(-1, X_encoder_train.shape[2]))
	scalers.update({'X_scaler': X_scaler})
	X_encoder_scaled = X_scaler.transform(X_encoder_train.reshape(-1, X_encoder_train.shape[2])).reshape(X_encoder_train.shape)
	y_scaler = MinMaxScaler(feature_range=(0, 1))
	y_scaler.fit(y_decoder_train.reshape(-1, 1))
	scalers.update({'y_scaler': y_scaler})
	y_decoder_scaled = y_scaler.transform(y_decoder_train.reshape(-1, 1)).reshape(y_decoder_train.shape)

	# Validate dataset sizes
	assert X_encoder_scaled.shape[0] == y_decoder_scaled.shape[0], "Mismatch in dataset sizes between X_encoder_scaled and y_decoder_scaled."

	# Step 1: Perform TimeSeriesSplit
	for fold, (train_index, val_index) in enumerate(tscv.split(X_encoder_scaled), start=1):
		# Validate indices
		if max(train_index) >= X_encoder_scaled.shape[0] or max(val_index) >= X_encoder_scaled.shape[0]:
			raise IndexError(f"Train or validation index is out of bounds in fold {fold}")

		print(f"\n--- Fold {fold} of {n_splits} ---")
		print(f"Fold {fold}: Train size: {len(train_index)}, Validation size: {len(val_index)}")
		print(f"Train indices: {train_index}")
		print(f"Validation indices: {val_index}")

		# Split data into training and validation sets
		X_train, X_val = X_encoder_scaled[train_index], X_encoder_scaled[val_index]
		y_train, y_val = y_decoder_scaled[train_index], y_decoder_scaled[val_index]

		print("Shape of X_train:", X_train.shape)
		print("Shape of X_val:", X_val.shape)
		print("Shape of y_train:", y_train.shape)
		print("Shape of y_val:", y_val.shape)

		# Create a new model for each fold
		model, model_name = model_fn()

		# Define early stopping
		early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

		# Train the model
		history = model.fit(
			X_train,
			y_train,
			validation_data=(X_val, y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=[early_stopping, lr_scheduler],
			verbose=1
		)

		# Evaluate the model on the validation set
		val_predictions = model.predict(X_val)
		print("Shape of val_predictions:", val_predictions.shape)  # Expected: (batch_size, n_steps_out, 1)

		# Reshape or squeeze the predictions and ground truth to remove the last dimension
		val_predictions = val_predictions.squeeze(axis=-1)  # Shape: (batch_size, n_steps_out)
		y_val = y_val.reshape(val_predictions.shape)  # Reshape y_val to (batch_size, n_steps_out)

		# Compute RMSE
		val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
		rmse_scores.append(val_rmse)
		val_losses.append(min(history.history['val_loss']))
		print(f"Fold {fold} RMSE: {val_rmse:.4f}")
		fold += 1

		# Optionally, plot training and validation loss for this fold
		if ax is not None:
			ax.plot(history.history['loss'], label=f'Fold {fold} Training Loss')
			ax.plot(history.history['val_loss'], label=f'Fold {fold} Validation Loss')

	# Calculate average RMSE and validation loss across all folds
	avg_rmse = np.mean(rmse_scores)
	avg_val_loss = np.mean(val_losses)
	print(f"\nAverage RMSE across {n_splits} folds: {avg_rmse:.4f}")
	print(f"Average Validation Loss across {n_splits} folds: {avg_val_loss:.4f}")

	return avg_rmse, avg_val_loss, rmse_scores, val_losses

#TODO: Add determinism and remove plotting functions
def train_seq2seq_with_timefolding_teacher_forcing(model, X_encoder_train, decoder_input_train, y_decoder_train, feature_columns,
												   n_splits, batch_size, epochs, patience, lr_scheduler, ax=None):
	"""
	Train the Seq2Seq model using k-fold cross-validation with teacher forcing.

	Parameters:
		model_fn (function): Function to create a new Seq2Seq model.
		X_encoder_train (numpy.ndarray): Encoder input data.
		decoder_input_train (numpy.ndarray): Decoder input data.
		y_decoder_train (numpy.ndarray): Ground truth output data.
		n_splits (int): Number of folds for k-fold cross-validation.
		batch_size (int): Batch size for training.
		epochs (int): Number of epochs for training.
		patience (int): Early stopping patience.
		ax (matplotlib.axes.Axes): Axis for plotting training and validation loss.

	Returns:
		dict: Dictionary containing average RMSE and validation losses for all folds.
	"""

	# Set random seeds for reproducibility
	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

	# Configure TensorFlow for deterministic operations
	tf.config.experimental.enable_op_determinism()

	fold = 1
	rmse_scores = []
	val_losses = []
	tscv = TimeSeriesSplit(n_splits=n_splits)

	print("Shape of X_encoder_train:", X_encoder_train.shape)
	print("n_steps_in:", X_encoder_train.shape[1])
	print("n_steps_out:", y_decoder_train.shape[1])
	scalers = {}  # Cache the scalers for use across folds

	# Step 1: Scale the data
	# Scale the train and validation sets using the same scaler
	X_scaler = MinMaxScaler(feature_range=(0, 1))
	X_scaler.fit(X_encoder_train.reshape(-1, X_encoder_train.shape[2]))
	scalers.update({'X_scaler': X_scaler})
	X_encoder_scaled = X_scaler.transform(X_encoder_train.reshape(-1, X_encoder_train.shape[2])).reshape(X_encoder_train.shape)
	y_scaler = MinMaxScaler(feature_range=(0, 1))
	y_scaler.fit(y_decoder_train.reshape(-1, 1))
	scalers.update({'y_scaler': y_scaler})
	y_decoder_scaled = y_scaler.transform(y_decoder_train.reshape(-1, 1)).reshape(y_decoder_train.shape)

	# Validate dataset sizes
	assert X_encoder_scaled.shape[0] == y_decoder_scaled.shape[0], "Mismatch in dataset sizes between X_encoder_scaled and y_decoder_scaled."

	# Step 1: Perform TimeSeriesSplit
	for fold, (train_index, val_index) in enumerate(tscv.split(X_encoder_scaled), start=1):
		# Validate indices
		if max(train_index) >= X_encoder_scaled.shape[0] or max(val_index) >= X_encoder_scaled.shape[0]:
			raise IndexError(f"Train or validation index is out of bounds in fold {fold}")

		print(f"\n--- Fold {fold} of {n_splits} ---")
		print(f"Fold {fold}: Train size: {len(train_index)}, Validation size: {len(val_index)}")
		print(f"Train indices: {train_index}")
		print(f"Validation indices: {val_index}")

		# Split data into training and validation sets
		X_train, X_val = X_encoder_scaled[train_index], X_encoder_scaled[val_index]
		y_train, y_val = y_decoder_scaled[train_index], y_decoder_scaled[val_index]

		print("Shape of X_train:", X_train.shape)
		print("Shape of X_val:", X_val.shape)
		print("Shape of y_train:", y_train.shape)
		print("Shape of y_val:", y_val.shape)

		# Create a new model for each fold
		model, model_name = model_fn()

		# Define early stopping
		early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

		# Train the model
		history = model.fit(
			X_train,
			y_train,
			validation_data=(X_val, y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=[early_stopping, lr_scheduler],
			verbose=1
		)

		# Evaluate the model on the validation set
		val_predictions = model.predict(X_val)
		print("Shape of val_predictions:", val_predictions.shape)  # Expected: (batch_size, n_steps_out, 1)

		# Reshape or squeeze the predictions and ground truth to remove the last dimension
		val_predictions = val_predictions.squeeze(axis=-1)  # Shape: (batch_size, n_steps_out)
		y_val = y_val.reshape(val_predictions.shape)  # Reshape y_val to (batch_size, n_steps_out)

		# Compute RMSE
		val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
		rmse_scores.append(val_rmse)
		val_losses.append(min(history.history['val_loss']))
		print(f"Fold {fold} RMSE: {val_rmse:.4f}")
		fold += 1

		# Optionally, plot training and validation loss for this fold
		if ax is not None:
			ax.plot(history.history['loss'], label=f'Fold {fold} Training Loss')
			ax.plot(history.history['val_loss'], label=f'Fold {fold} Validation Loss')

	# Calculate average RMSE and validation loss across all folds
	avg_rmse = np.mean(rmse_scores)
	avg_val_loss = np.mean(val_losses)
	print(f"\nAverage RMSE across {n_splits} folds: {avg_rmse:.4f}")
	print(f"Average Validation Loss across {n_splits} folds: {avg_val_loss:.4f}")

	return avg_rmse, avg_val_loss, rmse_scores, val_losses

def train_seq2seq_without_timefolding(model_fn, X_encoder_train, decoder_input_train, y_decoder_train, feature_columns,
									  validation_split=0.2, batch_size=32, epochs=100, patience=10, lr_scheduler=None, teacher_forcing=False):
	"""
	Train the Seq2Seq model without timefolding, using a fixed validation split.

	Parameters:
		model_fn (function): Function to create a new Seq2Seq model.
		X_encoder_train (numpy.ndarray): Encoder input data.
		decoder_input_train (numpy.ndarray): Decoder input data.
		y_decoder_train (numpy.ndarray): Ground truth output data.
		feature_columns (list): List of feature column names.
		validation_split (float): Fraction of data to use for validation.
		batch_size (int): Batch size for training.
		epochs (int): Number of epochs for training.
		patience (int): Early stopping patience.
		lr_scheduler (keras.callbacks.Callback): Learning rate scheduler.
		ax (matplotlib.axes.Axes): Axis for plotting training and validation loss.

	Returns:
		dict: Dictionary containing training results.
	"""

	# Set random seeds for reproducibility
	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

	# Configure TensorFlow for deterministic operations
	tf.config.experimental.enable_op_determinism()

   # Calculate the number of sequences
	num_sequences = min(X_encoder_train_3d.shape[0], decoder_input_train_3d.shape[0], y_decoder_train_3d.shape[0])

	# Calculate the split index based on the number of sequences
	split_index = int((1 - validation_split) * num_sequences)

	# Split the data into training and validation sets
	X_train = X_encoder_train_3d[:split_index]
	X_val = X_encoder_train_3d[split_index:]
	decoder_input_train = decoder_input_train_3d[:split_index]
	decoder_input_val = decoder_input_train_3d[split_index:]
	y_train = y_decoder_train_3d[:split_index]
	y_val = y_decoder_train_3d[split_index:]

	# Debugging: Print the shapes of the split data
	print("Shape of X_train:", X_train.shape)
	print("Shape of X_val:", X_val.shape)
	print("Shape of decoder_input_train:", decoder_input_train.shape)
	print("Shape of decoder_input_val:", decoder_input_val.shape)
	print("Shape of y_train:", y_train.shape)
	print("Shape of y_val:", y_val.shape)

	# Validate data cardinality
	# assert X_train.shape[0] == y_train.shape[0], f"Mismatch in training samples: X_train has {X_train.shape[0]} samples, but y_train has {y_train.shape[0]} samples."
	# assert X_val.shape[0] == y_val.shape[0], f"Mismatch in validation samples: X_val has {X_val.shape[0]} samples, but y_val has {y_val.shape[0]} samples."

	# Debugging before model.fit()
	print("Debugging before model.fit():")
	print("X_train shape:", X_train.shape)
	print("y_train shape:", y_train.shape)
	print("X_val shape:", X_val.shape)
	print("y_val shape:", y_val.shape)
   
	if teacher_forcing:
		print("decoder_input_train shape:", decoder_input_train.shape)
		print("decoder_input_val shape:", decoder_input_val.shape)

	# Define early stopping
	early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

	if teacher_forcing:
		# Train the model
		history = model.fit(
			[X_train, decoder_input_train],  # Use teacher forcing
			y_train,
			validation_data=([X_val, decoder_input_val], y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=[early_stopping, lr_scheduler],
			verbose=1
		)
		# Evaluate the model on the validation set
		val_predictions = model.predict([X_val, decoder_input_val])

	else:
		# Train the model without teacher forcing
		history = model.fit(
			X_train,
			y_train,
			validation_data=(X_val, y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=[early_stopping, lr_scheduler],
			verbose=1
		)
		val_predictions = model.predict(X_val)
		
	print("Shape of val_predictions:", val_predictions.shape)  # Expected: (batch_size, n_steps_out, 1)

	# Reshape or squeeze the predictions and ground truth to remove the last dimension
	val_predictions = val_predictions.squeeze(axis=-1)  # Shape: (batch_size, n_steps_out)
	y_val = y_val.reshape(val_predictions.shape)  # Reshape y_val to (batch_size, n_steps_out)

	# Compute RMSE
	val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
	print(f"Validation RMSE: {val_rmse:.4f}")

	# Compute average validation loss
	avg_val_loss = np.mean(history.history['val_loss'])
	print(f"Average Validation Loss: {avg_val_loss:.4f}")

	last_epoch = len(history.history['loss']) - 1

	return val_rmse, avg_val_loss, last_epoch, history

#TODO remove plotting functions
def train_seq2seq_with_timefolding_teacher_forcing(model_fn, X_encoder_train, decoder_input_train, y_decoder_train, feature_columns,
											 n_splits, batch_size, epochs, patience, lr_scheduler):
	"""
	Train the Seq2Seq model using k-fold cross-validation with teacher forcing.

	Parameters:
		model_fn (function): Function to create a new Seq2Seq model.
		X_encoder_train (numpy.ndarray): Encoder input data.
		decoder_input_train (numpy.ndarray): Decoder input data.
		y_decoder_train (numpy.ndarray): Ground truth output data.
		n_splits (int): Number of folds for k-fold cross-validation.
		batch_size (int): Batch size for training.
		epochs (int): Number of epochs for training.
		patience (int): Early stopping patience.
		ax (matplotlib.axes.Axes): Axis for plotting training and validation loss.

	Returns:
		dict: Dictionary containing average RMSE and validation losses for all folds.
	"""

	# Set random seeds for reproducibility
	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

	# Configure TensorFlow for deterministic operations
	tf.config.experimental.enable_op_determinism()

	fold = 1
	rmse_scores = []
	val_losses = []
	tscv = TimeSeriesSplit(n_splits=n_splits)

	print("Shape of X_encoder_train:", X_encoder_train.shape)
	print("n_steps_in:", X_encoder_train.shape[1])
	print("n_steps_out:", y_decoder_train.shape[1])
	scalers = {}  # Cache the scalers for use across folds

	# Step 1: Scale the data
	# Scale the train and validation sets using the same scaler
	X_scaler = MinMaxScaler(feature_range=(0, 1))
	X_scaler.fit(X_encoder_train.reshape(-1, X_encoder_train.shape[2]))
	scalers.update({'X_scaler': X_scaler})
	X_encoder_scaled = X_scaler.transform(X_encoder_train.reshape(-1, X_encoder_train.shape[2])).reshape(X_encoder_train.shape)
	y_scaler = MinMaxScaler(feature_range=(0, 1))
	y_scaler.fit(y_decoder_train.reshape(-1, 1))
	scalers.update({'y_scaler': y_scaler})
	y_decoder_scaled = y_scaler.transform(y_decoder_train.reshape(-1, 1)).reshape(y_decoder_train.shape)

	# print("decoder_input_train shape:", decoder_input_train.shape)
	print("Validating scalers for X_encoder_scaled")
	validate_scalers_entire_dataset(X_encoder_scaled, X_scaler, feature_columns, feature_axis=2)
	print("Validating scalers for y_decoder_scaled")
	validate_scalers_entire_dataset(y_decoder_scaled, y_scaler, ['Tgt'],  feature_axis=2)

	# Step 1: Perform TimeSeriesSplit
	for fold, (train_index, val_index) in enumerate(tscv.split(X_encoder_scaled), start=1):
		if max(train_index) >= X_encoder_scaled.shape[0] or max(val_index) >= X_encoder_scaled.shape[0]:
			raise IndexError(f"Train or validation index is out of bounds in fold {fold}")
		
		print(f"\n--- Fold {fold} of {n_splits} ---")
		print(f"Fold {fold}: Train size: {len(train_index)}, Validation size: {len(val_index)}")
		print(f"Train indices: {train_index}")
		print(f"Validation indices: {val_index}")        

		# Validate indices
		if max(train_index) >= X_encoder_scaled.shape[0] or max(val_index) >= X_encoder_scaled.shape[0]:
			raise IndexError(f"Train or validation index is out of bounds in fold {fold}")

		# Split data into training and validation sets
		X_train, X_val = X_encoder_scaled[train_index], X_encoder_scaled[val_index]
		# decoder_input_train, decoder_input_val = decoder_input[train_index], decoder_input[val_index]
		y_train, y_val = y_decoder_scaled[train_index], y_decoder_scaled[val_index]

		print("Shape of X_train:", X_train.shape)
		print("Shape of X_val:", X_val.shape)
		# print("Shape of decoder_input_train:", decoder_input_train.shape)
		# print("Shape of decoder_input_val:", decoder_input_val.shape)
		print("Shape of y_train:", y_train.shape)
		print("Shape of y_val:", y_val.shape)

		# Scale the target variable in decoder_input
		# decoder_input[:, :, -1] = target_scaler.fit_transform(decoder_input[:, :, -1].reshape(-1, 1)).reshape(decoder_input.shape[0], decoder_input.shape[1])

		# Initialize scalers for each feature of decoder_input
		# decoder_scalers_dict = {}
		# for i, feature in enumerate(feature_columns):
		#     decoder_scaler = MinMaxScaler(feature_range=(0, 1))()
		#     decoder_scalers_dict[feature] = decoder_scaler
		#     decoder_input[:, :, i] = decoder_scaler.fit_transform(decoder_input[:, :, i])  # Scale the entire dataset

		# # Add a scaler for the target variable of decoder_input to scalers_dict. Each scaler expects multiple timesteps.
		# for t in range(decoder_input.shape[1]):  # Iterate over timesteps
		#     decoder_target_scaler = MinMaxScaler(feature_range=(0, 1))()
		#     decoder_scalers_dict['Tgt'] = decoder_target_scaler
		#     decoder_input[:, t, -1] = decoder_target_scaler.fit_transform(decoder_input[:, t, -1].reshape(-1, 1)).flatten()
		
		# Extract the last timestep for each sample
		X_train_last_timestep = X_train[:, -1, :]  # Shape: (n_samples, n_features)
		X_val_last_timestep = X_val[:, -1, :]  # Shape: (n_samples, n_features)

		# # Step 1: Load data corresponding to the feature columns
		# print("decoder_input_train_df constructed in two steps:")
		# decoder_input_train_features = decoder_input_train.reshape(-1, decoder_input_train.shape[2])
		# decoder_input_train_features_df = pd.DataFrame(decoder_input_train_features, columns=feature_columns)
		# # Step 2: Load data corresponding to the target column
		# decoder_input_train_target = decoder_input_train[:, :, -1].reshape(-1, 1)
		# decoder_input_train_target_df = pd.DataFrame(decoder_input_train_target, columns=['Tgt'])
		# # Combine the feature and target DataFrames
		# decoder_input_train_df = pd.concat([decoder_input_train_features_df, decoder_input_train_target_df], axis=1)
		# # Debug: Print the first few rows of the DataFrame
		# print(np.round(decoder_input_train_df.head(),2))

		# print("decoder_input_val_df constructed in two steps:")
		# decoder_input_val_features = decoder_input_val.reshape(-1, decoder_input_val.shape[2])
		# decoder_input_val_features_df = pd.DataFrame(decoder_input_val_features, columns=feature_columns)
		# # Step 2: Load data corresponding to the target column
		# decoder_input_val_target = decoder_input_val[:, :, -1].reshape(-1, 1)
		# decoder_input_val_target_df = pd.DataFrame(decoder_input_val_target, columns=['Tgt'])
		# # Combine the feature and target DataFrames
		# decoder_input_val_df = pd.concat([decoder_input_val_features_df, decoder_input_val_target_df], axis=1)
		# # Debug: Print the first few rows of the DataFrame
		# print(np.round(decoder_input_val_df.head(),2))

		# y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape[0], y_train.shape[1], 1)
		# y_val = target_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape[0], y_val.shape[1], 1)
  
		# Create DataFrames using the feature_columns list
		X_train_df = pd.DataFrame(X_train_last_timestep, columns=feature_columns)
		X_val_df = pd.DataFrame(X_val_last_timestep, columns=feature_columns)
		y_train_df = pd.DataFrame(y_train.reshape(-1, 1), columns=['Tgt'])
		y_val_df = pd.DataFrame(y_val.reshape(-1, 1), columns=['Tgt'])
		#concatenate the last timestep of X_train and y_train
		print("X_train (last timestep):")
		new_X_train_df = pd.concat([X_train_df, y_train_df], axis=1)
		print(np.round(new_X_train_df.head(),2))
		print("X_val (last timestep):")
		new_X_val_df = pd.concat([X_val_df, y_val_df], axis=1)
		print(np.round(new_X_val_df.head(),2))

		# Create a new model for each fold
		model, model_name = model_fn()

		# Define early stopping
		early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

		# Train the model with no teacher forcing
		history = model.fit(
			# [X_train, decoder_input_train], #with teacher forcing
			X_train,
			y_train,
			# validation_data=([X_val, decoder_input_val], y_val), #with teacher forcing
			validation_data=(X_val, y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=[early_stopping, lr_scheduler],
			verbose=1
		)

		# Evaluate the model on the validation set
		# val_predictions = model.predict([X_val, decoder_input_val]) #with teacher-forcing
		val_predictions = model.predict(X_val) #with  no teacher-forcing
		print("Shape of val_predictions:", val_predictions.shape)  # Expected: (batch_size, n_steps_out, 1)
		# Reshape or squeeze the predictions and ground truth to remove the last dimension
		val_predictions = val_predictions.squeeze(axis=-1)  # Shape: (batch_size, n_steps_out)
		# y_val = y_val[:, 0]  # Select only the first timestep from y_val to match val_predictions
		# Ensure y_val has the same shape as val_predictions
		y_val = y_val.reshape(val_predictions.shape)  # Reshape y_val to (batch_size, n_steps_out, 1)
		# Compute RMSE
		val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
		rmse_scores.append(val_rmse)
		val_losses.append(min(history.history['val_loss']))
		print(f"Fold {fold} RMSE: {val_rmse:.4f}")
		fold += 1

		# Optionally, plot training and validation loss for this fold
		if ax is not None:
			ax.plot(history.history['loss'], label=f'Fold {fold} Training Loss')
			ax.plot(history.history['val_loss'], label=f'Fold {fold} Validation Loss')

	# Calculate average RMSE and validation loss across all folds
	avg_rmse = np.mean(rmse_scores)
	avg_val_loss = np.mean(val_losses)
	print(f"\nAverage RMSE across {n_splits} folds: {avg_rmse:.4f}")
	print(f"Average Validation Loss across {n_splits} folds: {avg_val_loss:.4f}")

	# # Return results
	# return {
	#     "avg_rmse": avg_rmse,
	#     "avg_val_loss": avg_val_loss,
	#     "rmse_scores": rmse_scores,
	#     "val_losses": val_losses
	# }

	#return last_epoch
	last_epoch = len(history.history['loss'])    

	return avg_rmse, avg_val_loss, rmse_scores, val_losses, last_epoch, scalers#, decoder_scalers_dict
	
def evaluate_seq2seq_model_on_test_data(model, target_scaler, X_encoder_test_3d, decoder_input_test_3d, 
		X_encoder_test_df, y_decoder_test_df, n_steps_out,feature_columns, use_teacher_forcing=False):
	"""
	Forecast multiple steps ahead using the Seq2Seq model.

	Parameters:
		model (keras.Model): Trained Seq2Seq model.
		X_encoder_test (numpy.ndarray): Encoder input data for testing.
		decoder_input_test (numpy.ndarray): Decoder input data for testing.
		y_decoder_test (numpy.ndarray): True output data for testing.
		n_steps_out (int): Number of output time steps to forecast.
		disable_scaling (bool): If True, disable scaling of inputs.

	Returns:
		predictions (numpy.ndarray): Forecasted values.
	"""

	print("Shape of X_encoder_test_df:", X_encoder_test_df.shape)
	print("Shape of X_encoder_test_3d before scaling:", X_encoder_test_3d.shape)
	print("Some rows of X_encoder_test_df before scaling:")
	print(np.round(X_encoder_test_df.head(),2))
	# X_encoder_test_3d is already scaled
	# Reshape the encoder input data
	X_encoder_test_reshaped = X_encoder_test_3d.reshape(
		X_encoder_test_3d.shape[0], X_encoder_test_3d.shape[1], X_encoder_test_3d.shape[2]
	)
	# Make predictions with teacher_forcing
	# predictions = model.predict([X_encoder_test_reshaped, decoder_input_test])
	# Make predictions without teacher_forcing
	# Ensure the input tensor matches the expected structure
	#normal seq2seq model requires "encoder_input" as input but attention model requires "encoder_inputs"

	attention_models = {
		"seq2seq_model_with_attention",
		"bidirectional_seq2seq_model_with_attention",
		"bidirectional_seq2seq_model_with_attention_no_teacher_forcing",
		"seq2seq_model_with_attention_and_teacher_forcing"
	}

	# inputs = {"encoder_inputs" if model_name in attention_models else "encoder_input": X_encoder_test_reshaped}
	# if use_teacher_forcing:
	# 	inputs["decoder_input"] = decoder_input_test_3d

	# Get model input names
	input_names = [inp.name.split(":")[0] for inp in model.inputs]
	print("Model input names:", input_names)

	# Determine encoder input key
	encoder_key = "encoder_inputs" if any("encoder_inputs" in name for name in input_names) else "encoder_input"

	inputs = {encoder_key: X_encoder_test_reshaped}

	# Add decoder input if teacher forcing is used
	if use_teacher_forcing:
		if any("decoder_inputs" in name for name in input_names):
			decoder_key = "decoder_inputs"
		elif any("decoder_input" in name for name in input_names):
			decoder_key = "decoder_input"
		else:
			raise ValueError(f"Decoder input not found in model inputs: {input_names}")
		inputs[decoder_key] = decoder_input_test_3d

	predictions_scaled = model.predict(inputs)
	print("Shape of predictions_scaled:", predictions_scaled.shape)
	#print a few rows of predictions_scaled
	print("Predictions (scaled):")
	print(np.round(predictions_scaled[:5],2))

	#unscale the predictions. Note that the predictions are 3D and may contain more than one prediction for each timestep,
	#based on n_steps_out
	predictions_unscaled = np.zeros(predictions_scaled.shape)
	
	for i in range(predictions_scaled.shape[1]):
		predictions_unscaled[:, i] = target_scaler.inverse_transform(predictions_scaled[:, i].reshape(-1, 1)).reshape(predictions_scaled.shape[0], predictions_scaled.shape[2])
	
	# Debugging: Check the shape of predictions_unscaled
	print("Shape of predictions_unscaled:", predictions_unscaled.shape)

	# Reshape predictions_scaled to 2D: (samples, timesteps)
	predictions_unscaled_reshaped = predictions_unscaled.reshape(predictions_unscaled.shape[0], predictions_unscaled.shape[1])
	
	# Debugging: Check the shape of predictions_unscaled_reshaped
	print("Shape of predictions_unscaled_reshaped:", predictions_unscaled_reshaped.shape)

	# Create predictions_df and align the index with X_encoder_test_df
	predictions_df = pd.DataFrame(
		predictions_unscaled_reshaped,
		columns=[f'Trend_Predicted_t{i+1}' for i in range(predictions_unscaled.shape[1])],
		index=X_encoder_test_df.index  # Align the index
	)
	#add predictions to X_encoder_test_df as a new set of columns (depending on n_steps_out)
	#concatenate predictions_df with X_encoder_test_df
	predictions_with_X_test_df = pd.concat([X_encoder_test_df, predictions_df], axis=1)
	#print a few rows of predictions_df
	print("Predictions dataFrame before unscaling:")
	print(np.round(predictions_with_X_test_df.head(),2))
	print("Shape of predictions_with_X_test_df:", predictions_with_X_test_df.shape)

	return predictions_with_X_test_df

def plot_predictions(model_name, X_encoder_test_df_unscaled, y_decoder_test_df_unscaled, predictions_with_X_test_df, ax):
	print("Shape of predictions_df dataframe:", predictions_with_X_test_df.shape)
	print("Shape of X_encoder_test_df dataframe:", X_encoder_test_df_unscaled.shape)

	# Check if predictions_df is empty or contains NaN values
	if predictions_with_X_test_df.empty or predictions_with_X_test_df.isnull().values.all():
		print("No predictions available for plotting.")
		return

	# Check if test_df is empty or contains NaN values
	if X_encoder_test_df_unscaled.empty or X_encoder_test_df_unscaled.isnull().values.all():
		print("No test data available for plotting.")
		return

	# Ensure the index is set to 'Date' if available for X_encoder_test_df_unscaled
	if 'Date' in X_encoder_test_df_unscaled.index.names:
		print("Date column is already set as index in X_encoder_test_df_unscaled.")
	elif 'Date' in X_encoder_test_df_unscaled.columns:
		print("Setting 'Date' column as index for X_encoder_test_df_unscaled.")
		X_encoder_test_df_unscaled.set_index('Date', inplace=True)
	else:
		print("Warning: 'Date' column not found in X_encoder_test_df_unscaled. Using default index.")
	
	# Ensure the index is set to 'Date' if available for y_decoder_test_df_unscaled
	if 'Date' in y_decoder_test_df_unscaled.index.names:
		print("Date column is already set as index in y_decoder_test_df_unscaled.")
	elif 'Date' in y_decoder_test_df_unscaled.columns:
		print("Setting 'Date' column as index for y_decoder_test_df_unscaled.")
		y_decoder_test_df_unscaled.set_index('Date', inplace=True)
	else:
		print("Warning: 'Date' column not found in y_decoder_test_df_unscaled. Using default index.")
	# Ensure the indices of X_encoder_test_df_unscaled and y_decoder_test_df_unscaled are aligned
	y_decoder_test_df_unscaled.index = X_encoder_test_df_unscaled.index  # Align the index of y_decoder_test_df_unscaled with X_encoder_test_df_unscaled

	# Join X_encoder_test_df_unscaled and y_decoder_test_df_unscaled for plotting
	test_df = pd.concat([X_encoder_test_df_unscaled, y_decoder_test_df_unscaled], axis=1)
	print("Combined DataFrame for plotting:")
	print(test_df.head())

	# Ensure the index is set to 'Date' if available
	if 'Date' in test_df.columns:
		test_df.set_index('Date', inplace=True)
	else:
		print("Warning: 'Date' column not found in test_df. Using default index.")

	# Get the last 10 rows for plotting
	test_last_10 = test_df.iloc[-10:]
	predictions_last_10 = predictions_with_X_test_df.iloc[-10:]

	# Debugging: Check the index of test_last_10
	print("Index of test_last_10:", test_last_10.index)

	# Check if test_last_10 is empty or contains NaN values
	if test_last_10.empty or test_last_10.isnull().values.all():
		print("No test data available for plotting.")
		input("Press Enter to continue...")
		return

	# Check if predictions_last_10 is empty or contains NaN values
	if predictions_last_10.empty or predictions_last_10.isnull().values.all():
		print("No predictions available for plotting.")
		input("Press Enter to continue...")
		return

	# Dynamically determine the target column name(s) based on n_steps_out
	if n_steps_out == 1:
		target_column = 'Tgt'
	else:
		target_column = 'Tgt(t)'  # Use the first target column for plotting

	# Plot the actual trend
	ax.plot(test_last_10.index.to_numpy(), test_last_10[target_column].to_numpy(), label='Actual Trend (Last 10 Days)', marker='o')

	# Plot the predicted trends
	for col in predictions_last_10.columns:
		if col.startswith('Trend_Predicted_t'):
			# Align indices of predictions_last_10 with test_last_10 before plotting
			predictions_last_10_aligned = predictions_last_10[col].reindex(test_last_10.index)
			ax.plot(test_last_10.index.to_numpy(), predictions_last_10_aligned.to_numpy(), label=f'Predicted {col} (Last 10 Days)', linestyle='--', marker='o')

	# Format the x-axis to display dates
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	ax.xaxis.set_major_locator(mdates.DayLocator())
	plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

	# Annotate the actual trend values on the graph
	for i, value in enumerate(test_last_10[target_column]):
		ax.text(test_last_10.index[i], value, f'{value:.2f}', fontsize=8, color='blue', ha='center', va='bottom')
	#Annotate the predicted trend values on the graph. There can be more than one predicted trend
	for col in predictions_last_10.columns:
		if col.startswith('Trend_Predicted_t'):
			for i, value in enumerate(predictions_last_10[col]):
				ax.text(test_last_10.index[i], value, f'{value:.2f}', fontsize=8, color='orange', ha='center', va='bottom')
	# Add a notes box with MAE, MSE, and RMSE metrics
	metrics_text = '\n'.join([
		f'{metric} ({col}): {value:.4f}'
		for col in predictions_last_10.columns if col.startswith('Trend_Predicted_t')
		for metric, value in [
			('MAE', mean_absolute_error(test_last_10[target_column], predictions_last_10[col])),
			('MSE', mean_squared_error(test_last_10[target_column], predictions_last_10[col])),
			('RMSE', np.sqrt(mean_squared_error(test_last_10[target_column], predictions_last_10[col])))
		]
	])
	textstr = f'Model Name: {model_name}\n{metrics_text}'

	# Add a box with the text
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(0.01, 0.01, textstr, transform=ax.transAxes, fontsize=10,
			verticalalignment='bottom', horizontalalignment='left', bbox=props)

	ax.set_title('Financial Time Series Trend Forecasting with LSTM Seq2Seq Model')
	ax.set_xlabel('Date')
	ax.set_ylabel('Adj Close Price')
	ax.legend()

def predict_without_teacher_forcing_v0(model, current_input, n_steps_in, n_steps_out, n_features):
    """
    Predict without teacher forcing by using the encoder's states and feeding back predictions.

    Parameters:
        model (keras.Model): Trained Seq2Seq model.
        current_input (numpy.ndarray): Encoder input data for testing (shape: [1, n_steps_in, n_features]).
        n_steps_in (int): Number of input timesteps.
        n_steps_out (int): Number of output timesteps.
        n_features (int): Number of features in the input data.

    Returns:
        numpy.ndarray: Forecasted values for the next n_steps_out timesteps.
    """
    from keras.models import Model
    from keras.layers import Input

    # Ensure input shape is correct
    assert current_input.shape == (1, n_steps_in, n_features), \
        f"Expected shape (1, n_steps_in, {n_features}), but got {current_input.shape}"

    # Get encoder states
    encoder_lstm_layer = model.get_layer("encoder_lstm")
    encoder_model = Model(inputs=model.input[0], outputs=encoder_lstm_layer.output[1:])
    state_h, state_c = encoder_model.predict(current_input)

    # Try to get the correct decoder LSTM layer (handle both possible names)
    try:
        decoder_lstm = model.get_layer("decoder_lstm")
    except ValueError:
        decoder_lstm = model.get_layer("decoder_lstm_no_teacher_forcing")
    decoder_dense = model.get_layer("decoder_dense")

    # Check units
    if decoder_lstm.units is None:
        raise ValueError("decoder_lstm.units is None. Check the layer name and model structure.")

    # Build the inference decoder model for one step
    decoder_input_layer = Input(shape=(1, n_features))
    decoder_state_input_h = Input(shape=(decoder_lstm.units,))
    decoder_state_input_c = Input(shape=(decoder_lstm.units,))
    decoder_outputs, state_h_out, state_c_out = decoder_lstm(
        decoder_input_layer, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_input_layer, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs, state_h_out, state_c_out]
    )

    # Prepare the decoder input (start with last known features, except target)
    decoder_input = np.zeros((1, 1, n_features))
    decoder_input[0, 0, :-1] = current_input[0, -1, :-1]

    predictions = []
    h, c = state_h, state_c
    for t in range(n_steps_out):
        output, h, c = decoder_model.predict([decoder_input, h, c])
        predictions.append(output[0, 0, 0])
        # Update decoder_input for next timestep (autoregressive)
        decoder_input[0, 0, -1] = output[0, 0, 0]  # Only update the target feature

    return np.array(predictions)

def predict_without_teacher_forcing(model, current_input, n_steps_in, n_steps_out, n_features):
    """
    Predict without teacher forcing for both vanilla and attention-based seq2seq models.

    Parameters:
        model (keras.Model): Trained Seq2Seq model.
        current_input (numpy.ndarray): Encoder input data for testing (shape: [1, n_steps_in, n_features]).
        n_steps_in (int): Number of input timesteps.
        n_steps_out (int): Number of output timesteps.
        n_features (int): Number of features in the input data.

    Returns:
        numpy.ndarray: Forecasted values for the next n_steps_out timesteps.
    """
    from keras.models import Model
    from keras.layers import Input, Concatenate, Dot, Activation, TimeDistributed, Dense

    # Ensure input shape is correct
    assert current_input.shape == (1, n_steps_in, n_features), \
        f"Expected shape (1, n_steps_in, {n_features}), but got {current_input.shape}"

    # Detect if model uses attention by checking input names
    input_names = [inp.name.split(":")[0] for inp in model.inputs]
    uses_attention = any("encoder_outputs" in name or "encoder_inputs" in name for name in input_names)

    # Get encoder states and outputs
    encoder_lstm_layer = model.get_layer("encoder_lstm")
    if uses_attention:
        # Attention model: encoder outputs sequences
        encoder_model = Model(inputs=model.input[0], outputs=[encoder_lstm_layer.output[0], encoder_lstm_layer.output[1], encoder_lstm_layer.output[2]])
        encoder_outputs, state_h, state_c = encoder_model.predict(current_input)
    else:
        # Vanilla model: encoder outputs last state only
        encoder_model = Model(inputs=model.input[0], outputs=encoder_lstm_layer.output[1:])
        state_h, state_c = encoder_model.predict(current_input)
        encoder_outputs = None

    # Try to get the correct decoder LSTM layer (handle both possible names)
    try:
        decoder_lstm = model.get_layer("decoder_lstm")
    except ValueError:
        decoder_lstm = model.get_layer("decoder_lstm_no_teacher_forcing")

    # Get the Dense/TimeDistributed layer
    try:
        decoder_dense = model.get_layer("decoder_dense")
    except ValueError:
        # For TimeDistributed(Dense(...), name="decoder_dense")
        decoder_dense = [l for l in model.layers if l.name == "decoder_dense"][0]

    # Build the inference decoder model
    decoder_input_layer = Input(shape=(1, n_features), name="decoder_input_infer")
    decoder_state_input_h = Input(shape=(decoder_lstm.units,), name="decoder_h_infer")
    decoder_state_input_c = Input(shape=(decoder_lstm.units,), name="decoder_c_infer")

    if uses_attention:
        # Attention inference: need encoder_outputs as input
        encoder_outputs_input = Input(shape=(n_steps_in, decoder_lstm.units), name="encoder_outputs_infer")
        # Decoder LSTM
        decoder_outputs, state_h_out, state_c_out = decoder_lstm(
            decoder_input_layer, initial_state=[decoder_state_input_h, decoder_state_input_c]
        )
        # Attention mechanism
        attention_scores = Dot(axes=[2, 2], name="attention_scores_infer")([decoder_outputs, encoder_outputs_input])
        attention_weights = Activation('softmax', name="attention_weights_infer")(attention_scores)
        context_vector = Dot(axes=[2, 1], name="context_vector_infer")([attention_weights, encoder_outputs_input])
        decoder_combined_context = Concatenate(name="decoder_combined_context_infer")([decoder_outputs, context_vector])
        # Final output
        decoder_outputs_final = decoder_dense(decoder_combined_context)
        decoder_model = Model(
            [decoder_input_layer, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
            [decoder_outputs_final, state_h_out, state_c_out]
        )
    else:
        # Vanilla inference
        decoder_outputs, state_h_out, state_c_out = decoder_lstm(
            decoder_input_layer, initial_state=[decoder_state_input_h, decoder_state_input_c]
        )
        decoder_outputs_final = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_input_layer, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs_final, state_h_out, state_c_out]
        )

    # Prepare the decoder input (start with last known features, except target)
    decoder_input = np.zeros((1, 1, n_features))
    decoder_input[0, 0, :-1] = current_input[0, -1, :-1]

    predictions = []
    h, c = state_h, state_c
    for t in range(n_steps_out):
        if uses_attention:
            output, h, c = decoder_model.predict([decoder_input, h, c, encoder_outputs])
        else:
            output, h, c = decoder_model.predict([decoder_input, h, c])
        predictions.append(output[0, 0, 0])
        # Update decoder_input for next timestep (autoregressive)
        decoder_input[0, 0, -1] = output[0, 0, 0]  # Only update the target feature

    return np.array(predictions)
def forecast_next_n_days(model, target_scaler, X_encoder_test_3d, y_encoder_test_3d, n_steps_in, n_steps_out, n_features, n_days, ax):
	"""
	Forecast the next n_days using a trained Seq2Seq model without using ground truth.

	Parameters:
		model (keras.Model): Trained Seq2Seq model.
		target_scaler (scaler): Scaler used for the target variable.
		X_encoder_test_3d (numpy.ndarray): Encoder input data for testing (3D array).
		n_steps_in (int): Number of input timesteps.
		n_steps_out (int): Number of output timesteps.
		n_features (int): Number of features in the input data.
		n_days (int): Number of business days to forecast.

	Returns:
		predictions (list): Forecasted values for the next n_days.
	"""
	# Start with the last n_steps_in timesteps from the test data
	#Get the last row of data from X_encoder_test_3d, which contains the last n_steps_in timesteps
	current_input = X_encoder_test_3d[-1, -n_steps_in:, :]  # Shape: (n_steps_in, n_features)
	# current_input = X_encoder_test_3d[-1]  # Shape: (n_steps_in, n_features)
	current_input = current_input.reshape(1, n_steps_in, n_features)  # Add batch dimension
	print("Shape of current_input:", current_input.shape)  # Expected: (1, n_steps_in, n_features)

	# Define the attention models set outside the loop
	attention_models = {
		"seq2seq_model_with_attention",
		"bidirectional_seq2seq_model_with_attention",
		"bidirectional_seq2seq_model_with_attention_no_teacher_forcing"
	}

	# Determine the input key based on the model name
	input_key = "encoder_inputs" if model.name in attention_models else "encoder_input"

	# Generate a range of business days for the next n_days
	last_date = pd.Timestamp.now()  # Replace with the last date in your dataset if available
	business_days = pd.bdate_range(start=last_date, periods=n_days)
	#enumerate all daates including weekends and holidays
	days = pd.date_range(start=last_date, periods=n_days, freq='D')
	predictions = []
	current_input_df = pd.DataFrame()
	step=0
	for day in days:
		if step == 0:
			previous_adj_close = y_encoder_test_3d[-1, -1, -1]  # Last known value
		else:
			previous_adj_close = predicted_scaled[0]  # Use the predicted value from the previous iteration

		#capture current_input and step into a dataframe for debugging. Append to the dataframe
		current_input_unscaled = feature_scaler.inverse_transform(current_input.reshape(-1, n_features)).reshape(current_input.shape)		
		# Select only the last timestep of current_input_unscaled
		last_timestep_data = current_input_unscaled[0, -1, :]  # Shape: (n_features,)
		# Create a DataFrame with a single row for the last timestep
		current_input_df_inc = pd.DataFrame([last_timestep_data], columns=column_names)
		current_input_df_inc['Step'] = step
		current_input_df_inc['Date'] = day
		previous_adj_close_unscaled = target_scaler.inverse_transform(np.array([[previous_adj_close]])).reshape(-1, 1).item()
		current_input_df_inc['Previous Adj Close'] = previous_adj_close_unscaled

		print(f"Forecasting for business day: {day}, step: {step}")	
		print("Shape of current_input:", current_input.shape)  # Expected: (1, n_steps_in, n_features)
		print("Input scaled:", current_input[:, n_steps_in-1, :])

		# Call the predict_without_teacher_forcing function for autoregressive forecasting
		predicted_scaled = predict_without_teacher_forcing(model, current_input, n_steps_in, n_steps_out, n_features)
		print("Shape of predicted_scaled:", predicted_scaled.shape)  # Expected: (n_steps_out, n_features)
		print(predicted_scaled)
		# predicted_scaled = predicted_scaled.squeeze(axis=-1)  # Remove the last dimension

		# Unscale the predictions
		predicted_unscaled = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

		# Append the first prediction (next day's value) to the results
		predictions.append(predicted_unscaled[0])

		# Update the input for the next iteration
		# Slide the window forward by 1 timestep and append the predicted value
		# next_input = current_input[:, 1:, :]  # Remove the first timestep
		# predicted_features = np.zeros((1, 1, n_features))  # Placeholder for the predicted value
		# # Use only the first predicted value (for the next timestep)
		# predicted_features[0, 0, -1] = predicted_scaled[0] if n_steps_out == 1 else predicted_scaled[0]
		# next_input = np.append(next_input, predicted_features, axis=1)  # Append the predicted value
		
		#take the first predicted value out of the n_steps_out = 2 and update the features
		#pass in unscaled values for calculations and scale them back afterwards
		predicted_value_unscaled = predictions[step]  
		current_input_unscaled = feature_scaler.inverse_transform(current_input.reshape(-1, n_features)).reshape(current_input.shape)
		previous_adj_close_unscaled = target_scaler.inverse_transform(np.array([[previous_adj_close]])).reshape(-1, 1).item()
		X_encoder_test_3d_unscaled = feature_scaler.inverse_transform(X_encoder_test_3d.reshape(-1, n_features)).reshape(X_encoder_test_3d.shape)
		y_decoder_test_3d_unscaled = target_scaler.inverse_transform(y_decoder_test_3d.reshape(-1, 1)).reshape(y_decoder_test_3d.shape)
		next_input = update_features_after_prediction_seq2seq(predicted_value_unscaled, current_input_unscaled, n_steps_in, previous_adj_close_unscaled, step, X_encoder_test_3d_unscaled, y_decoder_test_3d_unscaled, predictions, column_names)
		print("Shape of next_input:", next_input.shape)  # Expected: (1, n_steps_in, n_features)
		current_input = feature_scaler.transform(next_input.reshape(-1, n_features)).reshape(next_input.shape)  # Scale the new input
		#only last time-step is used for prediction, all other timesteps are set to 0
		# # Iterate over all time-steps
		# for t in range(next_input.shape[1]):  # Loop over the time-steps dimension
		# 	time_step_data = next_input[0, t, :]  # Shape: (n_features,)
		# 	print(f"Data for time-step {t}:", time_step_data)
		# current_input = feature_scaler.transform(next_input.reshape(-1, n_features)).reshape(next_input.shape)  # Scale the new input
		current_input = current_input.reshape(1, n_steps_in, n_features)  # Add batch dimension

		step += 1
		#add predicted value to the current_input_df for the next iteration
		current_input_df_inc_cp = current_input_df_inc.copy()  
		if step>0:
			current_input_df_inc['Predicted'] = predicted_unscaled[0]

		current_input_df = pd.concat([current_input_df, current_input_df_inc], ignore_index=True)

	# Plot the predictions after the loop
	if len(predictions) == len(days):
		ax.plot(days.to_numpy(), predictions, marker='o', label='Predicted')
		# Add annotations for each predicted value
		for i, value in enumerate(predictions):
			ax.text(days[i], value, f'{value:.2f}', fontsize=8, color='blue', ha='center', va='bottom')

		ax.set_title('Forecast for the Next Business Days')
		ax.set_xlabel('Date')
		ax.set_ylabel('Predicted Value')
		ax.legend()
		ax.grid()
	else:
		print(f"Error: Mismatch between business_days ({len(days)}) and predictions ({len(predictions)})")

	#set Date as index for current_input_df
	current_input_df['Date'] = pd.to_datetime(current_input_df['Date'])
	#Sort the columns to have the order Step, Date, Predicted, Previous Adj Close, followed by the other columns
	current_input_df = current_input_df[['Step', 'Date', 'Previous Adj Close','Predicted'] + [col for col in current_input_df.columns if col not in ['Step', 'Date', 'Predicted', 'Previous Adj Close']]]
	# current_input_df.set_index('Date', inplace=True)
	#save the current_input_df to a csv file
	#convert all values to 2 decimal places
	current_input_df = current_input_df.round(2)
	current_input_df.to_csv(f'{output_files_folder}/seq2seq_current_input_df.csv', index=False)

	#save x_encoder_test_3d_unscaled to a csv file
	X_encoder_test_3d_unscaled_df = pd.DataFrame(X_encoder_test_3d_unscaled.reshape(-1, n_features), columns=column_names)
	X_encoder_test_3d_unscaled_df = X_encoder_test_3d_unscaled_df.round(2)
	X_encoder_test_3d_unscaled_df.to_csv(f'{output_files_folder}/seq2seq_X_encoder_test_3d_unscaled_df.csv', index=False)

	#save y_decoder_test_3d_unscaled to a csv file
	y_decoder_test_3d_unscaled_df = pd.DataFrame(y_decoder_test_3d_unscaled.reshape(-1, 1), columns=['Tgt'])
	y_decoder_test_3d_unscaled_df = y_decoder_test_3d_unscaled_df.round(2)
	y_decoder_test_3d_unscaled_df.to_csv(f'{output_files_folder}/seq2seq_y_decoder_test_3d_unscaled_df.csv', index=False)
	return predictions

def save_model_and_results(training_params_df, plt, model, output_files_folder, file_prefix):

	#do not look for yesterday's models.
	look_for_yesterday=False
	version,_ = get_max_version(output_files_folder,file_prefix,look_for_yesterday)
	version += 1 #increment the version

	from datetime import datetime
	current_date = datetime.now().strftime('%Y-%m-%d')
	model_path = f'{output_files_folder}/seq2seq_model_{current_date}_v{version}.keras'
	training_params_path = f'{output_files_folder}/seq2seq_model_{current_date}_training_params_v{version}.csv'

	#save the training parameters to a csv file
	training_params_df.to_csv(training_params_path)

	print("Saving model to:", model_path)
	model.export(model_path)
	print("Model saved to disk as: ", model_path)

	plot_path = f'{output_files_folder}/{file_prefix}_plot_training_{current_date}_v{version}.pdf'
	plt.savefig(f'{plot_path}')

if __name__ == '__main__':

	# Set display options to control wrapping
	pd.set_option('display.max_columns', None)  # Do not truncate the list of columns
	pd.set_option('display.max_rows', None)     # Do not truncate the list of rows
	pd.set_option('display.width', 1000)        # Set the display width to a large value
	pd.set_option('display.max_colwidth', 40)   # Set the maximum column width
	pd.set_option('display.colheader_justify', 'left')  # Justify column headers to the left
	pd.set_option('display.float_format', '{:.2f}'.format)  # Set float format
	file_prefix = 'seq2seq_model'

	#create a plot with multiple sections
	fig = plt.figure(figsize=(14, 12))  # width, height
	ax = [
		plt.subplot2grid((2, 2), (0, 0), colspan=2),  # Row 1, spans both columns
		plt.subplot2grid((2, 2), (1, 0)),  # Row 2, Column 1
		plt.subplot2grid((2, 2), (1, 1))   # Row 2, Column 2
	]

	#--Configuration Parameters -- START
	config_params = load_configuration_params()
	output_files_folder = config_params['output_files_folder']
	start_date = config_params['start_date']
	end_date = config_params['end_date']
	increment = config_params['increment']
	split_date = config_params['split_date']
	timeframes_df = get_timeframes_df(start_date, end_date, increment)
	load_from_files=False
	n_steps_in = 10  # Number of input time steps
	n_steps_out = 1  # Number of output time steps
	#--Configuration Parameters -- END

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
	# Calculate rolling averages
	stock_data['Tgt'] = stock_data['Adj Close'].shift(-1) # next day's price 

	# Add engineered features
	stock_data,feature_columns = add_engineered_features(stock_data)
	stock_data.dropna(inplace=True)
	#check if there are any NaN values in the DataFrame
	if stock_data.isnull().sum().sum() > 0:
		count_of_nan_values = stock_data.isnull().sum().sum()
		print(f"{count_of_nan_values} NaN values found in the Stock Data dataFrame.")
	else:
		print("No NaN values found in the Stock Data dataFrame.")
	print("Feature columns after adding engineered features: ", feature_columns)
	#stock_data contains extra columns like "High", "Low", etc.
	print("Stock data columns after adding engineered features: ", stock_data.columns) 
	print("Shape of stock data after adding engineered features: ", stock_data.shape)
	# Remove unnecessary columns. 'Adj Close'/'Close' is also removed as it has been replaced by the 'Tgt' column
	unnecessary_columns = ['Ticker', 'Volume', 'Close', 'High', 'Low', 'Open', 'Adj Close']
	stock_data.drop(columns=unnecessary_columns, inplace=True, errors='ignore')
	#check if target column is present
	target_column = 'Tgt'
	if target_column not in stock_data.columns:
		print("Target column 'Tgt' not found in the stock data DataFrame.")
		raise ValueError("Target column 'Tgt' not found in the stock data DataFrame.")

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

	print("train_prepared shape: ", train_prepared_df.shape)
	print("Index names of train_prepared: ", train_prepared_df.index)
	print("test_prepared shape: ", test_prepared_df.shape)
	print("Index names of test_prepared: ", test_prepared_df.index)

	train_prepared_array = train_prepared_df.values
	train_with_timesteps = series_to_supervised(train_prepared_array, data_frame_columns=feature_columns+['Tgt'], 
											 target_column=target_column, n_in=n_steps_in, n_out=n_steps_out, 
											 dropnan=True, index=train_prepared_df.index)
	print("Shape of train_with_timesteps: ", train_with_timesteps.shape)
	print("Index names of train_with_timesteps: ", train_with_timesteps.index)
	#print the names of the columns in train_with_timesteps with 'Tgt' in them
	print("Columns with 'Tgt' in them: ", [col for col in train_with_timesteps.columns if 'Tgt' in col])

	print("Columns in train_with_timesteps: ", train_with_timesteps.columns)
	#create X_encoder_train and decoder_input_train and y_decoder_train by taking out columns with 'Tgt' in their names
	X_encoder_train_df = train_with_timesteps[[col for col in train_with_timesteps.columns if 'Tgt' not in col]]
	print("Shape of X_encoder_train_df: ", X_encoder_train_df.shape)
	
	y_decoder_train_df = train_with_timesteps[[col for col in train_with_timesteps.columns if 'Tgt' in col]]
	print("Shape of y_decoder_train_df: ", y_decoder_train_df.shape)

	n_features = len(feature_columns)  # Define n_features based on the length of feature_columns
	# Use the last timestep of X_encoder_train_df as the starting point for the decoder input
	last_timestep_features = X_encoder_train_df.iloc[:, -n_features:]  # Last timestep features
	# Repeat the last timestep features for n_steps_out timesteps
	decoder_input_train_df = pd.concat([last_timestep_features] * n_steps_out, axis=1)  # Shape: (4133, 54)
	print("Shape of decoder_input_train_df:", decoder_input_train_df.shape)

	test_prepared_array = test_prepared_df.values
	test_with_timesteps = series_to_supervised(test_prepared_array, data_frame_columns=feature_columns+['Tgt'], 
											target_column=target_column, n_in=n_steps_in, n_out=n_steps_out, 
											dropnan=True, index=test_prepared_df.index)
	print("Shape of test_with_timesteps: ", test_with_timesteps.shape)
	print("Index names of test_with_timesteps: ", test_with_timesteps.index)
	X_encoder_test_df = test_with_timesteps[[col for col in test_with_timesteps.columns if 'Tgt' not in col]]
	print("Shape of X_encoder_test_df after adding timesteps: ", X_encoder_test_df.shape)
	print("Columns in X_encoder_test_df: ", X_encoder_test_df.columns)
	print("A few rows of X_encoder_test_df: ", X_encoder_test_df.head())
	y_decoder_test_df = test_with_timesteps[[col for col in test_with_timesteps.columns if 'Tgt' in col]]
	print("Shape of y_decoder_test_df: ", y_decoder_test_df.shape)

	# Use the last timestep of X_encoder_test_df as the starting point for the decoder input
	last_timestep_features = X_encoder_test_df.iloc[:, -n_features:]  # Last timestep features
	# Repeat the last timestep features for n_steps_out timesteps
	decoder_input_test_df = pd.concat([last_timestep_features] * n_steps_out, axis=1)  # Shape: (4133, 54)
	print("Shape of decoder_input_test_df:", decoder_input_test_df.shape)

	# reshape X_encoder_train to be 3D [samples, timesteps, features]
	X_encoder_train_3d = X_encoder_train_df.values.reshape((X_encoder_train_df.shape[0], n_steps_in, n_features))
	# decoder_input_train_3d = decoder_input_train.values.reshape((decoder_input_train.shape[0], n_steps_out, n_features))
	y_decoder_train_3d = y_decoder_train_df.values.reshape((y_decoder_train_df.shape[0], n_steps_out, 1))
	
	# reshape X_encoder_test to be 3D [samples, timesteps, features]	
	X_encoder_test_3d = X_encoder_test_df.values.reshape((X_encoder_test_df.shape[0], n_steps_in, n_features))
	# decoder_input_test_3d = decoder_input_test.values.reshape((decoder_input_test.shape[0], n_steps_out, n_features))
	y_decoder_test_3d = y_decoder_test_df.values.reshape((y_decoder_test_df.shape[0], n_steps_out, 1))
	
	decoder_input_train_3d = decoder_input_train_df.values.reshape((decoder_input_train_df.shape[0], n_steps_out, n_features))	
	decoder_input_test_3d = decoder_input_test_df.values.reshape((decoder_input_test_df.shape[0], n_steps_out, n_features))

	print("Shape of X_encoder_train_3d:", X_encoder_train_3d.shape)
	print("Shape of y_decoder_train_3d:", y_decoder_train_3d.shape)
	print("Shape of decoder_input_train_3d:", decoder_input_train_3d.shape)
	print("Shape of X_encoder_test_3d:", X_encoder_test_3d.shape)
	print("Shape of y_decoder_test_3d:", y_decoder_test_3d.shape)
	print("Shape of decoder_input_test_3d:", decoder_input_test_3d.shape)	
	#TODO: Need to create a dataframe version of validate_data_processing
	# validate_data_processing(X_encoder_train_df, X_encoder_test_df, y_decoder_train_df, y_decoder_test_df, feature_columns, 'Tgt')

	# Define the number of neurons (LSTM units) for the Seq2Seq model
	neurons = 100
	# Define the dropout rate for regularization
	dropout = 0.000  # Adjust this value as needed
	# from tensorflow.keras.mixed_precision import experimental as mixed_precision
	# policy = mixed_precision.Policy('mixed_float16')
	# mixed_precision.set_policy(policy)
	optimizer_name = 'adamw'
	learning_rate=0.0001
	l1_reg = 0.0000
	l2_reg = 0.0005
	num_layers = 2
	
	batch_size = 4
	epochs = 10
	#add early stopping
	patience = 200 #epochs
	early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
	 #dynamically reduce learning rate
	lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
	training_start_time= datetime.now()
	# last_epoch = train_seq2seq_model_with_teacher_forcing(model, X_encoder_train, decoder_input_train, y_decoder_train, X_encoder_test, decoder_input_test, y_decoder_test, 
	#                                  batch_size, epochs, early_stopping, ax[0])

	#invoke the function to train the model using dynamic teacher forcing
	#model, X_encoder_train, y_decoder_train, X_encoder_test, y_decoder_test, 
	#batch_size, epochs, optimizer, loss_fn, ax, teacher_forcing_ratio=0.5
	# loss_fn = tf.keras.losses.MeanSquaredError()
	# teacher_forcing_ratio=0.5
	# last_epoch = train_seq2seq_model_with_dynamic_teacher_forcing(model, X_encoder_train, y_decoder_train,
	#                                             X_encoder_test, y_decoder_test,
	#                                             batch_size, epochs, optimizer, loss_fn, teacher_forcing_ratio, ax[0])
	
	# loss_fn = tf.keras.losses.MeanSquaredError()
	# train_seq2seq_with_scheduled_sampling(model, X_encoder_train, decoder_input_train, y_decoder_train,
	#                                       batch_size, epochs, optimizer, loss_fn, ax[0], sampling_decay=0.99)

	use_teacher_forcing = True
	# optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features, neurons=50, dropout=0.2, l1_reg=0.01, l2_reg=0.01, use_teacher_forcing=False
	model_fn = lambda: create_model(optimizer_name, learning_rate, n_steps_in, n_steps_out,
									n_features, neurons, dropout, l1_reg, l2_reg, use_teacher_forcing)
	# Access the model by invoking the lambda function
	model, model_name = model_fn()

	# n_splits = 2 #minimum 2
	# # Perform timefolded Cross-Validation
	# avg_rmse, avg_val_loss, rmse_scores, val_losses, last_epoch, scalers = (
	#     train_seq2seq_with_timefolding_teacher_forcing(
	#         model_fn, X_encoder_train_3d, decoder_input_train_3d, y_decoder_train_3d,
	#         feature_columns, n_splits, batch_size=batch_size,
	#         epochs=epochs, patience=patience, lr_scheduler=lr_scheduler, ax=ax[0]
	#     )
	# )
	validation_split=0.2
	
	teacher_forcing = True
	avg_rmse, avg_val_loss, last_epoch, history = (
		train_seq2seq_without_timefolding(
			model, X_encoder_train_3d, decoder_input_train_3d, y_decoder_train_3d,
			feature_columns, validation_split=validation_split, batch_size=batch_size,
			epochs=epochs, patience=patience, lr_scheduler=lr_scheduler, teacher_forcing=teacher_forcing
		)
	)

	ax[1].plot(history.history['loss'], label='Training Loss')
	ax[1].plot(history.history['val_loss'], label='Validation Loss')
	ax[1].legend()
	ax[1].set_title('Training and Validation Loss')
	ax[1].set_xlabel('Epoch')
	ax[1].set_ylabel('Loss')
	
	training_end_time= datetime.now()
	print("Last epoch: ", last_epoch)

	# plot_attention_weights(model, X_encoder_test)
	#get all column names except the target column
	print("columns: ", stock_data.columns)
	#remove the column with the name 'Tgt'
	column_names = stock_data.columns.tolist()
	column_names.remove('Tgt')
	
	predictions_with_X_test_df = evaluate_seq2seq_model_on_test_data(
		model, target_scaler, X_encoder_test_3d, decoder_input_test_3d, 
		X_encoder_test_df, y_decoder_test_df, n_steps_out,feature_columns,use_teacher_forcing)

	#check if predictions_with_X_test_df is empty or contains NaN values
	if predictions_with_X_test_df.empty or predictions_with_X_test_df.isnull().values.all():
		print("No predictions available for plotting.")
		input("Press Enter to continue...")
		
	# Preserve the 'Date' index as a column before resetting the index
	if 'Date' in X_encoder_test_df.index.names:
		print("Resetting index to preserve 'Date' column.")
		X_encoder_test_df = X_encoder_test_df.reset_index(level='Date')  # Retain 'Date' as a column

	# Unscale the test data
	X_encoder_test_df = X_encoder_test_df.reset_index(drop=True)
	X_encoder_test_df = X_encoder_test_df.loc[:, ~X_encoder_test_df.columns.str.contains('^Index$')]
	feature_columns_without_date_and_step = [col for col in feature_columns if col not in ['Date', 'Step']]
	X_encoder_test_array = X_encoder_test_df[feature_columns_without_date_and_step].values
	y_decoder_test_array = y_decoder_test_df[[col for col in y_decoder_test_df.columns if col.startswith('Tgt')]].values
	
	X_encoder_test_array_unscaled = feature_scaler.inverse_transform(X_encoder_test_array)

	# Debugging: Check the shape and size of y_decoder_test_array before reshaping
	print("Shape of y_decoder_test_array before reshaping:", y_decoder_test_array.shape)
	print("Size of y_decoder_test_array:", y_decoder_test_array.size)

	# Validate the size of y_decoder_test_array
	expected_size = y_decoder_test_array.shape[0] * y_decoder_test_array.shape[1]  # Target shape (128, 1) or (127,2) etc.
	actual_size = y_decoder_test_array.size
	assert actual_size == expected_size, f"Mismatch: expected size {expected_size}, but got {actual_size}."

	# Reshape y_decoder_test_array
	y_decoder_test_array_unscaled = target_scaler.inverse_transform(
		y_decoder_test_array.reshape(-1, 1)
	).reshape(y_decoder_test_array.shape[0], y_decoder_test_array.shape[1])

	# Debugging: Check the shape of y_decoder_test_array_unscaled
	print("Shape of y_decoder_test_array_unscaled:", y_decoder_test_array_unscaled.shape)

	# Convert to DataFrame
	X_encoder_test_df_unscaled = pd.DataFrame(X_encoder_test_array_unscaled, columns=feature_columns_without_date_and_step)
	# Add the 'Date' column back to X_encoder_test_df_unscaled
	if 'Date' in X_encoder_test_df.columns:
		X_encoder_test_df_unscaled['Date'] = X_encoder_test_df['Date']
		X_encoder_test_df_unscaled.set_index('Date', inplace=True)

	# Dynamically generate column names for the target columns based on n_steps_out
	if n_steps_out == 1:
		target_columns = ['Tgt']  # Single column for one-step prediction
	else:
		target_columns = [f'Tgt(t+{i})' if i > 0 else 'Tgt(t)' for i in range(n_steps_out)]

	y_decoder_test_df_unscaled = pd.DataFrame(y_decoder_test_array_unscaled, columns=target_columns)

	# Pass the DataFrame to the plot_predictions function
	plot_predictions(model_name, X_encoder_test_df_unscaled, y_decoder_test_df_unscaled, predictions_with_X_test_df, ax[0])
	duration = training_end_time - training_start_time
	#save training parameters to a dataframe
	training_params = {'model_name': model_name,'neurons': neurons, 'n_features': n_features, 'dropout': dropout, 'optimizer': optimizer_name, 
						'learning_rate': learning_rate, 'l1_reg': l1_reg, 'l2_reg':l2_reg, 'batch_size': batch_size, 'epochs': epochs,
						'patience': patience, 'training_start_time': training_start_time, 'training_end_time': training_end_time, 'duration':duration, 
						'last_epoch': last_epoch, 'feature_columns': feature_columns}
	# Ensure all values in the dictionary are scalar or convert non-scalar values to strings
	training_params_cleaned = {key: (str(value) if isinstance(value, (list, dict)) else value) for key, value in training_params.items()}

	# if 'n_splits' in locals():
	#     training_params['n_splits'] = n_splits
	training_params_df = pd.DataFrame([training_params_cleaned], index=[0])
	#save the plot to a pdf
	current_date = datetime.now().strftime('%Y-%m-%d')
	
	# test_update_features_after_prediction_seq2seq_all(
	# 	X_encoder_test_df_unscaled, y_decoder_test_df_unscaled, feature_columns_without_date_and_step,
	#     n_steps_in, n_steps_out
	# )
	# input("Press Enter to continue...")

	# Forecast the next 10 days
	n_days = 10
	predictions = forecast_next_n_days(
		model=model,
		target_scaler=target_scaler,
		X_encoder_test_3d=X_encoder_test_3d,
		y_encoder_test_3d=y_decoder_test_3d,
		n_steps_in=n_steps_in,
		n_steps_out=n_steps_out,
		n_features=n_features,
		n_days=n_days,
		ax=ax[2]
	)

	# Print the predictions
	print("Predictions for the next 10 days:", predictions)

	plt.tight_layout()
	save_model_and_results(training_params_df, plt, model, output_files_folder,file_prefix)

	plt.show()

#create a method to use sqllite to persist the model and training parameters for analytics
def save_model_and_results_to_db(training_params_df, plt, model, db_path):
	# Create a connection to the SQLite database
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	# Create a table to store the model and training parameters
	cursor.execute('''
		CREATE TABLE IF NOT EXISTS model_results (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			neurons INTEGER,
			n_steps_in INTEGER,
			n_steps_out INTEGER,
			n_features INTEGER,
			dropout REAL,
			optimizer TEXT,
			learning_rate REAL,
			l1_reg REAL,
			l2_reg REAL,
			num_layers INTEGER,
			batch_size INTEGER,
			epochs INTEGER,
			patience INTEGER,
			training_start_time TEXT,
			training_end_time TEXT,
			last_epoch INTEGER
		)
	''')

