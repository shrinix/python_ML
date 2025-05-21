import pandas as pd
import numpy as np
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from seq2seq_trainer import create_seq2seq_model
from itertools import product
import tensorflow as tf
from utils import (
    get_max_version, get_timeframes_df, load_configuration_params, prepare_data_for_training, prepare_seq2seq_data,
    fetch_data_into_files, load_and_transform_data_by_ticker, add_engineered_features, get_scaler,
    reverse_lookup,compare_dataframes
)
from seq2seq_trainer import (create_seq2seq_model, create_bidirectional_seq2seq_with_attention,
        create_seq2seq_model_with_attention
)
from sklearn.model_selection import KFold
from scipy.fft import fft
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import copy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

def test_prepare_seq2seq_data():
    # Define test cases
    test_cases = [
        {
            "name": "Basic Test Case",
            "data": pd.DataFrame({
                "Feature1": [1, 2, 3, 4, 5],
                "Feature2": [10, 20, 30, 40, 50],
                "Tgt": [100, 200, 300, 400, 500]
            }),
            "feature_columns": ["Feature1", "Feature2"],
            "target_column": "Tgt",
            "n_steps_in": 2,
            "n_steps_out": 2,
            "expected_X_encoder": np.array([
                [[1, 10], [2, 20]],
                [[2, 20], [3, 30]],
                [[3, 30], [4, 40]]
            ]),
            "expected_decoder_input": np.array([
                [[3, 30], [4, 40]],
                [[4, 40], [5, 50]],
                [[5, 50], [0, 0]]  # Padding for last sequence
            ]),
            "expected_y_decoder": np.array([
                [[300], [400]],
                [[400], [500]],
                [[500], [0]]  # Padding for last sequence
            ])
        }
    ]

    # Run test cases
    for test in test_cases:
        print(f"Running test: {test['name']}")
        X_encoder, decoder_input, y_decoder = prepare_seq2seq_data(
            data=test["data"],
            feature_columns=test["feature_columns"],
            target_column=test["target_column"],
            n_steps_in=test["n_steps_in"],
            n_steps_out=test["n_steps_out"]
        )

        # Debugging: Print actual and expected outputs
        print(f"Actual X_encoder for {test['name']}:")
        print(X_encoder)
        print(f"Expected X_encoder for {test['name']}:")
        print(test["expected_X_encoder"])

        # Validate results
        assert np.array_equal(X_encoder, test["expected_X_encoder"]), f"X_encoder mismatch in {test['name']}"
        assert np.array_equal(decoder_input, test["expected_decoder_input"]), f"decoder_input mismatch in {test['name']}"
        assert np.array_equal(y_decoder, test["expected_y_decoder"]), f"y_decoder mismatch in {test['name']}"
        print(f"Test {test['name']} passed!")

def test_create_seq2seq_model():
    """
    Test the creation of the Seq2Seq model with and without teacher forcing.
    """
    # Define test parameters
    optimizer_name = 'adam'
    learning_rate = 0.001
    n_steps_in = 10  # Number of input timesteps
    n_steps_out = 5  # Number of output timesteps
    n_features = 3   # Number of features
    neurons = 50     # Number of LSTM units
    dropout = 0.2    # Dropout rate
    l1_reg = 0.01    # L1 regularization
    l2_reg = 0.01    # L2 regularization

    # Test with teacher forcing
    print("\nTesting Seq2Seq model with teacher forcing...")
    model_with_teacher_forcing = create_seq2seq_model(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=True
    )

    # Check if the model is an instance of keras.Model
    assert isinstance(model_with_teacher_forcing, Model), "The created model with teacher forcing is not an instance of keras.Model."

    # Check the model's input and output shapes
    encoder_input_shape = model_with_teacher_forcing.input_shape[0]  # Encoder input shape
    decoder_input_shape = model_with_teacher_forcing.input_shape[1]  # Decoder input shape
    decoder_output_shape = model_with_teacher_forcing.output_shape   # Decoder output shape

    assert encoder_input_shape == (None, n_steps_in, n_features), (
        f"Expected encoder input shape to be (None, {n_steps_in}, {n_features}), "
        f"but got {encoder_input_shape}."
    )
    assert decoder_input_shape == (None, n_steps_out, n_features), (
        f"Expected decoder input shape to be (None, {n_steps_out}, {n_features}), "
        f"but got {decoder_input_shape}."
    )
    assert decoder_output_shape == (None, n_steps_out, 1), (
        f"Expected decoder output shape to be (None, {n_steps_out}, 1), "
        f"but got {decoder_output_shape}."
    )

    # Compile the model and check if it compiles successfully
    try:
        model_with_teacher_forcing.compile(optimizer='adam', loss='mse')
    except Exception as e:
        assert False, f"Model compilation failed with error: {e}"

    # Run a forward pass with dummy data and check the output shape
    dummy_encoder_input = np.random.rand(2, n_steps_in, n_features)  # Batch size = 2
    dummy_decoder_input = np.random.rand(2, n_steps_out, n_features)  # Batch size = 2

    predictions = model_with_teacher_forcing.predict([dummy_encoder_input, dummy_decoder_input])

    assert predictions.shape == (2, n_steps_out, 1), (
        f"Expected predictions shape to be (2, {n_steps_out}, 1), "
        f"but got {predictions.shape}."
    )

    print("Test passed: Seq2Seq model with teacher forcing created and verified successfully.")

    # Test without teacher forcing
    print("\nTesting Seq2Seq model without teacher forcing...")
    model_without_teacher_forcing = create_seq2seq_model(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=False
    )

    # Check if the model is an instance of keras.Model
    assert isinstance(model_without_teacher_forcing, Model), "The created model without teacher forcing is not an instance of keras.Model."

    # Check the model's input and output shapes
    encoder_input_shape = model_without_teacher_forcing.input_shape  # Encoder input shape
    decoder_output_shape = model_without_teacher_forcing.output_shape  # Decoder output shape

    assert encoder_input_shape == (None, n_steps_in, n_features), (
        f"Expected encoder input shape to be (None, {n_steps_in}, {n_features}), "
        f"but got {encoder_input_shape}."
    )
    assert decoder_output_shape == (None, n_steps_out, 1), (
        f"Expected decoder output shape to be (None, {n_steps_out}, 1), "
        f"but got {decoder_output_shape}."
    )

    # Compile the model and check if it compiles successfully
    try:
        model_without_teacher_forcing.compile(optimizer='adam', loss='mse')
    except Exception as e:
        assert False, f"Model compilation failed with error: {e}"

    # Run a forward pass with dummy data and check the output shape
    dummy_encoder_input = np.random.rand(2, n_steps_in, n_features)  # Batch size = 2

    predictions = model_without_teacher_forcing.predict(dummy_encoder_input)

    assert predictions.shape == (2, n_steps_out, 1), (
        f"Expected predictions shape to be (2, {n_steps_out}, 1), "
        f"but got {predictions.shape}."
    )

    print("Test passed: Seq2Seq model without teacher forcing created and verified successfully.")

def test_create_bidirectional_seq2seq_with_attention():
    """
    Test the creation of the Bidirectional Seq2Seq model with and without teacher forcing.
    """
    # Define test parameters
    optimizer_name = 'adam'
    learning_rate = 0.001
    n_steps_in = 10  # Number of input timesteps
    n_steps_out = 5  # Number of output timesteps
    n_features = 3   # Number of features
    neurons = 50     # Number of LSTM units
    dropout = 0.2    # Dropout rate
    l1_reg = 0.01    # L1 regularization
    l2_reg = 0.01    # L2 regularization

    # Test with teacher forcing
    print("\nTesting Bidirectional Seq2Seq model with teacher forcing...")
    model_with_teacher_forcing = create_bidirectional_seq2seq_with_attention(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=True
    )

    # Check if the model is an instance of keras.Model
    assert isinstance(model_with_teacher_forcing, Model), "The created model with teacher forcing is not an instance of keras.Model."

    # Check the model's input and output shapes
    encoder_input_shape = model_with_teacher_forcing.input_shape[0]  # Encoder input shape
    decoder_input_shape = model_with_teacher_forcing.input_shape[1]  # Decoder input shape
    decoder_output_shape = model_with_teacher_forcing.output_shape   # Decoder output shape

    assert encoder_input_shape == (None, n_steps_in, n_features), (
        f"Expected encoder input shape to be (None, {n_steps_in}, {n_features}), "
        f"but got {encoder_input_shape}."
    )
    assert decoder_input_shape == (None, n_steps_out, n_features), (
        f"Expected decoder input shape to be (None, {n_steps_out}, {n_features}), "
        f"but got {decoder_input_shape}."
    )
    assert decoder_output_shape == (None, n_steps_out, 1), (
        f"Expected decoder output shape to be (None, {n_steps_out}, 1), "
        f"but got {decoder_output_shape}."
    )

    # Compile the model and check if it compiles successfully
    try:
        model_with_teacher_forcing.compile(optimizer='adam', loss='mse')
    except Exception as e:
        assert False, f"Model compilation failed with error: {e}"

    # Run a forward pass with dummy data and check the output shape
    dummy_encoder_input = np.random.rand(2, n_steps_in, n_features)  # Batch size = 2
    dummy_decoder_input = np.random.rand(2, n_steps_out, n_features)  # Batch size = 2

    predictions = model_with_teacher_forcing.predict([dummy_encoder_input, dummy_decoder_input])

    assert predictions.shape == (2, n_steps_out, 1), (
        f"Expected predictions shape to be (2, {n_steps_out}, 1), "
        f"but got {predictions.shape}."
    )

    print("Test passed: Bidirectional Seq2Seq model with teacher forcing created and verified successfully.")

    # Test without teacher forcing
    print("\nTesting Bidirectional Seq2Seq model without teacher forcing...")
    model_without_teacher_forcing = create_bidirectional_seq2seq_with_attention(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=False
    )

    # Check if the model is an instance of keras.Model
    assert isinstance(model_without_teacher_forcing, Model), "The created model without teacher forcing is not an instance of keras.Model."

    # Check the model's input and output shapes
    encoder_input_shape = model_without_teacher_forcing.input_shape  # Encoder input shape
    decoder_output_shape = model_without_teacher_forcing.output_shape  # Decoder output shape

    assert encoder_input_shape == (None, n_steps_in, n_features), (
        f"Expected encoder input shape to be (None, {n_steps_in}, {n_features}), "
        f"but got {encoder_input_shape}."
    )
    assert decoder_output_shape == (None, n_steps_out, 1), (
        f"Expected decoder output shape to be (None, {n_steps_out}, 1), "
        f"but got {decoder_output_shape}."
    )

    # Compile the model and check if it compiles successfully
    try:
        model_without_teacher_forcing.compile(optimizer='adam', loss='mse')
    except Exception as e:
        assert False, f"Model compilation failed with error: {e}"

    # Run a forward pass with dummy data and check the output shape
    dummy_encoder_input = np.random.rand(2, n_steps_in, n_features)  # Batch size = 2

    predictions = model_without_teacher_forcing.predict(dummy_encoder_input)

    assert predictions.shape == (2, n_steps_out, 1), (
        f"Expected predictions shape to be (2, {n_steps_out}, 1), "
        f"but got {predictions.shape}."
    )

    print("Test passed: Bidirectional Seq2Seq model without teacher forcing created and verified successfully.")

def test_create_seq2seq_model_with_attention():
    """
    Test the creation of the Seq2Seq model with Attention, with and without teacher forcing.
    """
    # Define test parameters
    optimizer_name = 'adam'
    learning_rate = 0.001
    n_steps_in = 10  # Number of input timesteps
    n_steps_out = 5  # Number of output timesteps
    n_features = 3   # Number of features
    neurons = 50     # Number of LSTM units
    dropout = 0.2    # Dropout rate
    l1_reg = 0.01    # L1 regularization
    l2_reg = 0.01    # L2 regularization

    # Test with teacher forcing
    print("\nTesting Seq2Seq model with Attention and teacher forcing...")
    model_with_teacher_forcing = create_seq2seq_model_with_attention(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=True
    )

    # Check if the model is an instance of keras.Model
    assert isinstance(model_with_teacher_forcing, Model), "The created model with teacher forcing is not an instance of keras.Model."

    # Check the model's input and output shapes
    encoder_input_shape = model_with_teacher_forcing.input_shape[0]  # Encoder input shape
    decoder_input_shape = model_with_teacher_forcing.input_shape[1]  # Decoder input shape
    decoder_output_shape = model_with_teacher_forcing.output_shape   # Decoder output shape

    assert encoder_input_shape == (None, n_steps_in, n_features), (
        f"Expected encoder input shape to be (None, {n_steps_in}, {n_features}), "
        f"but got {encoder_input_shape}."
    )
    assert decoder_input_shape == (None, n_steps_out, n_features), (
        f"Expected decoder input shape to be (None, {n_steps_out}, {n_features}), "
        f"but got {decoder_input_shape}."
    )
    assert decoder_output_shape == (None, n_steps_out, 1), (
        f"Expected decoder output shape to be (None, {n_steps_out}, 1), "
        f"but got {decoder_output_shape}."
    )

    # Compile the model and check if it compiles successfully
    try:
        model_with_teacher_forcing.compile(optimizer='adam', loss='mse')
    except Exception as e:
        assert False, f"Model compilation failed with error: {e}"

    # Run a forward pass with dummy data and check the output shape
    dummy_encoder_input = np.random.rand(2, n_steps_in, n_features)  # Batch size = 2
    dummy_decoder_input = np.random.rand(2, n_steps_out, n_features)  # Batch size = 2

    predictions = model_with_teacher_forcing.predict([dummy_encoder_input, dummy_decoder_input])

    assert predictions.shape == (2, n_steps_out, 1), (
        f"Expected predictions shape to be (2, {n_steps_out}, 1), "
        f"but got {predictions.shape}."
    )

    print("Test passed: Seq2Seq model with Attention and teacher forcing created and verified successfully.")

    # Test without teacher forcing
    print("\nTesting Seq2Seq model with Attention and no teacher forcing...")
    model_without_teacher_forcing = create_seq2seq_model_with_attention(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=False
    )

    # Check if the model is an instance of keras.Model
    assert isinstance(model_without_teacher_forcing, Model), "The created model without teacher forcing is not an instance of keras.Model."

    # Check the model's input and output shapes
    encoder_input_shape = model_without_teacher_forcing.input_shape  # Encoder input shape
    decoder_output_shape = model_without_teacher_forcing.output_shape  # Decoder output shape

    assert encoder_input_shape == (None, n_steps_in, n_features), (
        f"Expected encoder input shape to be (None, {n_steps_in}, {n_features}), "
        f"but got {encoder_input_shape}."
    )
    assert decoder_output_shape == (None, n_steps_out, 1), (
        f"Expected decoder output shape to be (None, {n_steps_out}, 1), "
        f"but got {decoder_output_shape}."
    )

    # Compile the model and check if it compiles successfully
    try:
        model_without_teacher_forcing.compile(optimizer='adam', loss='mse')
    except Exception as e:
        assert False, f"Model compilation failed with error: {e}"

    # Run a forward pass with dummy data and check the output shape
    dummy_encoder_input = np.random.rand(2, n_steps_in, n_features)  # Batch size = 2

    predictions = model_without_teacher_forcing.predict(dummy_encoder_input)

    assert predictions.shape == (2, n_steps_out, 1), (
        f"Expected predictions shape to be (2, {n_steps_out}, 1), "
        f"but got {predictions.shape}."
    )

    print("Test passed: Seq2Seq model with Attention and no teacher forcing created and verified successfully.")

def test_seq2seq_model_learning_with_plot():
    """
    Test if the Seq2Seq model (without attention) is learning properly by checking
    loss reduction, prediction shape, prediction values, and plotting the results.
    """
    import numpy as np
    from keras.optimizers import Adam

    # Define test parameters
    optimizer_name = 'adam'
    learning_rate = 0.0005  # Reduced learning rate for smoother convergence
    n_steps_in = 10  # Number of input timesteps
    n_steps_out = 5  # Number of output timesteps
    n_features = 1   # Number of features
    neurons = 50     # Number of LSTM units
    dropout = 0.2    # Dropout rate
    l1_reg = 0.01    # L1 regularization
    l2_reg = 0.01    # L2 regularization

    # Create the Seq2Seq model (without attention)
    model = create_seq2seq_model(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=False
    )

    # Generate dummy training data (simple sine wave pattern)
    X_train = np.array([np.sin(np.linspace(0, 2 * np.pi, n_steps_in)) for _ in range(100)])  # 100 samples
    X_train = X_train.reshape(100, n_steps_in, n_features)  # Reshape to (samples, timesteps, features)
    y_train = np.array([np.sin(np.linspace(2 * np.pi, 2 * np.pi + n_steps_out, n_steps_out)) for _ in range(100)])
    y_train = y_train.reshape(100, n_steps_out, 1)  # Reshape to (samples, timesteps, features)

    # Normalize the data
    X_train = (X_train + 1) / 2  # Scale to [0, 1]
    y_train = (y_train + 1) / 2  # Scale to [0, 1]

    # Train the model for more epochs
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # Check if the loss decreases
    losses = history.history['loss']
    assert losses[-1] < losses[0], "Training loss did not decrease, model might not be learning."

    # Generate dummy test data
    X_test = np.array([np.sin(np.linspace(0, 2 * np.pi, n_steps_in)) for _ in range(10)])  # 10 samples
    X_test = X_test.reshape(10, n_steps_in, n_features)  # Reshape to (samples, timesteps, features)
    y_test = np.array([np.sin(np.linspace(2 * np.pi, 2 * np.pi + n_steps_out, n_steps_out)) for _ in range(10)])
    y_test = y_test.reshape(10, n_steps_out, 1)  # Reshape to (samples, timesteps, features)

    # Normalize the test data
    X_test = (X_test + 1) / 2  # Scale to [0, 1]
    y_test = (y_test + 1) / 2  # Scale to [0, 1]

    # Make predictions
    predictions = model.predict(X_test)

    # Check the shape of predictions
    assert predictions.shape == (10, n_steps_out, 1), (
        f"Expected predictions shape to be (10, {n_steps_out}, 1), but got {predictions.shape}."
    )

    # Check if the predictions are close to the expected values
    mse = np.mean((predictions - y_test) ** 2)
    assert mse < 0.01, f"Mean Squared Error is too high: {mse}. Model might not be learning properly."

    # Plot the input, expected output, and predictions for the first test sample
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_steps_in), X_test[0].flatten(), label="Input (X_test)", marker='o')
    plt.plot(range(n_steps_in - 1, n_steps_in + n_steps_out), 
            [X_test[0, -1, 0]] + y_test[0].flatten().tolist(), 
            label="Expected Output (y_test)", marker='o')
    plt.plot(range(n_steps_in - 1, n_steps_in + n_steps_out), 
            [X_test[0, -1, 0]] + predictions[0].flatten().tolist(), 
            label="Predicted Output", marker='o')
    plt.title("Seq2Seq Model Predictions")
    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.grid()
    plt.show()

    print("Test passed: Seq2Seq model (without attention) is learning properly with correct predictions.")

def test_seq2seq_model_learning_with_random_data(
    X_train, y_train, X_test, y_test, n_steps_in, n_steps_out, n_features, neurons,
    dropout, learning_rate, epochs, batch_size, use_teacher_forcing=False
):
    """
    Test if the Seq2Seq model (without attention) is learning properly by checking
    loss reduction, prediction shape, prediction values, and plotting the results
    using random data instead of a sine wave.
    n_steps_in: # Number of input timesteps
    n_steps_out: # Number of output timesteps
    n_features: # Number of features

    """
    import numpy as np
    import matplotlib.pyplot as plt

    print("Epochs:", epochs)

    # Define test parameters
    optimizer_name = 'adam'
    learning_rate = learning_rate  # Reduced learning rate for smoother convergence
    neurons = neurons     # Number of LSTM units
    dropout = dropout    # Dropout rate
    l1_reg = 0.01    # L1 regularization
    l2_reg = 0.01    # L2 regularization

    use_teacher_forcing=False
    # Create the Seq2Seq model (without attention)
    model = create_seq2seq_model(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing
    )
    # Check the model's output shape
    assert model.output_shape == (None, n_steps_out, 1), f"Expected output shape (None, {n_steps_out}, 1), but got {model.output_shape}"

    # Train the model for more epochs
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    # Check if the loss decreases
    losses = history.history['loss']
    assert losses[-1] < losses[0], "Training loss did not decrease, model might not be learning."

    # Make predictions
    predictions = model.predict(X_test)
    # Check the shape of predictions
    assert predictions.shape == (X_test.shape[0], n_steps_out, 1), (
        f"Expected predictions shape to be ({X_test.shape[0]}, {n_steps_out}, 1), but got {predictions.shape}."
    )

    # Check if the predictions are close to the expected values
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Mean Squared Error: {mse}")
    assert mse < 0.5, f"Mean Squared Error is too high: {mse}. Model might not be learning properly."

    print("Test passed: Seq2Seq model (without attention) is learning properly with random data.")
    return predictions

def test_seq2seq_model_learning_with_random_data_teacher_forcing(
    X_train, y_train, X_test, y_test, n_steps_in, n_steps_out, n_features, neurons,
    dropout, learning_rate, epochs, batch_size
):
    """
    Test if the Seq2Seq model (with teacher forcing) is learning properly by checking
    loss reduction, prediction shape, prediction values, and plotting the results
    using random data.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Define test parameters
    optimizer_name = 'adam'
    learning_rate = learning_rate  # Reduced learning rate for smoother convergence
    neurons = neurons   # Number of LSTM units
    dropout = dropout    # Dropout rate
    l1_reg = 0.1    # L1 regularization
    l2_reg = 0.1    # L2 regularization

    # Create the Seq2Seq model (with teacher forcing)
    model = create_seq2seq_model(
        optimizer_name, learning_rate, n_steps_in, n_steps_out, n_features,
        neurons, dropout, l1_reg, l2_reg, use_teacher_forcing=True
    )

    # When using teacher-forcing, actual target values (y_train) are fed as inputs to the
    # decoder at each timestep instead of the decoder's own predictions from the previous timestep.
    history = model.fit([X_train, y_train], y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Check if the loss decreases
    losses = history.history['loss']
    assert losses[-1] < losses[0], "Training loss did not decrease, model might not be learning."

    # Make predictions
    # predictions = model.predict([X_test, y_test])
    # Predict without teacher forcing during inference without using y_test. Instead, the decoder's predictions 
    # are fed back into the decoder for the next timestep.
    predictions = []
    decoder_input = np.zeros((X_test.shape[0], n_steps_out, n_features))  # Initialize with zeros

    decoder_input = np.zeros((X_test.shape[0], n_steps_out, n_features))  # Initialize with zeros

    # Start with the first timestep of y_test as the initial input
    decoder_input[:, 0, :] = y_test[:, 0, :]  # Use the first timestep of y_test

    for t in range(n_steps_out):
        # Predict the next timestep
        pred = model.predict([X_test, decoder_input], verbose=1)
        
        # Append the prediction for the current timestep
        predictions.append(pred[:, t:t+1, :])  # Extract the prediction for the current timestep
        
        # Update the decoder input for the next timestep
        if t + 1 < n_steps_out:
            decoder_input[:, t + 1, :] = pred[:, t, :]  # Use the current prediction as input for the next timestep

    # Combine predictions along the time axis
    predictions = np.concatenate(predictions, axis=1)  # Shape: (samples, n_steps_out, n_features)

    # Check the shape of predictions
    assert predictions.shape == (X_test.shape[0], n_steps_out, 1), (
        f"Expected predictions shape to be ({X_test.shape[0]}, {n_steps_out}, 1), but got {predictions.shape}."
    )

    # Check if the predictions are close to the expected values
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Mean Squared Error: {mse}")
    assert mse < 0.6, f"Mean Squared Error is too high: {mse}. Model might not be learning properly."

    print("Test passed: Seq2Seq model (with teacher forcing) is learning properly with random data.")

    return predictions

# Run the tests
# test_prepare_seq2seq_data()
# test_create_seq2seq_model()
# test_create_bidirectional_seq2seq_with_attention()
# test_create_seq2seq_model_with_attention()
# test_seq2seq_model_learning_with_plot()

def grid_search_seq2seq(X_train, y_train, X_val, y_val, n_steps_in, n_steps_out, n_features):
    """
    Perform grid search to find the best hyperparameters for the Seq2Seq model.

    Parameters:
    - X_train, y_train: Training data.
    - X_val, y_val: Validation data.
    - n_steps_in: Number of input timesteps.
    - n_steps_out: Number of output timesteps.
    - n_features: Number of features in the input data.

    Returns:
    - best_params: Dictionary of the best hyperparameters.
    - best_mse: Mean Squared Error for the best model.
    """
    # Define the hyperparameter space
    param_grid = {
        'neurons': [32, 50, 64],  # Number of LSTM units
        'dropout': [0.2, 0.3],   # Dropout rate
        'learning_rate': [0.001, 0.0005],  # Learning rate
        'batch_size': [16, 32],  # Batch size
        'epochs': [30, 50]       # Number of epochs
    }

    # Generate all combinations of hyperparameters
    param_combinations = list(product(
        param_grid['neurons'],
        param_grid['dropout'],
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['epochs']
    ))

    best_mse = float('inf')
    best_params = None

    # Iterate over all combinations of hyperparameters
    for neurons, dropout, learning_rate, batch_size, epochs in param_combinations:
        print(f"Testing combination: neurons={neurons}, dropout={dropout}, "
              f"learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

        # Create the model
        model = create_seq2seq_model(
            optimizer_name='adam',
            learning_rate=learning_rate,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            n_features=n_features,
            neurons=neurons,
            dropout=dropout,
            l1_reg=0.01,
            l2_reg=0.01,
            use_teacher_forcing=False
        )

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluate the model on validation data
        predictions = model.predict(X_val)
        mse = np.mean((predictions - y_val) ** 2)

        print(f"Mean Squared Error: {mse}")

        # Update the best parameters if the current combination is better
        if mse < best_mse:
            best_mse = mse
            best_params = {
                'neurons': neurons,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs
            }

    print(f"Best parameters: {best_params}")
    print(f"Best Mean Squared Error: {best_mse}")

    return best_params, best_mse

def grid_search_seq2seq_with_cv(X, y, n_steps_in, n_steps_out, n_features, n_splits=3):
    """
    Perform grid search with cross-validation to find the best hyperparameters for the Seq2Seq model.
    """
    param_grid = {
        'neurons': [32, 50, 64],
        'dropout': [0.2, 0.3],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [16, 32],
        'epochs': [30, 50]
    }

    param_combinations = list(product(
        param_grid['neurons'],
        param_grid['dropout'],
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['epochs']
    ))

    best_mse = float('inf')
    best_params = None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for neurons, dropout, learning_rate, batch_size, epochs in param_combinations:
        print(f"Testing combination: neurons={neurons}, dropout={dropout}, "
              f"learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

        mse_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = create_seq2seq_model(
                optimizer_name='adam',
                learning_rate=learning_rate,
                n_steps_in=n_steps_in,
                n_steps_out=n_steps_out,
                n_features=n_features,
                neurons=neurons,
                dropout=dropout,
                l1_reg=0.01,
                l2_reg=0.01,
                use_teacher_forcing=False
            )

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            predictions = model.predict(X_val)
            mse = np.mean((predictions - y_val) ** 2)
            mse_scores.append(mse)

        avg_mse = np.mean(mse_scores)
        print(f"Average MSE: {avg_mse}")

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = {
                'neurons': neurons,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs
            }

    print(f"Best parameters: {best_params}")
    print(f"Best Mean Squared Error: {best_mse}")
    return best_params, best_mse

#A random walk is a time-series where each value depends on the previous value plus some random noise.
def generate_random_walk(n_samples, n_timesteps, n_features):
    """
    Generate random walk time-series data.

    Parameters:
    - n_samples: Number of samples.
    - n_timesteps: Number of timesteps per sample.
    - n_features: Number of features.

    Returns:
    - random_walk: 3D NumPy array of shape (n_samples, n_timesteps, n_features).
    """
    random_walk = np.cumsum(np.random.randn(n_samples, n_timesteps, n_features), axis=1)
    return random_walk

#Simulate time-series data with seasonal and trend components.
def generate_seasonal_trend_series(n_samples, n_timesteps, n_features):
    """
    Generate time-series data with seasonal and trend components.

    Parameters:
    - n_samples: Number of samples.
    - n_timesteps: Number of timesteps per sample.
    - n_features: Number of features.

    Returns:
    - seasonal_trend_series: 3D NumPy array of shape (n_samples, n_timesteps, n_features).
    """
    trend = np.linspace(0, 1, n_timesteps).reshape(1, n_timesteps, 1)  # Linear trend
    trend = np.repeat(trend, n_samples, axis=0)  # Repeat for all samples
    seasonality = np.sin(np.linspace(0, 2 * np.pi, n_timesteps)).reshape(1, n_timesteps, 1)  # Seasonal pattern
    seasonality = np.repeat(seasonality, n_samples, axis=0)  # Repeat for all samples
    noise = 0.1 * np.random.randn(n_samples, n_timesteps, n_features)  # Random noise
    seasonal_trend_series = trend + seasonality + noise
    return seasonal_trend_series

#Generate sine waves with added random noise to simulate periodic patterns with variability.
def generate_sine_wave_with_noise(n_samples, n_timesteps, n_features, noise_level=0.1):
    """
    Generate sine wave time-series data with random noise.

    Parameters:
    - n_samples: Number of samples.
    - n_timesteps: Number of timesteps per sample.
    - n_features: Number of features.
    - noise_level: Standard deviation of the random noise.

    Returns:
    - sine_wave: 3D NumPy array of shape (n_samples, n_timesteps, n_features).
    """
    sine_wave = np.array([
        np.sin(np.linspace(0, 2 * np.pi, n_timesteps)) + noise_level * np.random.randn(n_timesteps)
        for _ in range(n_samples)
    ])
    sine_wave = sine_wave.reshape(n_samples, n_timesteps, n_features)
    return sine_wave

#Combine multiple patterns (e.g., random walk, sine wave, and noise) to create more complex time-series data.
def generate_combined_time_series(n_samples, n_timesteps, n_features):
    """
    Generate time-series data combining random walk, sine wave, and noise.

    Parameters:
    - n_samples: Number of samples.
    - n_timesteps: Number of timesteps per sample.
    - n_features: Number of features.

    Returns:
    - combined_series: 3D NumPy array of shape (n_samples, n_timesteps, n_features).
    """
    random_walk = generate_random_walk(n_samples, n_timesteps, n_features)
    sine_wave = generate_sine_wave_with_noise(n_samples, n_timesteps, n_features, noise_level=0.05)
    noise = 0.1 * np.random.randn(n_samples, n_timesteps, n_features)
    combined_series = random_walk + sine_wave + noise
    return combined_series

#Generate time-series data where each value depends on a weighted sum of previous values plus random noise.
def generate_autoregressive_series(n_samples, n_timesteps, n_features, coeff=0.8, noise_level=0.1):
    """
    Generate autoregressive time-series data.

    Parameters:
    - n_samples: Number of samples.
    - n_timesteps: Number of timesteps per sample.
    - n_features: Number of features.
    - coeff: Autoregressive coefficient (controls dependence on previous values).
    - noise_level: Standard deviation of the random noise.

    Returns:
    - autoregressive_series: 3D NumPy array of shape (n_samples, n_timesteps, n_features).
    """
    autoregressive_series = np.zeros((n_samples, n_timesteps, n_features))
    for i in range(1, n_timesteps):
        autoregressive_series[:, i, :] = (
            coeff * autoregressive_series[:, i - 1, :] + noise_level * np.random.randn(n_samples, n_features)
        )
    return autoregressive_series

def build_and_train_single_step_lstm(neurons=50, epochs=50, batch_size=16):
    """
    Train a single-step LSTM model to predict the next timestep.

    Parameters:
    - X_train: Input data of shape (samples, timesteps, features).
    - y_train: Target data of shape (samples, 1, features).
    - n_features: Number of features in the input data.
    - neurons: Number of LSTM units.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.

    Returns:
    - model: Trained LSTM model.
    """

    # Generate random time-series data
    n_samples = 10000
    n_timesteps = 10
    initial_features = 1
    # Choose a method to generate time-series data
    X_lstm = generate_combined_time_series(n_samples, n_timesteps, initial_features)
    y_lstm = generate_combined_time_series(n_samples, n_timesteps, initial_features)

    #split the data into training and test sets
    X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y_lstm, test_size=0.1, random_state=42)
    # X_test = generate_combined_time_series(10, n_timesteps, n_features)
    # y_test = generate_combined_time_series(10, n_timesteps, n_features)
    # Trim y_train and y_test to match n_steps_out
    y_lstm_train = y_lstm_train[:, :n_steps_out, :]  # Keep only the first n_steps_out timesteps
    y_lstm_test = y_lstm_test[:, :n_steps_out, :]    # Keep only the first n_steps_out timesteps

    print("Trimmed y_lstm_train shape:", y_lstm_train.shape)
    print("Trimmed y_lstm_test shape:", y_lstm_test.shape)
    print("X_lstm_train shape:", X_lstm_train.shape)
    print("y_lstm_train shape:", y_lstm_train.shape)

    print("Adding engineered features to the LSTM training and test data...")
    X_lstm_train_with_features, features= compute_and_add_engineered_features(X_lstm_train)
    print(X_lstm_train_with_features.shape)
    print(y_lstm_train.shape)

    X_lstm_test_with_features, features= compute_and_add_engineered_features(X_test)
    print(X_lstm_test_with_features.shape)
    print(y_lstm_test.shape)

    n_features = X_lstm_train_with_features.shape[1]  # Update n_features to include engineered features
    print(f"Number of features after engineering: {n_features}")

    # 8. Normalize all data
    scaler = MinMaxScaler()
    # Flatten the 3D array to 2D for scaling
    n_samples, n_timesteps, n_features = X_lstm_train_with_features.shape
    X_lstm_train_scaled = scaler.fit_transform(X_lstm_train_with_features.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)
    # Normalize y_train and y_test using a separate scaler
    scaler_y = MinMaxScaler()
    y_lstm_train_scaled = scaler_y.fit_transform(y_lstm_train.reshape(-1, 1)).reshape(y_lstm_train.shape)
    #X_train_scaled should have shape (samples, timesteps, augmented_features)
    #X_train_scaled should have shape (samples, timesteps, 1).
    print("X_lstm_train_scaled shape:", X_lstm_train_scaled.shape)
    print("y_lstm_train_scaled shape:", y_lstm_train_scaled.shape)

    n_samples_test, n_timesteps_test, n_features_test = X_lstm_test_with_features.shape
    X_lstm_test_scaled = scaler.transform(X_lstm_test_with_features.reshape(-1, n_features_test)).reshape(n_samples_test, n_timesteps_test, n_features_test)
    y_lstm_test_scaled = scaler_y.transform(y_lstm_test.reshape(-1, 1)).reshape(y_lstm_test.shape)
    print("X_lstm_test_scaled shape:", X_lstm_test_scaled.shape)
    print("y_lstm_test_scaled shape:", y_lstm_test_scaled.shape)

    model = Sequential()
    # First LSTM layer
    model.add(LSTM(neurons, return_sequences=True, input_shape=(X_lstm_train_scaled.shape[1], n_features)))
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting

    # Second LSTM layer
    model.add(LSTM(neurons, return_sequences=False))  # No need to return sequences in the final LSTM layer
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting

    # Fully connected Dense layers
    model.add(Dense(64, activation='relu'))  # Add a Dense layer with 64 units
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting

    model.add(Dense(1))  # Output a single feature to match y_train
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_lstm_train_scaled, y_lstm_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions
    predictions = model.predict(X_lstm_test_scaled)
    # Check the shape of predictions; only one timestep output is expected
    assert predictions.shape == (X_lstm_test_scaled.shape[0], 1), (
        f"Expected predictions shape to be ({X_lstm_test_scaled.shape[0]}, 1), but got {predictions.shape}."
    )
    
    # Use only the first timestep of y_test for comparison
    y_test_single_timestep = y_test_scaled[:, 0, :]  # Shape: (1000, 1)

    # Calculate MSE to check if the predictions are close to the expected values
    mse = np.mean((predictions - y_test_single_timestep) ** 2)
    print(f"Mean Squared Error: {mse}")
    assert mse < 1.5, f"Mean Squared Error is too high: {mse}. Model might not be learning properly."

    # #inverse transform X_train_scaled and y_train_scaled, X_test_scaled and y_test_scaled
    # X_lstm_train = scaler.inverse_transform(X_lstm_train_scaled.reshape(-1, n_features)).reshape(X_lstm_train_scaled.shape)
    # y_lstm_train_scaled = scaler_y.inverse_transform(y_lstm_train_scaled.reshape(-1, 1)).reshape(y_lstm_train_scaled.shape)
    # X_test_scaled = scaler.inverse_transform(X_test_scaled.reshape(-1, n_features)).reshape(X_test_scaled.shape)
    # y_test_scaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)

    return model, X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test

def add_lstm_features(X, lstm_predictions):
    """
    Use a trained LSTM model to generate features by predicting the next timestep.

    Parameters:
    - model: Trained LSTM model.
    - X: Input data of shape (samples, timesteps, features).
    - n_features: Number of features in the input data.

    Returns:
    - X_with_lstm_features: Augmented dataset with LSTM-generated features.
    """

    # Repeat the predicted features across all timesteps
    lstm_features_repeated = np.repeat(lstm_predictions[:, np.newaxis, :], X.shape[1], axis=1)

    # Append the LSTM-generated features to the original dataset
    X_with_lstm_features = np.concatenate([X, lstm_features_repeated], axis=2)
    return X_with_lstm_features

def compute_and_add_engineered_features(X):
    """
    Compute and add engineered features to the input dataset.

    Parameters:
    - X: 3D NumPy array of shape (samples, timesteps, features), representing the input dataset.

    Returns:
    - X_augmented: 3D NumPy array with additional engineered features added as new columns.
    """
    # Initialize the augmented dataset with the original data
    X_augmented = X.copy()

    # Number of samples, timesteps, and features
    n_samples, n_timesteps, n_features = X.shape

    features = []
    # 1. Statistical Features
    mean_feature = np.mean(X, axis=1, keepdims=True)  # Shape: (samples, 1, features)
    std_feature = np.std(X, axis=1, keepdims=True)    # Shape: (samples, 1, features)
    min_feature = np.min(X, axis=1, keepdims=True)    # Shape: (samples, 1, features)
    max_feature = np.max(X, axis=1, keepdims=True)    # Shape: (samples, 1, features)
    range_feature = max_feature - min_feature         # Shape: (samples, 1, features)
    
    # Broadcast statistical features to match the timesteps
    mean_feature = np.repeat(mean_feature, n_timesteps, axis=1)
    std_feature = np.repeat(std_feature, n_timesteps, axis=1)
    min_feature = np.repeat(min_feature, n_timesteps, axis=1)
    max_feature = np.repeat(max_feature, n_timesteps, axis=1)
    range_feature = np.repeat(range_feature, n_timesteps, axis=1)

    # Add statistical features
    X_augmented = np.concatenate([X_augmented, mean_feature, std_feature, min_feature, max_feature, range_feature], axis=2)
    features.extend(["mean", "std", "min", "max", "range"])

    # 2. Temporal Features
    time_index = np.linspace(0, 1, n_timesteps).reshape(1, n_timesteps, 1)  # Normalized time index
    time_index = np.repeat(time_index, n_samples, axis=0)  # Repeat for all samples
    X_augmented = np.concatenate([X_augmented, time_index], axis=2)
    features.append("time_index")

    # 3. Frequency Domain Features
    fft_features = np.abs(fft(X, axis=1))  # Compute FFT along the time axis
    fft_features = fft_features[:, :, :n_features]  # Keep only the first `n_features` FFT components
    X_augmented = np.concatenate([X_augmented, fft_features], axis=2)
    features.extend(["fft_real", "fft_imag"])  # Ensure this is correct

    # 4. Trend and Seasonality (using Savitzky-Golay filter for smoothing)
    smoothed = savgol_filter(X, window_length=5, polyorder=2, axis=1)  # Apply smoothing
    X_augmented = np.concatenate([X_augmented, smoothed], axis=2)
    features.append("smoothed")

    # 5. Derived Mathematical Features
    cumulative_sum = np.cumsum(X, axis=1)  # Cumulative sum
    cumulative_product = np.cumprod(X + 1e-6, axis=1)  # Cumulative product (add small value to avoid zeros)
    X_augmented = np.concatenate([X_augmented, cumulative_sum, cumulative_product], axis=2)
    features.extend(["cumulative_sum", "cumulative_product"])

    # 6. Rolling Window Features
    rolling_mean = np.zeros_like(X)
    rolling_variance = np.zeros_like(X)
    window_size = 3
    for i in range(n_timesteps):
        start = max(0, i - window_size + 1)
        rolling_mean[:, i, :] = np.mean(X[:, start:i+1, :], axis=1)
        rolling_variance[:, i, :] = np.var(X[:, start:i+1, :], axis=1)
    X_augmented = np.concatenate([X_augmented, rolling_mean, rolling_variance], axis=2)
    features.extend(["rolling_mean", "rolling_variance"])

    # 7. Interaction Features
    if n_features > 1:
        interaction_features = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_features.append(X[:, :, i] * X[:, :, j])  # Multiply features
        interaction_features = np.stack(interaction_features, axis=2)  # Stack along the feature axis
        X_augmented = np.concatenate([X_augmented, interaction_features], axis=2)
        features.extend([f"interaction_{i}_{j}" for i in range(n_features) for j in range(i + 1, n_features)])

    # Ensure that the output X_augmented has the shape (samples, timesteps, augmented_features)
    # Do not create a dataframe here, as it reduces dimensionality to 2D
    return X_augmented, features

# n_steps_in = 10
# n_steps_out = 5
# n_features = 1
# Generate random training and validation data for grid search
# X_train = np.random.rand(100, n_steps_in, n_features)
# y_train = np.random.rand(100, n_steps_out, 1)
# n_splits = 3 #used for k-fold cross-validation in grid search
# Perform grid search
# best_params, best_mse = grid_search_seq2seq_with_cv(
#     X_train, y_train, n_steps_in, n_steps_out, n_features, n_splits)
# print(f"Best parameters: {best_params}")

#Best parameters: {'neurons': 32, 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 16, 'epochs': 50}
best_params = {
    'neurons': 32,
    'dropout': 0.3,
    'learning_rate': 0.0001,
    'batch_size': 16,
    'epochs': 5
}
neurons = best_params['neurons']
dropout = best_params['dropout']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
epochs = best_params['epochs']

#create a plot with two rows with two subplots shown side by side in each row
fig, axs = plt.subplots(2, 2, figsize=(12, 6), squeeze=False)

# Define test parameters
n_steps_in = 10
n_steps_out = 5

# initial_features = 1
# # Generate random training and test data
# X_train = np.random.rand(100, n_steps_in, initial_features)  # 100 samples of random input sequences
# y_train = np.random.rand(100, n_steps_out, 1)  # 100 samples of random output sequences
# X_test = np.random.rand(10, n_steps_in, initial_features)  # 10 samples of random input sequences
# y_test = np.random.rand(10, n_steps_out, 1)  # 10 samples of random output sequences

# Generate random time-series data
n_samples = 10000
n_timesteps = 10
initial_features = 1
# Choose a method to generate time-series data
X_train = generate_combined_time_series(n_samples, n_timesteps, initial_features)
y_train = generate_combined_time_series(n_samples, n_timesteps, initial_features)

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# X_test = generate_combined_time_series(10, n_timesteps, n_features)
# y_test = generate_combined_time_series(10, n_timesteps, n_features)
# Trim y_train and y_test to match n_steps_out
y_train = y_train[:, :n_steps_out, :]  # Keep only the first n_steps_out timesteps
y_test = y_test[:, :n_steps_out, :]    # Keep only the first n_steps_out timesteps

print("Trimmed y_train shape:", y_train.shape)
print("Trimmed y_test shape:", y_test.shape)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("Adding engineered features to the training and test data...")
X_train_with_features, features= compute_and_add_engineered_features(X_train)
print(X_train_with_features.shape)
print(y_train.shape)

X_test_with_features, features= compute_and_add_engineered_features(X_test)
print(X_test_with_features.shape)
print(y_test.shape)

n_features = X_test_with_features.shape[1]  # Update n_features to include engineered features
print(f"Number of features after engineering: {n_features}")

# 8. Normalize all data
scaler = MinMaxScaler()
# Flatten the 3D array to 2D for scaling
n_samples, n_timesteps, n_features = X_train_with_features.shape
X_train_scaled = scaler.fit_transform(X_train_with_features.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)
# Normalize y_train and y_test using a separate scaler
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
#X_train_scaled should have shape (samples, timesteps, augmented_features)
#X_train_scaled should have shape (samples, timesteps, 1).
print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train_scaled shape:", y_train_scaled.shape)

n_samples_test, n_timesteps_test, n_features_test = X_test_with_features.shape
X_test_scaled = scaler.transform(X_test_with_features.reshape(-1, n_features_test)).reshape(n_samples_test, n_timesteps_test, n_features_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_test_scaled shape:", y_test_scaled.shape)

print("Building and training a single LSTM ...")
lstm_n_steps_in = 1
lstm_n_steps_out = 1

#Use train_single_step_lstm to train a single-step LSTM model
single_step_model, X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test = build_and_train_single_step_lstm(neurons=neurons, epochs=epochs, batch_size=batch_size)
#Use the trained single-step LSTM model to generate features for the training and test data
# Predict the next timestep for each sequence in X_train_scaled and X_test_scaled
#TODO: Move the LSTM code to a separate function and call it before original training and testing data are
#scaled. This will also require unscaling the LSTM predictions to match the original data before it is processed
#by the seq2seq model.
lstm_predictions_Xtrain = single_step_model.predict(X_train_scaled)  # Shape: (samples, features)
print("lstm_predictions_Xtrain shape:", lstm_predictions_Xtrain.shape)
print("lstm_predictions_Xtrain:", lstm_predictions_Xtrain)

X_train_with_lstm_features = add_lstm_features(X_train_scaled, lstm_predictions_Xtrain)
#Plot the input, expected output, and predictions for the the above single step LSTM model
#it takes X_train_scaled as input and lstm_predictions_Xtrain as output
#plot the predctions against the actual values
# Plot the input sequence
axs[0,0].plot(range(X_train_scaled.shape[1]), X_train_scaled[0, :, 0], label="Input (X_test)", marker='o')

# Plot the expected output
x_expected = range(n_timesteps - 1, n_timesteps - 1 + len(y_train_scaled[0].flatten()) + 1)
y_expected = [X_train_scaled[0, -1, 0]] + y_train_scaled[0].flatten().tolist()
axs[0,0].plot(x_expected, y_expected, label="Expected Output (y_test)", marker='o')

# Plot the predicted output
x_predicted = range(n_timesteps - 1, n_timesteps - 1 + len(lstm_predictions_Xtrain[0].flatten()) + 1)
y_predicted = [X_train_scaled[0, -1, 0]] + lstm_predictions_Xtrain[0].flatten().tolist()
axs[0,0].plot(x_predicted, y_predicted, label="Predicted Output", marker='o')

axs[0,0].set_title("LSTM Model Predictions")
axs[0,0].set_xlabel("Timesteps")
axs[0,0].set_ylabel("Normalized Values")
axs[0,0].legend()
axs[0,0].grid()

#add a text box with details of the training and testing
textstr = '\n'.join((
    rf'$\mathrm{{Epochs}}={epochs}$',
    rf'$\mathrm{{RMSE}}={np.sqrt(np.mean((lstm_predictions_Xtrain - y_train_scaled[:, 0, :]) ** 2)):.4f}$',
))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
# Add the text box to the plot
axs[0,0].text(0.05, 0.95, textstr, transform=axs[0,0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

lstm_predictions_Xtest = single_step_model.predict(X_test_scaled)  # Shape: (samples, features)
X_test_with_lstm_features = add_lstm_features(X_test_scaled, lstm_predictions_Xtest)

#X_train_scaled should have shape (samples, timesteps, augmented_features)
print("X_train_scaled shape after LSTM feature generation:", X_train_scaled.shape)
print("X_test_scaled shape after LSTM feature generation:", X_test_scaled.shape)

# Update the number of features to include the LSTM-generated feature
n_features_with_lstm = X_train_with_lstm_features.shape[2]
print(f"Number of features after LSTM feature generation: {n_features}")

# Verify the shape of X_train_with_lstm_features
print("X_train_with_lstm_features shape:", X_train_with_lstm_features.shape)
print("Epochs:", epochs)

predictions = test_seq2seq_model_learning_with_random_data(
    X_train=X_train_with_lstm_features,
    y_train=y_train_scaled,
    X_test=X_test_with_lstm_features,
    y_test=y_test_scaled,
    n_steps_in=n_steps_in,
    n_steps_out=n_steps_out,
    n_features=n_features_with_lstm,
    neurons=neurons,
    dropout=dropout,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    use_teacher_forcing=False
)

axs[1,0].plot(range(n_steps_in), X_test[0].flatten(), label="Input (X_test)", marker='o')
axs[1,0].plot(range(n_steps_in - 1, n_steps_in + n_steps_out),
        [X_test[0, -1, 0]] + y_test[0].flatten().tolist(),
        label="Expected Output (y_test)", marker='o')
axs[1,0].plot(range(n_steps_in - 1, n_steps_in + n_steps_out),
        [X_test[0, -1, 0]] + predictions[0].flatten().tolist(),
        label="Predicted Output", marker='o')
axs[1,0].set_title("Seq2Seq Model Predictions with Random Data (Without Teacher Forcing)")
axs[1,0].set_xlabel("Timesteps")
axs[1,0].set_ylabel("Values")
axs[1,0].legend()
axs[1,0].grid()
#add a text box with details of the training and testing
textstr = '\n'.join((
    rf'$\mathrm{{Epochs}}={epochs}$',
    rf'$\mathrm{{RMSE}}={np.sqrt(np.mean((predictions - y_test) ** 2)):.4f}$',
))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
axs[1,0].text(0.05, 0.95, textstr, transform=axs[1,0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

epochs = epochs * 2
predictions = test_seq2seq_model_learning_with_random_data_teacher_forcing(
    X_train=X_train_with_lstm_features,
    y_train=y_train_scaled,
    X_test=X_test_with_lstm_features,
    y_test=y_test_scaled,
    n_steps_in=n_steps_in,
    n_steps_out=n_steps_out,
    n_features=n_features_with_lstm,
    neurons=neurons,
    dropout=dropout,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size
)

# Plot the input, expected output, and predictions for the first test sample
axs[1,1].plot(range(n_steps_in), X_test[0].flatten(), label="Input (X_test)", marker='o')
axs[1,1].plot(range(n_steps_in - 1, n_steps_in + n_steps_out), 
            [X_test[0, -1, 0]] + y_test[0].flatten().tolist(), 
            label="Expected Output (y_test)", marker='o')
axs[1,1].plot(range(n_steps_in - 1, n_steps_in + n_steps_out), 
            [X_test[0, -1, 0]] + predictions[0].flatten().tolist(), 
            label="Predicted Output", marker='o')
axs[1,1].set_title("Seq2Seq Model Predictions with Random Data (Teacher Forcing)")
axs[1,1].set_xlabel("Timesteps")
axs[1,1].set_ylabel("Values")
axs[1,1].legend()
axs[1,1].grid()

#add a text box with details of the training and testing
textstr = '\n'.join((
    rf'$\mathrm{{Epochs}}={epochs}$',
    rf'$\mathrm{{RMSE}}={np.sqrt(np.mean((predictions - y_test) ** 2)):.4f}$',
))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
axs[1,1].text(0.05, 0.95, textstr, transform=axs[1,1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

