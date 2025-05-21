import pandas as pd
import numpy as np
from numpy import array

from utils import (create_overlapping_arrays, validate_input_data, 
                   convert_to_dataframes,generate_sequences, convert_dataframe_to_3d_array)

def test_create_overlapping_arrays():
    # Sample data
    data = pd.DataFrame({
        'Feature1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'Feature2': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        'Tgt': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    }, index=pd.date_range(start='2023-01-01', periods=10))

    # Parameters
    feature_columns = ['Feature1', 'Feature2']
    target_column = 'Tgt'
    n_steps_in = 1
    n_steps_out = 1

    # Step 1: Validate input data
    try:
        validate_input_data(data, feature_columns, target_column, n_steps_in, n_steps_out)
        print("Input data validation passed.")
    except ValueError as e:
        print(f"Input data validation failed: {e}")
        return

    # Step 2: Generate overlapping sequences
    try:
        X_encoder, decoder_input, y_decoder, X_encoder_dates, decoder_input_dates, y_decoder_dates = generate_sequences(
            data, feature_columns, target_column, n_steps_in, n_steps_out
        )
        print("\nGenerated Sequences:")
        print(f"X_encoder: {len(X_encoder)} sequences")
        print(f"decoder_input: {len(decoder_input)} sequences")
        print(f"y_decoder: {len(y_decoder)} sequences")
        print(f"X_encoder_dates: {len(X_encoder_dates)} date lists")
        print(f"decoder_input_dates: {len(decoder_input_dates)} date lists")
        print(f"y_decoder_dates: {len(y_decoder_dates)} date lists")
    except Exception as e:
        print(f"Error during sequence generation: {e}")
        return

    # Step 3: Convert sequences to DataFrames
    try:
        X_encoder_df, decoder_input_df, y_decoder_df = convert_to_dataframes(
            X_encoder, decoder_input, y_decoder,
            X_encoder_dates, decoder_input_dates, y_decoder_dates,
            feature_columns, target_column
        )
        print("\nGenerated DataFrames:")
        print(f"X_encoder_df shape: {X_encoder_df.shape}")
        print(f"decoder_input_df shape: {decoder_input_df.shape}")
        print(f"y_decoder_df shape: {y_decoder_df.shape}")
    except Exception as e:
        print(f"Error during DataFrame conversion: {e}")
        return

    # Step 4: Validate the shapes of the outputs
    expected_samples = len(data) - n_steps_in - n_steps_out + 1  # Corrected formula
    try:
        assert X_encoder_df.shape[0] == expected_samples, "Mismatch in number of samples for X_encoder_df"
        assert decoder_input_df.shape[0] == expected_samples, "Mismatch in number of samples for decoder_input_df"
        assert y_decoder_df.shape[0] == expected_samples, "Mismatch in number of samples for y_decoder_df"
        print("\nShape validation passed.")
    except AssertionError as e:
        print(f"Shape validation failed: {e}")
        return

    print("\nTest passed: create_overlapping_arrays works as expected.")

# Run the test
# test_create_overlapping_arrays()

def test_generate_sequences_edge_case():
    # Small dataset with exactly n_steps_in + n_steps_out rows
    data = pd.DataFrame({
        'Feature1': [0.1, 0.2, 0.3],
        'Feature2': [1.1, 1.2, 1.3],
        'Tgt': [2.1, 2.2, 2.3]
    }, index=pd.date_range(start='2023-01-01', periods=3))

    # Parameters
    feature_columns = ['Feature1', 'Feature2']
    target_column = 'Tgt'
    n_steps_in = 2
    n_steps_out = 1

    # Generate sequences
    try:
        X_encoder, decoder_input, y_decoder, X_encoder_dates, decoder_input_dates, y_decoder_dates = generate_sequences(
            data, feature_columns, target_column, n_steps_in, n_steps_out
        )
        print("\nGenerated Sequences for Edge Case:")
        print(f"X_encoder: {len(X_encoder)} sequences")
        print(f"decoder_input: {len(decoder_input)} sequences")
        print(f"y_decoder: {len(y_decoder)} sequences")
        print(f"X_encoder_dates: {len(X_encoder_dates)} date lists")
        print(f"decoder_input_dates: {len(decoder_input_dates)} date lists")
        print(f"y_decoder_dates: {len(y_decoder_dates)} date lists")

        # Validate the number of sequences
        expected_samples = len(data) - n_steps_in - n_steps_out + 1
        assert len(X_encoder) == expected_samples, "Mismatch in number of X_encoder sequences"
        assert len(decoder_input) == expected_samples, "Mismatch in number of decoder_input sequences"
        assert len(y_decoder) == expected_samples, "Mismatch in number of y_decoder sequences"

        # Validate the shape of each sequence
        for seq in X_encoder:
            assert seq.shape == (n_steps_in, len(feature_columns)), "Incorrect shape for X_encoder sequence"
        for seq in decoder_input:
            assert seq.shape == (n_steps_out, len(feature_columns)), "Incorrect shape for decoder_input sequence"
        for seq in y_decoder:
            assert seq.shape == (n_steps_out, 1), "Incorrect shape for y_decoder sequence"

        # Validate the number of unique dates in decoder_input_dates and y_decoder_dates
        assert len(decoder_input_dates) == expected_samples, \
            f"Mismatch in number of decoder_input_dates. Expected {expected_samples}, got {len(decoder_input_dates)}"
        assert len(y_decoder_dates) == expected_samples, \
            f"Mismatch in number of y_decoder_dates. Expected {expected_samples}, got {len(y_decoder_dates)}"

        print("\nTest passed: generate_sequences works correctly for edge case.")
    except Exception as e:
        print(f"Error during sequence generation for edge case: {e}")

# test_generate_sequences_edge_case()

def test_generate_sequences():
    # Small dataset with more rows to test n_steps_out > 1
    data = pd.DataFrame({
        'Feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Feature2': [1.1, 1.2, 1.3, 1.4, 1.5],
        'Tgt': [2.1, 2.2, 2.3, 2.4, 2.5]
    }, index=pd.date_range(start='2023-01-01', periods=5))

    # Parameters
    feature_columns = ['Feature1', 'Feature2']
    target_column = 'Tgt'
    n_steps_in = 3
    n_steps_out = 2

    # Generate sequences
    X_encoder, decoder_input, y_decoder, X_encoder_dates, decoder_input_dates, y_decoder_dates = generate_sequences(
        data, feature_columns, target_column, n_steps_in, n_steps_out
    )

    # Validate the number of sequences
    expected_samples = len(data) - n_steps_in - n_steps_out + 1
    assert len(X_encoder) == expected_samples, "Mismatch in number of X_encoder sequences"
    assert len(decoder_input) == expected_samples, "Mismatch in number of decoder_input sequences"
    assert len(y_decoder) == expected_samples, "Mismatch in number of y_decoder sequences"

    # Validate the shape of each sequence
    for seq in X_encoder:
        assert seq.shape == (n_steps_in, len(feature_columns)), "Incorrect shape for X_encoder sequence"
    for seq in decoder_input:
        assert seq.shape == (n_steps_out, len(feature_columns)), "Incorrect shape for decoder_input sequence"
    for seq in y_decoder:
        assert seq.shape == (n_steps_out, 1), "Incorrect shape for y_decoder sequence"

    print("Test passed: generate_sequences works correctly.")

# Run the test
# test_generate_sequences()

def test_generate_sequences_large_dataframe():
    # Larger dataset to test n_steps_in = 2 and n_steps_out = 1
    data = pd.DataFrame({
        'Feature1': np.linspace(0.1, 10.0, 20),  # 20 rows of data
        'Feature2': np.linspace(1.1, 20.0, 20),
        'Tgt': np.linspace(2.1, 30.0, 20)
    }, index=pd.date_range(start='2023-01-01', periods=20))

    # Parameters
    feature_columns = ['Feature1', 'Feature2']
    target_column = 'Tgt'
    n_steps_in = 5
    n_steps_out = 1 

    # Generate sequences
    try:
        X_encoder, decoder_input, y_decoder, X_encoder_dates, decoder_input_dates, y_decoder_dates = generate_sequences(
            data, feature_columns, target_column, n_steps_in, n_steps_out
        )
        print("\nGenerated Sequences for Large DataFrame:")
        print(f"X_encoder: {len(X_encoder)} sequences")
        print(f"decoder_input: {len(decoder_input)} sequences")
        print(f"y_decoder: {len(y_decoder)} sequences")
        print(f"X_encoder_dates: {len(X_encoder_dates)} date lists")
        print(f"decoder_input_dates: {len(decoder_input_dates)} date lists")
        print(f"y_decoder_dates: {len(y_decoder_dates)} date lists")

        # Validate the number of sequences
        expected_samples = len(data) - n_steps_in - n_steps_out + 1
        assert len(X_encoder) == expected_samples, "Mismatch in number of X_encoder sequences"
        assert len(decoder_input) == expected_samples, "Mismatch in number of decoder_input sequences"
        assert len(y_decoder) == expected_samples, "Mismatch in number of y_decoder sequences"

        # Validate the shape of each sequence
        for seq in X_encoder:
            assert seq.shape == (n_steps_in, len(feature_columns)), "Incorrect shape for X_encoder sequence"
        for seq in decoder_input:
            assert seq.shape == (n_steps_out, len(feature_columns)), "Incorrect shape for decoder_input sequence"
        for seq in y_decoder:
            assert seq.shape == (n_steps_out, 1), "Incorrect shape for y_decoder sequence"

        # Convert sequences to DataFrames
        X_encoder_df, decoder_input_df, y_decoder_df = convert_to_dataframes(
            X_encoder, decoder_input, y_decoder,
            X_encoder_dates, decoder_input_dates, y_decoder_dates,
            feature_columns, target_column
        )
        print("\nGenerated DataFrames:")
        print(f"X_encoder_df shape: {X_encoder_df.shape}")
        print(f"decoder_input_df shape: {decoder_input_df.shape}")
        print(f"y_decoder_df shape: {y_decoder_df.shape}")

       # Validate the number of sequences
        assert len(X_encoder) == expected_samples, \
            f"Mismatch in number of sequences. Expected {expected_samples}, got {len(X_encoder)}"
        assert len(decoder_input) == expected_samples, \
            f"Mismatch in number of decoder_input sequences. Expected {expected_samples}, got {len(decoder_input)}"
        assert len(y_decoder) == expected_samples, \
            f"Mismatch in number of y_decoder sequences. Expected {expected_samples}, got {len(y_decoder)}"

        # Validate the number of rows in X_encoder_df
        expected_rows = expected_samples * n_steps_in
        assert X_encoder_df.shape[0] == expected_rows, \
            f"Mismatch in number of rows for X_encoder_df. Expected {expected_rows}, got {X_encoder_df.shape[0]}"

        # Validate the number of unique dates in X_encoder_df
        unique_dates = len(X_encoder_df.index.get_level_values('Date').unique())
        assert unique_dates <= len(data), \
            f"Mismatch in unique dates: X_encoder_df has {unique_dates}, but original dataset has {len(data)}"

        # Validate the number of sequences
        assert len(X_encoder) == expected_samples, \
            f"Mismatch in number of sequences. Expected {expected_samples}, got {len(X_encoder)}"

        # Validate the number of rows in decoder_input_df and y_decoder_df
        assert decoder_input_df.shape[0] == expected_samples, \
            f"Mismatch in number of rows for decoder_input_df. Expected {expected_samples}, got {decoder_input_df.shape[0]}"
        assert y_decoder_df.shape[0] == expected_samples, \
            f"Mismatch in number of rows for y_decoder_df. Expected {expected_samples}, got {y_decoder_df.shape[0]}"

        print(f"\nTest passed: generate_sequences and convert_to_dataframes work correctly for large DataFrame with n_steps_in = {n_steps_in} and n_steps_out = {n_steps_out}.")
    except Exception as e:
        print(f"Error during sequence generation or DataFrame conversion for large DataFrame: {e}")

# test_generate_sequences_large_dataframe()

def test_convert_dataframe_to_3d_array():
    # Create a simple dataset with MultiIndex (Date, Step)
    data = {
        "Feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19],
        "Feature3": [21, 22, 23, 24, 25, 26, 27, 28, 29],
    }
    index = pd.MultiIndex.from_tuples(
        [("2023-01-01", i) for i in range(3)] + [("2023-01-02", i) for i in range(6)],
        names=["Date", "Step"],
    )
    df = pd.DataFrame(data, index=index)

    # Parameters for the function
    n_timesteps = 5
    n_features = 3  # Update to match the number of features in the DataFrame

    # Debugging: Print the input DataFrame
    print("Input DataFrame:")
    print(df)

    # Call the function
    try:
        array_3d = convert_dataframe_to_3d_array(df, n_timesteps, n_features)

        # Debugging: Print the resulting 3D array
        print("\nGenerated 3D Array:")
        print(array_3d)

        # Expected output
        expected_array = np.array([
            [
                [1, 11, 21],
                [2, 12, 22],
                [3, 13, 23],
                [0, 0, 0],  # Padded rows
                [0, 0, 0],
            ],
            [
                [4, 14, 24],
                [5, 15, 25],
                [6, 16, 26],
                [7, 17, 27],
                [8, 18, 28],
            ],
        ])

        # Assertions
        assert array_3d.shape == (2, 5, 3), f"Expected shape (2, 5, 3), but got {array_3d.shape}"
        assert np.array_equal(array_3d, expected_array), f"Expected array:\n{expected_array}\nBut got:\n{array_3d}"

        print("\nTest passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")

test_convert_dataframe_to_3d_array()

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

