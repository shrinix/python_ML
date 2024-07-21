import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode the target variable
encoder = LabelEncoder()
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from common_utils import load_csv_data,baseline_model,print_prediction_summary,show_model_metrics,load_or_train_model,save_model,load_saved_model
from sklearn.model_selection import KFold

def main():

    seed = 7
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logical_name = "iris_classifier"

    X_train, X_test, Y_train, Y_test, X, uniques, Y, dummy_y, encoder = load_csv_data("data/iris.data.csv",seed)

    choice = load_or_train_model()
    if choice == '1':
        model = baseline_model()
        # Define the number of folds for KFold
        n_splits = 5

        # Initialize KFold with the specified number of folds
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Initialize an empty list to store the model's performance metrics for each fold
        fold_metrics = []

        # Iterate over each fold
        for train_index, test_index in kf.split(X):
            # Split the data into train and test sets for the current fold
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            # Ensure that 'x' and 'y' arrays have the same number of samples
            assert len(X_train) == len(Y_train), "Data cardinality mismatch: 'x' and 'y' arrays have different number of samples."

            # Create and compile the model
            model = baseline_model()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            Y_train = encoder.fit_transform(Y_train)
            Y_test = encoder.fit_transform(Y_test)
            
            # Convert the target array to one-hot encoded format
            Y_train = to_categorical(Y_train)
            Y_test = to_categorical(Y_test)

            # Reshape the target array to have the same shape as the output array
            Y_train = Y_train.reshape(-1, 3)
            Y_test = Y_test.reshape(-1, 3)

            # Fit the model on the current fold's train data
            model.fit(X_train, Y_train, epochs=200, batch_size=5, verbose=1)

            # Evaluate the model on the current fold's test data
            fold_metrics.append(model.evaluate(X_test, Y_test, verbose=0))

        # Calculate the average performance metrics across all folds
        average_metrics = np.mean(fold_metrics, axis=0)
        # Get the loss and accuracy history for plots
        loss_history = []
        accuracy_history = []

        for train_index, test_index in kf.split(X):
            # Split the data into train and test sets for the current fold
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            # Ensure that 'x' and 'y' arrays have the same number of samples
            assert len(X_train) == len(Y_train), "Data cardinality mismatch: 'x' and 'y' arrays have different number of samples."
            # Create and compile the model
            model = baseline_model()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            Y_train = encoder.fit_transform(Y_train)
            Y_test = encoder.fit_transform(Y_test)
            
            # Convert the target array to one-hot encoded format
            Y_train = to_categorical(Y_train)
            Y_test = to_categorical(Y_test)
            # Reshape the target array to have the same shape as the output array
            Y_train = Y_train.reshape(-1, 3)
            Y_test = Y_test.reshape(-1, 3)
            # Fit the model on the current fold's train data
            history = model.fit(X_train, Y_train, epochs=200, batch_size=5, verbose=1)
            
            # Append the loss and accuracy to the history lists
            loss_history.append(history.history['loss'])
            accuracy_history.append(history.history['accuracy'])

        # Print the average performance metrics
        print("Average Performance Metrics:")
        print("Loss: {:.4f}".format(average_metrics[0]))
        print("Accuracy: {:.4f}".format(average_metrics[1]))


        # Save the model
        save_model(model,logical_name)

        # # Plot the loss history
        # import matplotlib.pyplot as plt
        # plt.plot(np.mean(loss_history, axis=0))
        # plt.title('Loss History')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.show()

        # # Plot the accuracy history
        # plt.plot(np.mean(accuracy_history, axis=0))
        # plt.title('Accuracy History')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.show()
    elif choice == '2':
        model = load_saved_model(logical_name,None)
    else:
        print("Exiting the program")
        return

    show_model_metrics(model, X_test, Y_test, uniques, encoder)

    predictions = model.predict(X_test)
    print_prediction_summary(predictions, uniques, X_test, Y_test, encoder)

# add invocation to main function
if __name__ == "__main__":
	main()