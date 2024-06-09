import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from common_utils import load_csv_data,baseline_model,print_prediction_summary,show_model_metrics,load_or_train_model,save_model,load_model

def main():

    seed = 7
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, X_test, Y_train, Y_test, X, uniques, Y, dummy_y, encoder = load_csv_data("data/iris.data.csv",seed)

    model = baseline_model()
    model.fit(X_train, Y_train, epochs=200, batch_size=5, 
          validation_data=(X_test,Y_test),verbose=1)
    show_model_metrics(model, X_test, Y_test, uniques, encoder)

    predictions = model.predict(X_test)
    print_prediction_summary(predictions, uniques, X_test, Y_test, encoder)

# add invocation to main function
if __name__ == "__main__":
	main()