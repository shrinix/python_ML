# multi-class classification with Keras
import numpy
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import random as tf_random
from common_utils import load_csv_data,baseline_model,print_prediction_summary,load_or_train_model,save_model,load_model
	
def train_model(model, X, dummy_y, X_train, X_test, Y_train, Y_test,  seed):
	estimator = KerasClassifier(model=model, epochs=200, batch_size=5, verbose=0)
	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(estimator, X, dummy_y, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	estimator.fit(X_train, Y_train)
	return model
	
def main():
	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)
	tf_random.set_seed(seed)

	X_train, X_test, Y_train, Y_test, X, uniques, Y, dummy_y, encoder = load_csv_data("data/iris.data.csv",seed)

	choice = load_or_train_model()
	if choice == '1':
		model = baseline_model()
		model = train_model(model, X, dummy_y, X_train, X_test, Y_train, Y_test,  seed)
		save_model(model)
	elif choice == '2':
		model = load_model()
	else:
		print("Exiting the program")
		return
	
	predictions = model.predict(X_test)
	print_prediction_summary(predictions, uniques, X_test, Y_test, encoder)

# add invocation to main function
if __name__ == "__main__":
	main()

