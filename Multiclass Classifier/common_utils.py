import pandas as pd
import numpy as np
import os
import shutil
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from sklearn.metrics import classification_report, confusion_matrix

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Input(shape=(4,)))
	model.add(Dense(8,activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def load_csv_data(path_and_file,seed):
	# load dataset
	dataframe = pd.read_csv(path_and_file, header=None)
	dataset = dataframe.values
	X = dataset[:,0:4].astype(float)
	Y = dataset[:,4]
	uniques = np.unique(Y)

	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = to_categorical(encoded_Y)

	X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1, random_state=seed)

	return X_train, X_test, Y_train, Y_test, X, uniques, Y, dummy_y, encoder

def print_prediction_summary(predictions, uniques, X_test, Y_test, encoder):
	print(predictions)
	print(uniques[predictions.argmax(1)])
	#capture the actual values from Y_test into an array of strings
	Y_test_uniques = uniques[Y_test.argmax(1)]
	pred_values = uniques[predictions.argmax(1)]

	#compare the actual values with the predicted values. Add a column that shows result of comparison
	#if the actual value is equal to the predicted value, then the result is True, otherwise it is False
	#print(np.column_stack((Y_test_uniques, pred_values, Y_test_uniques == pred_values)))
	#print total number of rows in the array
	print("Total number of rows in test set:",len(Y_test))
	print("Total number of correct predictions:",sum(Y_test_uniques == pred_values))

	#show the rows with incorrect predictions
	#fetch the incorrectly predicted rows from X_test and show actual and predicted values
	incorrect_rows = X_test[Y_test_uniques != pred_values]
	#print incorrect_rows as well as the actual and predicted values
	print(np.column_stack((X_test[Y_test_uniques != pred_values], Y_test_uniques[Y_test_uniques != pred_values], pred_values[Y_test_uniques != pred_values])))

	#show the classification report
	print("Classification Report:")
	print(classification_report(encoder.inverse_transform(Y_test.argmax(1)),encoder.inverse_transform(predictions.argmax(1))))
	print("Confusion Matrix:")
	print(confusion_matrix(encoder.inverse_transform(Y_test.argmax(1)),encoder.inverse_transform(predictions.argmax(1))
		,labels=uniques))
	
# write a function to save the model
def save_model(model, logical_name):
	#generate a model name based using a sequence number such that the format of the name is:
	# logical_name_yyyymmdd.nnn.keras, where nnn is a 3-digit sequence number. The sequence number should be incremented
	# each time a model is saved with the same logical name
	#save the model to the models directory

	#obtain sequence number by checking the names of the existing models
	#extract the sequence number from the existing model names
	#increment the sequence number by 1

	for file in os.listdir("models"):
		if file.startswith(logical_name):			
			match = re.search(r"\d{3}", file)
			if match:
				sequence_number = int(match.group())
			else:
				sequence_number = 0
			break
		else:
			sequence_number = 0
	
	sequence_number += 1

	model_name = logical_name + "-"+datetime.datetime.now().strftime("%Y%m%d") + "-"+str(sequence_number).zfill(3) + ".keras"

	#move existing models to a backup directory
	if not os.path.exists("models/backups"):
		os.makedirs("models/backups")
	#move existing models to the backup directory
	for file in os.listdir("models"):
		if file.endswith(".keras"):
			shutil.move("models/"+file, "models/backups/"+file)	

	# save the model along with model history
	model_history = model.history.history
	model_name = "model_with_history.h5"
	model.save("models/" + model_name)
	np.save("models/" + model_name + "_history.npy", model_history)

	print("Saving model as "+model_name)
	model.save("models/"+model_name)

# write a function to load the model
def load_saved_model(logical_name, sequence_number):
	#load the model from the models directory. Fetch the model with the highest sequence number or the specified sequence number
	#return the model
	if sequence_number == None:
		sequence_number = "001"

	#load the model from the models directory
	for file in os.listdir("models"):
		if file.startswith(logical_name):
			match = re.search(r"\d{3}", file)
			if match:
				sequence_number = int(match.group())
				if file[-10:-8] == sequence_number:
					model = load_model("models/" + file)
					return model
			model = load_model("models/"+file)
			return model

def load_or_train_model():
		print("Do you want to train a new model or load an existing model?")
		print("1. Train a new model")
		print("2. Load an existing model")
		print("3. Exit")
		#loop until user enters a valid choice
		while True:
			choice = input("Enter your choice: ")
			if choice in ['1','2','3']:
				break
			else:
				print("Invalid choice. Please enter a valid choice")
		return choice

def show_model_metrics(model, X_test, Y_test, uniques, encoder):

	#check if the model has a history attribute
	if hasattr(model, 'history'):
		#extract the loss and accuracy from the model history
		losses = pd.DataFrame(model.history.history)
			
		#plot loss and accuracy vs epochs
		#epochs are represented by the index of the dataframe

		plt.figure(figsize=(15, 6))

		plt.subplot(1, 2, 1)
		plt.plot(losses.index, losses['loss'])
		#plt.axhline(y=np.mean(losses['loss']), color='red', linestyle='--')
		plt.title('Loss vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')

		plt.subplot(1, 2, 2)
		plt.plot(losses.index, losses['accuracy'])
		#plt.axhline(y=np.mean(losses['accuracy']), color='red', linestyle='--')
		plt.title('Accuracy vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')

		plt.tight_layout()
		plt.show()

	print(model.metrics_names)

    # evaluate the model
	scores = model.evaluate(X_test, Y_test, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))