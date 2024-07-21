# mlp for multi-label classification
import numpy as np #requires numy version 1.26.4
import numpy as np
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold,RepeatedStratifiedKFold
from tensorflow.keras.models import Sequential, Model
from keras import layers
from tensorflow.keras.layers import Dense, Input, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import del_col, del_duplicates, impute_col, typecast_col, convert_case, label_encoder
from imblearn.over_sampling import SMOTE,SMOTENC,SVMSMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix,classification_report
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier

#TODO: winequalityN.csv

def convert_categories_to_one_hot(X, ignore_labels=False):
	X = pd.concat([X,pd.get_dummies(X['condition'], prefix='condition')],axis=1)
	# X = pd.concat([X,pd.get_dummies(X['X1'], prefix='X1')],axis=1)
	# X = pd.concat([X,pd.get_dummies(X['X2'], prefix='X2')],axis=1)
	X = pd.concat([X,pd.get_dummies(X['color_type'], prefix='color_type')],axis=1)
	X = pd.concat([X,pd.get_dummies(X['length_label'], prefix='length_label')],axis=1)
	X = pd.concat([X,pd.get_dummies(X['height_label'], prefix='height_label')],axis=1)
	X = del_col('condition',X)
	X = del_col('color_type',X)
	# X = del_col('X1',X)
	# X = del_col('X2',X)
	X = del_col('length(cm)',X)
	X = del_col('length_label',X)
	X = del_col('height_label',X)
	
	if (ignore_labels):
		return X
	
	pet_category_columns = pd.get_dummies(X['pet_category'], prefix='pet_category',dtype='int')
	X = pd.concat([X,pet_category_columns],axis=1)
	#get number of columns created for pet_category
	num_pet_category_cols = pet_category_columns.shape[1]
	breed_category_columns = pd.get_dummies(X['breed_category'], prefix='breed_category',dtype='int')
	X = pd.concat([X,breed_category_columns],axis=1)
	#get number of columns created for pet_category
	num_breed_category_cols = breed_category_columns.shape[1]

	X = del_col('pet_category',X)
	X = del_col('breed_category',X)

	return X, num_pet_category_cols, num_breed_category_cols

def resample_data(X,Y):
	
	smote = SMOTE()
	X_smote,y_smote_1 = smote.fit_resample(X,Y[0])

	from collections import Counter
	print("Before SMOTE :", Counter(Y[0].argmax(axis=-1)))
	print("After SMOTE :", Counter(y_smote_1.argmax(axis=-1)))

	X_smote,y_smote_2 = smote.fit_resample(X,Y[1])
	print("Before SMOTE :", Counter(Y[1].argmax(axis=-1)))
	print("After SMOTE :", Counter(y_smote_2.argmax(axis=-1)))

	Y_smote = (y_smote_1, y_smote_2)

	#get unique values in Y[0]
	print(np.unique(y_smote_1, return_counts=True))
	print(np.unique(y_smote_2, return_counts=True))

	return X_smote,Y_smote

def get_dataset(file_path, ignore_labels=False):
	
	#X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)

	data=pd.read_csv(file_path)
	#print(str(data.shape))	
		
	#drop issue_date and listing date columns
	data.drop(['pet_id'],axis=1,inplace=True)

	#converting both the columns into datetime format
	data['issue_date']=pd.to_datetime(data['issue_date'])
	data['listing_date']=pd.to_datetime(data['listing_date'])
	
	# Generating new feature 
	data['duration'] = (data['listing_date'] - data['issue_date']).dt.days
	data = del_col('issue_date',data)
	data = del_col('listing_date',data)

	#Impute missing values
	#TODO: Try KNN Imputer
	data=data.fillna(3.0)
	
	# Standardization - converting m to cm
	data['length(cm)'] = data['length(m)'].apply(lambda x: x*100)
	data = del_col('length(m)',data)
	# replace all 0 length with mean of lengths
	val = data['length(cm)'].mean()
	data['length(cm)'] = data['length(cm)'].replace(to_replace=0, value=val)

	#Quantile based binning for continuous variable length(cm)
	quantile_list = [0, .25, .5, .75, 1.]
	quantiles = data['length(cm)'].quantile(quantile_list)

	quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
	data['length_label'] = pd.qcut(data['length(cm)'],q=quantile_list, labels=quantile_labels)

	#Quantile based binning for continuous variable height(m)
	quantile_list = [0, .25, .5, .75, 1.]
	quantiles = data['height(cm)'].quantile(quantile_list)

	quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
	data['height_label'] = pd.qcut(data['height(cm)'],q=quantile_list, labels=quantile_labels)

	#Custom binning for continuous variable like duration
	data['duration'] = abs(data['duration'])
	data['duration'] =np.array(np.array(data['duration']) / 365.)

	# names = data.columns
	# scaler = preprocessing.StandardScaler()
	# scaled_df = scaler.fit_transform(data)
	# data = pd.DataFrame(scaled_df, columns=names)

	if (not ignore_labels):
		# data['breed_category'] = data['breed_category'].astype('int')
		# data['pet_category'] = data['pet_category'].astype('int')
		data, num_pet_category_cols, num_breed_category_cols = convert_categories_to_one_hot(data,ignore_labels)

		Y = get_labels(data) #Also drops columns 'breed_category', 'pet_category' from data

		#print(data.info())
		data, Y = resample_data(data,Y)
		return data, Y, num_pet_category_cols, num_breed_category_cols
	else:
		data = convert_categories_to_one_hot(data,ignore_labels)
		# print(data.info())
		#data,Y = resample_data(data,Y)
		return data

# Normalize the data, but with the mean and std of only train data. 
def scale_data(train_stats, df):
	return (df - train_stats['mean']) / train_stats['std']

#With this function we get the labels *pet_category* and *breed_category* to pass to the Model. 
def get_labels(df):

	#pop the columns with names starting with 'breed_category' and 'pet_category' from the dataframe
	#and convert them to numpy arrays
	pet_category_columns = [col for col in df.columns if 'pet_category' in col]
	#loop through the columns and pop them from the dataframe
	num_pet_category_cols = len(pet_category_columns)
	pet_category_df = pd.DataFrame()
	#concatenate the column values into a dataframe
	pet_category_df = pd.concat([pd.DataFrame(df.pop(col)) for col in pet_category_columns], axis=1)
	#convert the dataframe to a numpy array
	pet_category = np.array(pet_category_df).reshape(-1, num_pet_category_cols)
	print(f"pet_category shape: {pet_category.shape}")

	breed_category_columns = [col for col in df.columns if 'breed_category' in col]	
	num_breed_category_cols = len(breed_category_columns)
	breed_category_df = pd.DataFrame()
	#concatenate the column values into a dataframe
	breed_category_df = pd.concat([pd.DataFrame(df.pop(col)) for col in breed_category_columns], axis=1)
	#convert the dataframe to a numpy array
	breed_category = np.array(breed_category_df).reshape(-1, num_breed_category_cols)
	print(f"breed_category shape: {breed_category.shape}")

	return (pet_category,breed_category)

# get the model
def get_model(n_inputs, n_pet_category_outputs, n_breed_category_outputs):
	
	#Start with the input layer, where we must indicate the shape of the Data passed to the model. 
	inputs = tf.keras.layers.Input(shape=(n_inputs,))

	#Add dense layers to the input layer. These layers are commom to both predicted variables. 
	dense = Dense(units=64, activation='relu')(inputs)
	
	#Add dropout layer to prevent overfitting
	dropout = Dropout(0.2)(dense)
	
	#Add the output layer for the pet_category output using Sigmoid activation. 
	pet_category_layer = Dense(units=n_pet_category_outputs, activation='softmax', name='pet_category_layer')(dropout)

	#Add the output layer for the breed_category output using Sigmoid activation. 
	breed_category_layer = Dense(units=n_breed_category_outputs, activation='softmax', name='breed_category_layer')(dropout)

	model = Model(inputs=inputs, outputs=[pet_category_layer, breed_category_layer])
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	
	#loss funnction should be binary_crossentropy because we use sigmoid activation function in output layer
	model.compile( optimizer=optimizer,
				loss={'pet_category_layer':'categorical_crossentropy',
		  			'breed_category_layer':'categorical_crossentropy'},
					metrics= {'pet_category_layer' : 'accuracy', 
						 'breed_category_layer': 'accuracy'
					   }) 
	tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='./model_plot.png')

	#print(model.summary())
	return model

def get_training_data_after_split(X,y,t_id,v_id):

	tx = X.iloc[t_id]
	vx = X.iloc[v_id]

	#Extract rows corresponding to t_id from each of the two Series in tuple y
	ty_extracted_1 = y[0][t_id]
	ty_extracted_2 = y[1][t_id]

	# Combine extracted Series back into a tuple ty
	ty = (ty_extracted_1, ty_extracted_2)

	vy_extracted_1 = y[0][v_id]
	vy_extracted_2 = y[1][v_id]
	vy = (vy_extracted_1, vy_extracted_2)

	return tx, ty, vx, vy
# Define the custom callback
class MetricsPrintCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            print(f"Metrics at the end of epoch {epoch}:")
            for metric, value in logs.items():
               print(f"{metric}: {value:.2f}", end=" ")
            print("\n")

# evaluate a model using repeated k-fold cross-validation
def train_model(file_and_path):

	# load dataset
	X,y,num_pet_category_cols, num_breed_category_cols  = get_dataset(file_and_path, False)
	print(f"Loaded test data with input shape {X.shape} from file: {file_and_path}")
	print(f"Number of pet_category columns: {num_pet_category_cols} and breed_category columns: {num_breed_category_cols}")

	# define model
	model = get_model(X.shape[1],num_pet_category_cols, num_breed_category_cols)

	kf = RepeatedStratifiedKFold(n_splits=7,n_repeats=1, random_state=300)
	f1_scores = [], []
	# enumerate folds
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

	for fold,(t_id,v_id) in enumerate(kf.split(X,np.zeros(shape=(X.shape[0], 1)))):

		print(f"Before split\n"
			f"Train data shape: {X.shape}\n"
			f"Train label shape - pet category: {y[0].shape} - breed category: {y[1].shape}\n")

		tx,ty,vx,vy = get_training_data_after_split(X,y,t_id,v_id)
		print(f"After fold {fold} split\n"
			f"Train data shape: {tx.shape}\n"
			f"Train label shape - pet category: {ty[0].shape} - breed category: {ty[1].shape}\n"
			f"Validation data shape: {vx.shape}\n"
			f"Validation label shape - pet category: {vy[0].shape}- breed category: {vy[1].shape}\n")

		# fit model
		history=model.fit(tx, ty, epochs=10, verbose=0, validation_data=(vx, vy), callbacks=[es,MetricsPrintCallback()])
		plt.plot(history.history['pet_category_layer_accuracy'])
		plt.plot(history.history['breed_category_layer_accuracy'])
		plt.title(model.name+' accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		# plt.show()
		
		# evaluate the model
		val_y = model.predict(vx, verbose=0)
		pet_category_pred = val_y[0].argmax(axis=-1)
		pet_category_val = vy[0].argmax(axis=-1)
		F1_score_1 = f1_score(pet_category_pred, pet_category_val, average='weighted')
		f1_scores[0].append(F1_score_1)

		breed_category_pred = val_y[1].argmax(axis=-1)
		breed_category_val = vy[1].argmax(axis=-1)
		F1_score_2 = f1_score(breed_category_pred, breed_category_val, average='weighted')
		f1_scores[1].append(F1_score_2)
		print(f"f1 pet category {F1_score_1} breed_category {F1_score_2}")
		print("Confusion matrix for pet_category:")
		print(confusion_matrix(pet_category_pred, pet_category_val))
		print("Confusion matrix for breed_category:")
		print(confusion_matrix(breed_category_pred, breed_category_val))
	
	#save the plot to a file
	plt.savefig(model.name+'_accuracy.png')

	#plt.show()
	#save the model
	model_file_and_path = './multi_output_pet_classification.keras'
	model.save(model_file_and_path)
	print(f"Model saved as {model_file_and_path}")

	f1_score_pet = np.mean(f1_scores[0])
	f1_score_breed = np.mean(f1_scores[1])
	print(f"Mean f1 score for pet_category: {f1_score_pet}")
	print(f"Mean f1 score for breed_category: {f1_score_breed}")

	return

def predict(file_and_path):
	
	vx = get_dataset(file_and_path, True)
	print(f"Loaded test data with shape {vx.shape} from file: {file_and_path}")

	# load model
	model = tf.keras.Sequential([
	    tf.keras.models.load_model('./multi_output_pet_classification.keras')
	])
	print(f"Loaded model: {model.name})")
	
	#print([layer.name for layer in first_model.layers])

	print(vx.shape)
	val_y = model.predict(vx)
	pet_category_pred = val_y[0]
	breed_category_pred = val_y[1]
	#save the predictions to a file. format the values as integers
	pet_category_pred = pet_category_pred.argmax(axis=-1)
	breed_category_pred = breed_category_pred.argmax(axis=-1)
	
	np.savetxt('./pet_category_predictions.csv', pet_category_pred, delimiter=',', fmt='%d')
	print("Pet category predictions saved to pet_category_predictions.csv")
	np.savetxt('./breed_category_predictions.csv', breed_category_pred, delimiter=',',fmt='%d')
	print("Breed category predictions saved to breed_category_predictions.csv")
	print(pet_category_pred)
	print(breed_category_pred)

def main():

	#present the user with a menu to choose the operation - whether to train the model or to predict
	#the output for the test data. 
	print("Choose the operation to perform:")
	print("1. Train the model")
	print("2. Predict the output for the test data")
	print("3. Exit")
	operation = input("Enter the operation number: ")

	#loop through the menu until the user chooses to exit
	while operation not in ['1','2','3']:
		print("Invalid operation. Please enter a valid operation number.")
		operation = input("Enter the operation number: ")

	if operation == '1':
		file_and_path = './pet_data_train.csv'
		train_model(file_and_path)
	elif operation == '2':
		file_and_path = './pet_data_test.csv'
		predict(file_and_path)
	elif operation == '3':
		exit()

if __name__ == "__main__":
	main()