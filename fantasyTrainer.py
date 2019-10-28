import tensorflow.keras as keras
from math import floor
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def main():
	#if 'Tot' Predict total fantasy points
	#if "A" gives the average
	prediction_type = 'avg'
	#season we will use to train model
	season_1 = '2017-18'
	#we will get training labels and testing data from here
	season_2 = '2018-19'


	print("Getting Stats")
	#training stats will be used for training the model, prediction stats are used for getting the raw fantasy score
	training_stats, training_labels = get_stats(season_1, season_2, prediction_type)
	#next we need to generate the actual total of fantasy points each player scored in the season we want to predict

	print('syncing stats')
	#now we need to sync the training labels with the stats as they come from two different years
	training_data, training_labels = sync_stats(training_stats, training_labels)

	print("Splitting Data for Testing Data")
	#splits the training data and training labels to get testing data and labels
	training_data, training_labels, testing_data, testing_labels = get_testing(training_data, training_labels)


	print('vectorizing data')
	training_data_arrays = vectorize_data(training_data)
	training_labels_arrays = vectorize_data(training_labels)
	testing_data_arrays = vectorize_data(testing_data)
	testing_labels_arrays = vectorize_data(testing_labels)

	print("Training model")
	model, history = learning(training_data_arrays, training_labels_arrays, testing_data_arrays, testing_labels_arrays)

	visualizeModel(model, history)


	save_model_and_weights(model, prediction_type)


	


def get_stats(season_1, season_2, prediction_type):
	


	training_stats = pd.read_csv(f'{season_1}_Player_stats.csv').set_index("PLAYER")
	prediction_stats = pd.read_csv(f'{season_2}_Player_stats.csv').set_index("PLAYER")
	training_labels = get_fantasy(prediction_stats, prediction_type)


	return training_stats, training_labels 



def get_fantasy(prediction_stats, prediction_type):
	'''Calculates the toal fantasy points each player earned over the course of the season'''
	#fantasy point conversions
	PTS = 1
	REB = 1.2
	AST = 1.5
	BLK = 3
	STL = 3
	TOV = -1
	#ending conversion table
	#choose whether you want to predict one's average stats per game, or total points for the season
	#choose 'A' for average and 'Tot' for Total
	
	#gives total fantasy points
	if prediction_type == 'tot':
		tot_fantasy_pts = prediction_stats["GP"] * ((prediction_stats['PTS'] * PTS) + (prediction_stats['REB'] * REB) + (prediction_stats['AST']*AST) + (prediction_stats['BLK'] * BLK) + (prediction_stats["STL"] * STL) + (prediction_stats['TOV'] * TOV))
		return tot_fantasy_pts

	else:
		avg_fantasy_pts = ((prediction_stats['PTS'] * PTS) + (prediction_stats['REB'] * REB) + (prediction_stats['AST']*AST) + (prediction_stats['BLK'] * BLK) + (prediction_stats["STL"] * STL) + (prediction_stats['TOV'] * TOV))
		return avg_fantasy_pts


	return fantasy_pts 

def sync_stats(training_stats, training_labels):
	print(training_stats)
	print(training_labels)
	training_stats, training_labels = training_stats.align(training_labels, axis = 0)



	training_stats['fantasy_points'] = training_labels

	training_stats_and_labels = training_stats.dropna()

	training_labels = training_stats_and_labels['fantasy_points']

	training_stats = training_stats_and_labels.drop(['fantasy_points', 'TEAM'], axis=1)


	return training_stats, training_labels

def get_testing(training_data, training_labels):
	"""splits the training data and labels to get testing data and labels"""

	cutoff_player = floor(len(training_data)*.3)
	testing_data = training_data[0:cutoff_player]
	testing_labels = training_labels[0:cutoff_player]

	training_data = training_data[cutoff_player:]
	training_labels = training_labels[cutoff_player:]

	return training_data, training_labels, testing_data, testing_labels



def vectorize_data(data):
	'''takes stats that contains both the training stats, as well as the training labels, will split them and a '''
	return np.array(data)


def learning(training_data_arrays, training_labels_arrays, testing_data_arrays, testing_labels_arrays):
	'''Predicts the fantasy scores for players in the following season'''
	print(training_data_arrays.shape, training_data_arrays.shape[0])

	model = keras.models.Sequential()
	#input layer
	model.add(keras.layers.Dense(128, kernel_initializer='normal', input_dim = training_data_arrays.shape[1], activation='relu'))
	#hidden layers
	model.add(keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
	model.add(keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
	model.add(keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
	#output layer
	model.add(keras.layers.Dense(1, kernel_initializer='normal',activation='linear'))
	#compile the neural net
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
	model.summary()

	history = model.fit(training_data_arrays, training_labels_arrays, epochs=1000, batch_size=32, validation_split=.2)
	score = model.evaluate(testing_data_arrays, testing_labels_arrays, verbose = 0)

	print('score', score)

	#creating a checkpoint call back which saves the neural networks weights if the program stops before it was supposed to
	# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
	# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
	# callbacks_list = [checkpoint]

	#Train this mother fucking model
	


	return model, history

	

def visualizeModel(model, history):
	# Plot training & validation accuracy values
	plt.plot(history.history['mean_squared_error'])
	# plt.plot(history.history['val_acc'])
	plt.title('Model MSE')
	plt.ylabel('MSE')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

def predicting_scores(model, training_data_arrays, testing_data_arrays, training_labels_arrays, testing_labels_arrays):
	predictions = model.predict(training_data_arrays)
	

def save_model_and_weights(model, prediction_type):
	# serialize model to JSON
	model_json = model.to_json()
	model.save(f'{prediction_type}_model.h5')
	# serialize weights to HDF5
	# model.save_weights(f"{prediction_type}_weights.h5")
	print("Saved model to disk")




if __name__ == '__main__':
	main()