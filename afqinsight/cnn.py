import numpy as np
from sklearn.utils.validation import (check_X_y, check_is_fitted)
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, MaxPooling1D, Dropout
from keras.regularizers import l1_l2, l2
from keras.callbacks import ModelCheckpoint
from sklearn.impute import SimpleImputer
import kerastuner as kt


def model_builder(hp, conv_layers, input_shape):
	"""
	Uses keras tuner to build model - can control # layers, # filters in each layer, kernel size,
	regularization etc

	"""
	model = Sequential()

	filters1 = hp.Int('filters1', min_value=32, max_value=512, step=32)
	model.add(Dense(filters1, activation='relu', input_shape=input_shape))

	for i in range(conv_layers):
		filters2 = hp.Int('filters2', min_value=32, max_value=512, step=32)
		kernel1 = hp.Int('kernel1', min_value=1, max_value=4, step=1)
		model.add(Conv1D(filters2, kernel_size=kernel1, activation='relu'))

		model.add(MaxPool1D(pool_size=2, padding='same'))

		model.add(Dropout(0.25))


	model.add(Flatten())
	filters7 = hp.Int('filters7', min_value=32, max_value=512, step=32)
	model.add(Dense(filters7, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
	return model

class ModelBuilder:
	"""
	This class controls the building of complex model architecture and the number of layers in the model.

	"""

	def __init__(self, class_type, layers, max_epochs):
		self.class_type = class_type
		self.layers = layers
		self.max_epochs = max_epochs

	def _get_tuner(self):
		if self.class_type == "Hyperband":
			tuner = kt.Hyperband(model_builder, objective='mean_squared_error', max_epochs=self.max_epochs, overwrite=True)
		elif self.class_type == "BayesianOptimization":
			tuner = kt.BayesianOptimization(model_builder, objective='mean_squared_error', max_trials=self.max_epochs, overwrite=True)
		elif self.class_type == "RandomSearch":
			tuner = kt.RandomSearch(model_builder, objective='mean_squared_error', max_trials=self.max_epochs, overwrite=True)
		return tuner

	def fit(self, X_train, y_train):
		# initialize tuner
		tuner = self._get_tuner()

		# Find the optimal hyperparameters
		tuner.search(X_train, y_train, epochs=50, validation_split=0.2)

		# Save the optimal hyperparameters
		best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

		# make CNN model using best hyperparameters
		model = tuner.hypermodel.build(best_hps)

		# return the model
		return model



class CNN:
	"""
	This class implements the common sklearn model interface (has a fit and predict function).

	A random forest is a collection of decision trees that are trained on random subsets of the
	dataset. When predicting the value for an example, takes a majority vote from the trees.
	"""

	# TODO:
	#   .fit()
	#   .predict()
	#   .score()
	#   .set_params()
	#   .get_params()

	def __init__(self, pixels, channels, tuner="Hyperband"):
		"""
		Constructs a CNN that uses the given number of pixels, each with a
		max depth of max_depth.
		"""
		self.pixels = pixels
		self.channels = channels
		self.tuner = tuner # tuner can be BayesianOptimization, Hyperband, or RandomSearch
		self.model = None
		self.best_hps_ = None

	def _preprocess(self, X, y):
		nan_mask = np.logical_not(np.isnan(y))
		y = y[nan_mask]
		X = X[nan_mask, :]

		imp = SimpleImputer(strategy='median')
		imp.fit(X)
		X = imp.transform(X)

		subjects = X.shape[0]
		# pixels * channels must = X.shape[1]
		X = np.swapaxes(X.reshape((subjects, self.channels, self.nodes)), 1, 2)

		return X, y


	def fit(self, X, y, fit_epochs, objective, builder_epochs=5, validation_split=0.2):
		"""
		Takes an input dataset X and a series of targets y and trains the CNN.
		"""

		X, y = self.__preprocess(X, y)

		self.model = ModelBuilder(self.tuner, X.shape[1:])

		# Build the model with the optimal hyperparameters and train it on the data for 50 epochs, find
		# best # of epochs to get min mse
		#history = self.model.fit(X, y)
		#objective_per_epoch = history.history[objective]

		# TODO: change min to max depending on the objective chosen
		#best_epoch = objective_per_epoch.index(min(objective_per_epoch)) + 1

		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			monitor='mean_squared_error',
			mode='min',
			save_best_only=True)

		# Retrain the model with the best epoch
		self.model.fit(X, y, )
		self.is_fitted_ = True

		return self


	def predict(self, X):
		"""
		Takes an input dataset X and returns the predictions for each example in X.
		"""
		check_is_fitted(self, 'is_fitted_')
		pred = self.model.predict(X)
		return pred
