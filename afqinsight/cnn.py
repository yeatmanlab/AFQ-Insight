import numpy as np
from sklearn.utils.validation import check_X_y, check_is_fitted
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, MaxPooling1D, Dropout
from keras.regularizers import l1_l2, l2
from keras.callbacks import ModelCheckpoint
from sklearn.impute import SimpleImputer
import kerastuner as kt
import functools
import tempfile


def build_model(hp, conv_layers, input_shape):
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

	def __init__(self, class_type, layers, input_shape, max_epochs):
		self.class_type = class_type
		self.layers = layers
		self.input_shape = input_shape
		self.max_epochs = max_epochs

	def _get_tuner(self):
		# setting parameters beforehand
		hypermodel = functools.partial(build_model, conv_layers=self.layers, input_shape=self.input_shape)

		# instantiating tuner based on user's choice
		if self.class_type == "Hyperband":
			tuner = kt.Hyperband(hypermodel=hypermodel, objective='mean_squared_error', max_epochs=self.max_epochs, overwrite=True)
		elif self.class_type == "BayesianOptimization":
			tuner = kt.BayesianOptimization(hypermodel=hypermodel, objective='mean_squared_error', max_trials=self.max_epochs, overwrite=True)
		elif self.class_type == "RandomSearch":
			tuner = kt.RandomSearch(hypermodel=hypermodel, objective='mean_squared_error', max_trials=self.max_epochs, overwrite=True)
		return tuner

	def find_best_hps(self, X, y):
		# initialize tuner
		tuner = self._get_tuner()

		# Find the optimal hyperparameters
		tuner.search(X, y, epochs=50, validation_split=0.2)

		# Save the optimal hyperparameters
		best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

		# make CNN model using best hyperparameters
		model = tuner.hypermodel.build(best_hps)

		# making temporary file
		# path = "weights.best.hdf5"
		weights = tempfile.mkstemp()

		# making model checkpoint to save best model (# epochs) to file
		model_checkpoint_callback = ModelCheckpoint(
			filepath=weights,
			monitor='val_accuracy',
			mode='max',
			save_best_only=True)

		# Fitting model using model checkpoint callback to find best model which is saved to 'weights'
		model.fit(X, y, epochs=self.max_epochs, callbacks=[model_checkpoint_callback])

		# loading in weights
		model.load_weights(weights)

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

	def __init__(self, nodes, channels, layers, max_epochs, tuner="Hyperband"):
		"""
		Constructs a CNN that uses the given number of nodes, each with a
		max depth of max_depth.
		"""
		self.nodes = nodes
		self.channels = channels
		self.layers = layers
		self.max_epochs = max_epochs
		self.tuner = tuner # tuner can be BayesianOptimization, Hyperband, or RandomSearch
		self.model = None
		self.best_hps_ = None

	def _preprocess(self, X, y):
		X, y = check_X_y(X, y)
		nan_mask = np.logical_not(np.isnan(y))
		y = y[nan_mask]
		X = X[nan_mask, :]

		imp = SimpleImputer(strategy='median')
		imp.fit(X)
		X = imp.transform(X)

		subjects = X.shape[0]
		# nodes * channels must = X.shape[1]
		if self.nodes * self.channels != X.shape[1]:
			pass
			# error
		else:
			X = np.swapaxes(X.reshape((subjects, self.channels, self.nodes)), 1, 2)

		return X, y


	def fit(self, X, y):
		"""
		Takes an input dataset X and a series of targets y and trains the CNN.
		"""

		X, y = self._preprocess(X, y)
		# class_type, layers, input_shape, max_epochs
		builder = ModelBuilder(self.tuner, self.layers, X.shape[1:])
		self.model = builder.find_best_hps(X, y)
		self.is_fitted_ = True

		return self


	def predict(self, X):
		"""
		Takes an input dataset X and returns the predictions for each example in X.
		"""
		check_is_fitted(self, 'is_fitted_')
		pred = self.model.predict(X)
		return pred
