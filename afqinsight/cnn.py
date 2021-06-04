import numpy as np
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.impute import SimpleImputer
import kerastuner as kt
import functools
import tempfile
import os.path as op


def build_model(hp, conv_layers, input_shape):
	"""
	Uses keras tuner to build model - can control # layers, # filters in each layer, kernel size,
	regularization etc

	"""
	model = Sequential()

	dense_filters = hp.Int('dense_filters', min_value=32, max_value=512, step=32)
	model.add(Dense(dense_filters, activation='relu', input_shape=input_shape))

	for i in range(conv_layers):
		model.add(Conv1D(filters=hp.Int('filters' + str(i), min_value=32, max_value=512, step=32),
		                 kernel_size=hp.Int('kernel' + str(i), min_value=1, max_value=4, step=1),
		                 activation='relu'))

		model.add(MaxPool1D(pool_size=2, padding='same'))

		model.add(Dropout(0.25))

	model.add(Flatten())

	dense_filters_2 = hp.Int('dense_filters_2', min_value=32, max_value=512, step=32)
	model.add(Dense(dense_filters_2, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
	return model


class ModelBuilder:
	"""
	This class controls the building of complex model architecture and the number of layers in the model.
	"""

	def __init__(self, class_type, input_shape, layers, max_epochs, val_split, **tuner_kwargs):
		self.class_type = class_type
		self.layers = layers
		self.input_shape = input_shape
		self.max_epochs = max_epochs
		self.val_split = val_split
		self.tuner_kwargs = tuner_kwargs

	def _get_tuner(self):
		# setting parameters beforehand
		hypermodel = functools.partial(build_model, conv_layers=self.layers, input_shape=self.input_shape)
		if isinstance(self.class_type, str):
			# instantiating tuner based on user's choice
			if self.class_type == "hyperband":
				tuner = kt.Hyperband(hypermodel=hypermodel,
				                     objective='mean_squared_error',
				                     max_epochs=self.max_epochs,
				                     overwrite=True,
				                     **self.tuner_kwargs)

			elif self.class_type == "bayesian":
				tuner = kt.BayesianOptimization(hypermodel=hypermodel,
				                                objective='mean_squared_error',
				                                max_trials=self.max_epochs,
				                                overwrite=True,
				                                **self.tuner_kwargs)

			elif self.class_type == "random":
				tuner = kt.RandomSearch(hypermodel=hypermodel,
				                        objective='mean_squared_error',
				                        max_trials=self.max_epochs,
				                        overwrite=True,
				                        **self.tuner_kwargs)
			else:
				raise ValueError("tuner parameter expects 'hyperband', 'bayesian', or 'random'")
			return tuner
		else:
			raise TypeError()

	def _get_best_weights(self, model, X, y):
		weights_path = op.join(tempfile.mkdtemp(), 'weights.hdf5')
		# making model checkpoint to save best model (# epochs) to file
		model_checkpoint_callback = ModelCheckpoint(
			filepath=weights_path,
			monitor='val_loss',
			mode='auto',
			save_best_only=True,
			save_weights_only=True,
			verbose=True)

		# Fitting model using model checkpoint callback to find best model which is saved to 'weights'
		model.fit(X, y, epochs=self.max_epochs, callbacks=[model_checkpoint_callback], validation_split=self.val_split)

		# loading in weights
		model.load_weights(weights_path)

		# return the model
		return model

	# TODO: Ask if I should have tuning on this
	def build_basic_model(self, X, y):
		model = Sequential()
		model.add(Dense(128, activation='relu', input_shape=X.shape[1:]))
		model.add(Conv1D(24, kernel_size=2, activation='relu'))
		model.add(MaxPool1D(pool_size=2, padding='same'))
		model.add(Conv1D(32, kernel_size=2, activation='relu'))
		model.add(MaxPool1D(pool_size=2, padding='same'))
		model.add(Conv1D(64, kernel_size=3, activation='relu'))
		model.add(MaxPool1D(pool_size=2, padding='same'))
		model.add(Conv1D(128, kernel_size=4, activation='relu'))
		model.add(MaxPool1D(pool_size=2, padding='same'))
		model.add(Conv1D(256, kernel_size=4, activation='relu'))
		model.add(MaxPool1D(pool_size=2, padding='same'))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.25))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(1, activation='linear'))

		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

		best_model = self._get_best_weights(model, X, y)
		return best_model

	def build_tuned_model(self, X, y):
		# initialize tuner
		tuner = self._get_tuner()

		# Find the optimal hyperparameters
		tuner.search(X, y, epochs=50, validation_split=self.val_split)

		# Save the optimal hyperparameters
		best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

		# make CNN model using best hyperparameters
		model = tuner.hypermodel.build(best_hps)

		best_model = self._get_best_weights(model, X, y)
		return best_model


class CNN:
	"""
	This class implements the common sklearn model interface (has a fit and predict function).
	"""

	def __init__(self, nodes, channels, max_epochs, tuner=None, layers=1, val_split=0.2, **tuner_kwargs):
		"""
		Constructs a CNN that uses the given number of nodes, each with a
		max depth of max_depth.
		"""
		# checking nodes is passed as int
		if not type(nodes) is int:
			raise TypeError("Parameter nodes must be an integer.")
		else:
			self.nodes = nodes

		# checking channels is passed as int
		if not type(channels) is int:
			raise TypeError("Parameter channels must be an integer.")
		else:
			self.channels = channels

		# checking layers is passed as int
		if not type(layers) is int:
			raise TypeError("Parameter layers must be an integer.")
		else:
			self.layers = layers

		# checking max epochs is passed as int
		if not type(max_epochs) is int:
			raise TypeError("Parameter max_epochs must be an integer.")
		else:
			self.max_epochs = max_epochs

		# checking tiner is passed as str or None
		if type(tuner) is not str and tuner is not None:
			raise TypeError("Parameter tuner must be str.")
		else:
			self.tuner = tuner  # tuner can be None (no tuning) BayesianOptimization, Hyperband, or RandomSearch

		# checking val split is passed as float
		if not type(val_split) is float:
			raise TypeError("Parameter val_split must be a float.")
		else:
			self.val_split = val_split
		self.tuner_kwargs = tuner_kwargs
		self.model_ = None
		self.best_hps_ = None

	def _preprocess(self, X, y=None):
		if len(X.shape) > 2:
			raise ValueError("Expected X to be a 2D matrix.")
		if y is not None:
			nan_mask = np.logical_not(np.isnan(y))
			X = X[nan_mask, :]
			y = y[nan_mask]

		imp = SimpleImputer(strategy='median')
		imp.fit(X)
		X = imp.transform(X)

		if y is not None:
			X, y = check_X_y(X, y)

		subjects = X.shape[0]

		# nodes * channels must = X.shape[1]
		if self.nodes * self.channels != X.shape[1]:
			raise ValueError("The product nodes and channels is not the correct shape.")
		# error
		else:
			X = np.swapaxes(X.reshape((subjects, self.channels, self.nodes)), 1, 2)

		if y is not None:
			return X, y
		else:
			return X

	def fit(self, X, y):
		"""
		Takes an input dataset X and a series of targets y and trains the CNN.
		"""

		X, y = self._preprocess(X, y)

		# CNN gets nodes, channels, max_epochs, tuner=None, layers=None
		# Model Builder takes class_type, input_shape, layers, max_epochs, **kwargs
		builder = ModelBuilder(self.tuner, X.shape[1:], self.layers, self.max_epochs, self.val_split, **self.tuner_kwargs)
		if self.tuner is None:
			self.model_ = builder.build_basic_model(X, y)
		else:
			self.model_ = builder.build_tuned_model(X, y)

		self.is_fitted_ = True

		return self

	def predict(self, X):
		"""
		Takes an input dataset X and returns the predictions for each example in X.
		"""
		X = self._preprocess(X)
		check_is_fitted(self, 'is_fitted_')
		pred = self.model_.predict(X)
		return pred

	def score(self, y_test, y_hat):
		nan_mask = np.logical_not(np.isnan(y_test))
		y_test = y_test[nan_mask]
		return r2_score(y_test, y_hat)
