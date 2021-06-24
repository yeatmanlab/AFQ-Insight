import numpy as np
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import functools
import tempfile
import os.path as op

try:
	import kerastuner as kt
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, MaxPooling1D, Dropout
	from tensorflow.keras.callbacks import ModelCheckpoint

    HAS_KERAS = True
except ImportError:  # pragma: no cover
    HAS_KERAS = False


def build_model(hp, conv_layers, input_shape):
	"""
	build_model(hp, conv_layers, input_shape)

	Uses keras tuner to build model - can control # layers, # filters in each layer, kernel size,
	regularization etc

	Parameters
	----------
	hp : tensorflow.keras.HyperParameters()
		Hyperparameters class from which to sample hyperparameters

	conv_layers : int
		number of layers (one layer is Conv and MaxPool) in the sequential model.

	input_shape : int
		input shape of X so the model gets built continuously as you are adding layers

	Returns
	-------
	model : tensorflow.keras.Model
		compiled model that uses hyperparameters defined inline to hypertune the model

	"""
	model = Sequential()
	model.add(Conv1D(filters=hp.Int('init_conv_filters' + str(i), min_value=32, max_value=512, step=32),
	                 kernel_size=hp.Int('init_conv_kernel' + str(i), min_value=1, max_value=4, step=1),
	                 activation='relu', input_shape=input_shape ))

	for i in range(conv_layers - 1):
		model.add(Conv1D(filters=hp.Int('conv_filters' + str(i), min_value=32, max_value=512, step=32),
		                 kernel_size=hp.Int('conv_kernel' + str(i), min_value=1, max_value=4, step=1),
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

def _check_keras():
	if not HAS_KERAS:
		raise ImportError(
			"To use afqinsight's convolutional neural nets for tractometry data, you will need "
			"to have tensorflow, keras, and kerastuner installed. You can do this by installing "
			"afqinsight with `pip install afqinsight[cnn]`, or by separately installing these packages "
			"with `pip install tensorflow keras kerastuner`."
		)

class ModelBuilder:
	"""
	This class controls the building of complex model architecture and the number of layers in the model.
	"""

	def __init__(self, class_type, input_shape, layers, max_epochs, X_test, y_test, batch_size, val_split,
	             **tuner_kwargs):
		self.class_type = class_type
		self.layers = layers
		self.input_shape = input_shape
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.val_split = val_split
		self.X_test = X_test
		self.y_test = y_test
		self.tuner_kwargs = tuner_kwargs

	def _get_tuner(self):
		"""
		_get_tuner()

		Calls build_model and instantiates a Keras Tuner
		for the returned model depending on user choice of tuner.

		Returns
		-------
		tuner : kerastuner.tuners
			BayesianOptimization, Hyperband, or RandomSearch tuner

		"""
		# setting parameters beforehand
		hypermodel = functools.partial(build_model, conv_layers=self.layers, input_shape=self.input_shape)
		if isinstance(self.class_type, str):
			# instantiating tuner based on user's choice
			if self.class_type == "hyperband":
				tuner = kt.Hyperband(hypermodel=hypermodel,
				                     objective='mean_squared_error',
				                     max_epochs=10,
				                     overwrite=True,
				                     **self.tuner_kwargs)

			elif self.class_type == "bayesian":
				tuner = kt.BayesianOptimization(hypermodel=hypermodel,
				                                objective='mean_squared_error',
				                                max_trials=10,
				                                overwrite=True,
				                                **self.tuner_kwargs)

			elif self.class_type == "random":
				tuner = kt.RandomSearch(hypermodel=hypermodel,
				                        objective='mean_squared_error',
				                        max_trials=10,
				                        overwrite=True,
				                        **self.tuner_kwargs)
			else:
				raise ValueError("tuner parameter expects 'hyperband', 'bayesian', or 'random'")
			return tuner
		else:
			raise TypeError()

	def _get_best_weights(self, model, X, y):
		"""
		_get_best_weights(model, X, y)

		Uses keras ModelCheckpoint to fit CNN and save the weights from the epoch that produced
		the lowest validation loss to a temporary file. Uses temporary file to load the
		best weights into the CNN model and returns this best model.

		Parameters
		----------
		model : tensorflow.keras.Sequential()
			Hyperparameters class from which to sample hyperparameters

		X : array-like of shape (n_samples, n_features)
			The feature samples

		y : array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values

		Returns
		-------
		model : tensorflow.keras.Model
			fitted keras model with best weights loaded

		"""
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
		# model.fit(X, y, epochs=self.max_epochs, batch_size=self.batch_size, callbacks=[model_checkpoint_callback], validation_split=self.val_split)
		model.fit(X, y, epochs=self.max_epochs, batch_size=self.batch_size, callbacks=[model_checkpoint_callback],
		          validation_data=(self.X_test, self.y_test))
		# loading in weights
		model.load_weights(weights_path)

		# return the model
		return model

	def build_basic_model(self, X, y):
		"""
		build_basic_model(X, y)

		Builds a static, basic sequential model with no hyperparameter tuning with the
		architecture used to produce state of the art Weston Havens results.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The feature samples

		y : array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values

		Returns
		-------
		model : tensorflow.keras.Model
			compiled model using basic Weston Havens architecture

		"""
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
		"""
		build_tuned_model(X, y)

		Initializes a Keras tuner on user's model, searches for best hyperparameters, and saves them.
		Then builds "best" model using saved best hyperparameters found during the search and returns model
		with best weights loaded from _get_best_weights.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The feature samples

		y : array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values

		Returns
		-------
		model : tensorflow.keras.Model
			compiled model that uses hyperparameters defined inline to hypertune the model

		"""
		# initialize tuner
		tuner = self._get_tuner()

		# Find the optimal hyperparameters
		tuner.search(X, y, epochs=50, validation_split=0.2)

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

	def __init__(self, nodes, channels, max_epochs=50, batch_size=32, tuner=None, layers=1, val_split=0.2,
	             random_state=200, **tuner_kwargs):
		"""
		Constructs a CNN that uses the given number of nodes, each with a
		max depth of max_depth.
		"""
		# checking nodes is passed as int
		if not isinstance(nodes, int):
			raise TypeError("Parameter nodes must be an integer.")
		else:
			self.nodes = nodes

		# checking channels is passed as int
		if not isinstance(channels, int):
			raise TypeError("Parameter channels must be an integer.")
		else:
			self.channels = channels

		# checking layers is passed as int
		if not isinstance(layers, int):
			raise TypeError("Parameter layers must be an integer.")
		else:
			self.layers = layers

		# checking max epochs is passed as int
		if not isinstance(max_epochs, int):
			raise TypeError("Parameter max_epochs must be an integer.")
		else:
			self.max_epochs = max_epochs

		if not isinstance(batch_size, int):
			raise TypeError("Parameter max_epochs must be an integer.")
		else:
			self.batch_size = batch_size

		# checking tiner is passed as str or None
		if not isinstance(tuner, str) and tuner is not None:
			raise TypeError("Parameter tuner must be str.")
		else:
			self.tuner = tuner  # tuner can be None (no tuning) BayesianOptimization, Hyperband, or RandomSearch

		# checking val split is passed as float
		if not isinstance(val_split, float):
			raise TypeError("Parameter val_split must be a float.")
		else:
			self.val_split = val_split

		if not isinstance(random_state, int):
			raise TypeError("Parameter random_state must be an int.")
		else:
			self.random_state = random_state

		self.tuner_kwargs = tuner_kwargs
		self.model_ = None
		self.best_hps_ = None

	def _preprocess(self, X, y=None):
		"""
		_preprocess(X, y)

		Masks NAN values for X and y (if y is given), imputes X, and reshapes X
		to be in proper form for CNN model.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The feature samples

		y : array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values

		Returns
		-------
		X : array-like of shape (n_samples, n_features)
			The feature samples

		y : array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values

		"""
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

		X = np.swapaxes(X.reshape((subjects, self.channels, self.nodes)), 1, 2)

		if y is not None:
			return X, y
		else:
			return X

	def fit(self, X, y):
		"""
		fit(X, y)

		Preprocesses X and y, builds CNN model, tunes model hyperparameters and
		fits the model to given X and y, using X_test and y_test to validate and
		find best weights and hyperparameters.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The feature samples

		y : array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values

		X_test : array-like of shape (n_samples, n_features)
			The feature test samples

		y_test : array-like of shape (n_samples,) or (n_samples, n_targets)
			Target test values

		Returns
		-------
		self : CNN
			updated CNN instantiation

		"""
		# nodes * channels must = X.shape[1]
		if self.nodes * self.channels != X.shape[1]:
			raise ValueError("The product nodes and channels is not the correct shape.")
		# error
		X, y = self._preprocess(X, y)
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)
		# CNN gets nodes, channels, max_epochs, tuner=None, layers=None
		# Model Builder takes class_type, input_shape, layers, max_epochs, **kwargs
		builder = ModelBuilder(self.tuner, X_train.shape[1:], self.layers, self.max_epochs, X_test, y_test,
		                       self.batch_size, self.val_split, **self.tuner_kwargs)
		if self.tuner is None:
			self.model_ = builder.build_basic_model(X_train, y_train)
		else:
			self.model_ = builder.build_tuned_model(X_train, y_train)

		self.is_fitted_ = True

		return self

	def predict(self, X):
		"""
		predict(X)

		Preprocesses X and returns predicted y values for x from fitted CNN model.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The feature samples

		Returns
		-------
		pred : array-like of shape (n_samples,) or (n_samples, n_targets)
			predicted values

		"""

		X = self._preprocess(X)
		check_is_fitted(self, 'is_fitted_')
		pred = self.model_.predict(X).squeeze()
		return pred

	def score(self, y_test, y_hat):
		"""
		predict(X)

		Applies a NAN mask to y_test and returns r-squared score for the CNN model.

		Parameters
		----------
		y_test : array-like of shape (n_samples,) or (n_samples, n_targets)
			Testing target values

		y_hat : array-like of shape (n_samples,) or (n_samples, n_targets)
			Predicted target values

		Returns
		-------
		r2_score : float
			r-squared score for y_test and y_hat for CNN model

		"""
		nan_mask = np.logical_not(np.isnan(y_test))
		y_test = y_test[nan_mask]
		return r2_score(y_test, y_hat)
