import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, MaxPooling1D, Dropout
from keras.regularizers import l1_l2, l2
from keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer
import kerastuner as kt

class ModelBuilder:
	"""
	This class controls the buidling of complex model architecture and the number of layers in the model.

	"""

	def __init__(self, objective, max_epochs,):
		pass

	def


class CNN:
	"""
	This class implements the common sklearn model interface (has a fit and predict function).

	A random forest is a collection of decision trees that are trained on random subsets of the
	dataset. When predicting the value for an example, takes a majority vote from the trees.
	"""

	def __init__(self, pixels, channels, cnn_type="1D"):
		"""
		Constructs a CNN that uses the given number of pixels, each with a
		max depth of max_depth.
		"""
		self.pixels = pixels
		self.channels = channels
		self.cnn_type = cnn_type

	def __preprocess(self, X, y):
		nan_mask = np.logical_not(np.isnan(y))
		y = y[nan_mask]
		X = X[nan_mask, :]

		X_train, X_test, y_train, y_test = train_test_split(X, y)

		imp = SimpleImputer(strategy='median')
		imp.fit(X_train)
		X_train = imp.transform(X_train)
		X_test = imp.transform(X_test)

		if self.cnn_type == "1D"
			train_subjects = X_train.shape[0]
			test_subjects = X_test.shape[0]
			X_train = np.swapaxes(X_train.reshape((train_subjects, self.pixels, self.channels)), 1, 2)
			X_test = np.swapaxes(X_test.reshape((test_subjects, self.pixels, self.channels)), 1, 2)

		return X_train, X_test

	def fit(self, X, y, epochs, validation_split):
		"""
		Takes an input dataset X and a series of targets y and trains the CNN.
		"""


	def predict(self, X):
		"""
		Takes an input dataset X and returns the predictions for each example in X.
		"""
