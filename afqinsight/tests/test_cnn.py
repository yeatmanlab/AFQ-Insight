import pytest
import os.path as op
# Bring your packages onto the path

# Now do your import
import afqinsight as afqi
from afqinsight.cnn import CNN
from afqinsight.datasets import load_afq_data

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")

def test_basic_cnn():
	X, y, groups, feature_names, group_names, subjects, classes = load_afq_data(
		workdir=test_data_path,
		target_cols=["test_class"],
		label_encode_cols=["test_class"],
	)
	model = CNN(100, 6, 5)
	model.fit(X,y)
	assert model.is_fitted_ is True
	y_hat = model.predict(X)
	score = model.score(y, y_hat)

def test_hyperband_cnn():
	X, y, groups, feature_names, group_names, subjects, classes = load_afq_data(
		workdir=test_data_path,
		target_cols=["test_class"],
		label_encode_cols=["test_class"],
	)

	model = CNN(100, 6, 5, "hyperband")
	model.fit(X,y)
	assert model.is_fitted_ is True
	y_hat = model.predict(X)
	score = model.score(y, y_hat)

	model2 = CNN(100, 6, 5, "hyperband", 4)
	model2.fit(X,y)
	assert model2.is_fitted_ is True
	y_hat2 = model2.predict(X)
	score2 = model2.score(y, y_hat2)

	model3 = CNN(100, 6, 5, "hyperband", 4, 0.3)
	model3.fit(X, y)
	assert model3.is_fitted_ is True
	y_hat3 = model3.predict(X)
	score3 = model3.score(y, y_hat3)

	model4 = CNN(100, 6, 5, "hyperband", 4, 0.3, factor=2, hyperband_iterations=2, seed=2)
	model4.fit(X, y)
	assert model4.is_fitted_ is True
	y_hat4 = model4.predict(X)
	score4 = model4.score(y, y_hat4)

def test_bayesian_cnn():
	X, y, groups, feature_names, group_names, subjects, classes = load_afq_data(
		workdir=test_data_path,
		target_cols=["test_class"],
		label_encode_cols=["test_class"],
	)

	model = CNN(100, 6, 5, "bayesian")
	model.fit(X,y)
	assert model.is_fitted_ is True
	y_hat = model.predict(X)
	score = model.score(y, y_hat)

	model2 = CNN(100, 6, 5, "bayesian", 4)
	model2.fit(X,y)
	assert model2.is_fitted_ is True
	y_hat2 = model2.predict(X)
	score2 = model2.score(y, y_hat2)

	model3 = CNN(100, 6, 5, "bayesian", 4, 0.3)
	model3.fit(X, y)
	assert model3.is_fitted_ is True
	y_hat3 = model3.predict(X)
	score3 = model3.score(y, y_hat3)

	model4 = CNN(100, 6, 5, "bayesian", 4, 0.3, num_initial_points=2, alpha=.02, beta=.5, seed=5)
	model4.fit(X, y)
	assert model4.is_fitted_ is True
	y_hat4 = model4.predict(X)
	score4 = model4.score(y, y_hat4)

def test_random_cnn():
	X, y, groups, feature_names, group_names, subjects, classes = load_afq_data(
		workdir=test_data_path,
		target_cols=["test_class"],
		label_encode_cols=["test_class"],
	)

	model = CNN(100, 6, 5, "random")
	model.fit(X,y)
	assert model.is_fitted_ is True
	y_hat = model.predict(X)
	score = model.score(y, y_hat)

	model2 = CNN(100, 6, 5, "random", 4)
	model2.fit(X,y)
	assert model2.is_fitted_ is True
	y_hat2 = model2.predict(X)
	score2 = model2.score(y, y_hat2)

	model3 = CNN(100, 6, 5, "random", 4, 0.3)
	model3.fit(X, y)
	assert model3.is_fitted_ is True
	y_hat3 = model3.predict(X)
	score3 = model3.score(y, y_hat3)

	model4 = CNN(100, 6, 5, "random", 4, 0.3, seed=5)
	model4.fit(X, y)
	assert model4.is_fitted_ is True
	y_hat4 = model4.predict(X)
	score4 = model4.score(y, y_hat4)

def test_fail_cnn():
	with pytest.raises(ValueError):
		model = CNN(100, 6, 5, "wrong")

	with pytest.raises(TypeError):
		model = CNN(100, 6, 5, 0)

	with pytest.raises(TypeError):

		model = CNN(100, 6, 5, wrong)

	with pytest.raises(TypeError):
		model = CNN(100, 6, 5, 0.0)

