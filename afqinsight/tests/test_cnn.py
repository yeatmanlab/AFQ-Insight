import afqinsight as afqi
import os.path as op
import pytest
import tempfile

from afqinsight.cnn import CNN
from afqinsight.datasets import load_afq_data

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")

X, y, groups, feature_names, group_names, subjects, _, _ = load_afq_data(
    fn_nodes=op.join(test_data_path, "nodes.csv"),
    fn_subjects=op.join(test_data_path, "subjects.csv"),
    target_cols=["test_class"],
    label_encode_cols=["test_class"],
)


def test_basic_cnn():
    with tempfile.TemporaryDirectory() as tdir:
        model = CNN(100, 6, 5, project_name="test-project", directory=tdir)
        model.fit(X, y)
        assert model.is_fitted_ is True
        y_hat = model.predict(X)
        _ = model.score(y, y_hat)


def test_hyperband_cnn():
    with tempfile.TemporaryDirectory() as tdir:
        model = CNN(
            100, 6, 5, 64, "hyperband", project_name="test-project", directory=tdir
        )
        model.fit(X, y)
        assert model.is_fitted_ is True
        y_hat = model.predict(X)
        _ = model.score(y, y_hat)

        model2 = CNN(
            100, 6, 5, 64, "hyperband", 4, project_name="test-project", directory=tdir
        )
        model2.fit(X, y)
        assert model2.is_fitted_ is True
        y_hat2 = model2.predict(X)
        _ = model2.score(y, y_hat2)

        model3 = CNN(
            100,
            6,
            5,
            64,
            "hyperband",
            4,
            0.3,
            project_name="test-project",
            directory=tdir,
        )
        model3.fit(X, y)
        assert model3.is_fitted_ is True
        y_hat3 = model3.predict(X)
        _ = model3.score(y, y_hat3)

        model4 = CNN(
            100,
            6,
            5,
            64,
            "hyperband",
            4,
            0.3,
            factor=2,
            hyperband_iterations=2,
            seed=2,
            project_name="test-project",
            directory=tdir,
        )
        model4.fit(X, y)
        assert model4.is_fitted_ is True
        y_hat4 = model4.predict(X)
        _ = model4.score(y, y_hat4)


def test_bayesian_cnn():
    with tempfile.TemporaryDirectory() as tdir:
        model = CNN(100, 6, 5, 64, "bayesian", directory=tdir)
        model.fit(X, y)
        assert model.is_fitted_ is True
        y_hat = model.predict(X)
        _ = model.score(y, y_hat)

        model2 = CNN(100, 6, 5, 64, "bayesian", 4, directory=tdir)
        model2.fit(X, y)
        assert model2.is_fitted_ is True
        y_hat2 = model2.predict(X)
        _ = model2.score(y, y_hat2)

        model3 = CNN(100, 6, 5, 64, "bayesian", 4, 0.3, directory=tdir)
        model3.fit(X, y)
        assert model3.is_fitted_ is True
        y_hat3 = model3.predict(X)
        _ = model3.score(y, y_hat3)

        model4 = CNN(
            100,
            6,
            5,
            64,
            "bayesian",
            4,
            0.3,
            num_initial_points=2,
            alpha=0.02,
            beta=0.5,
            seed=5,
            directory=tdir,
        )
        model4.fit(X, y)
        assert model4.is_fitted_ is True
        y_hat4 = model4.predict(X)
        _ = model4.score(y, y_hat4)


def test_random_cnn():
    with tempfile.TemporaryDirectory() as tdir:
        model = CNN(100, 6, 5, 64, "random", directory=tdir)
        model.fit(X, y)
        assert model.is_fitted_ is True
        y_hat = model.predict(X)
        _ = model.score(y, y_hat)

        model2 = CNN(100, 6, 5, 64, "random", 4, directory=tdir)
        model2.fit(X, y)
        assert model2.is_fitted_ is True
        y_hat2 = model2.predict(X)
        _ = model2.score(y, y_hat2)

        model3 = CNN(100, 6, 5, 64, "random", 4, 0.3, directory=tdir)
        model3.fit(X, y)
        assert model3.is_fitted_ is True
        y_hat3 = model3.predict(X)
        _ = model3.score(y, y_hat3)

        model4 = CNN(
            100, 6, 5, 64, "random", 4, 0.3, impute_strategy="mean", directory=tdir
        )
        model4.fit(X, y)
        assert model4.is_fitted_ is True
        y_hat4 = model4.predict(X)
        _ = model4.score(y, y_hat4)


def test_fail_cnn():

    with pytest.raises(ValueError):
        # passing in wrong shape of X (not 2d):
        model = CNN(100, 6, 5, 64)
        model.fit(X.reshape((7, 100, -1)), y)

    with pytest.raises(ValueError):
        # passing in wrong tuner value
        model = CNN(100, 6, 5, 64, "wrong")
        model.fit(X, y)

    with pytest.raises(TypeError):
        # passing in int for tuner
        model = CNN(100, 6, 5, 64, 0)

    with pytest.raises(ValueError):
        # passing in n_nodes and n_channels that multiply to equal
        # proper dimension for given x
        model = CNN(78, 6, 5, 64, "random")
        model.fit(X, y)

    with pytest.raises(TypeError):
        # passing in float for tuner_type
        model = CNN(100, 6, 5, 64, 0.0)

    with pytest.raises(TypeError):
        # passing in float for n_nodes
        model = CNN(1.1, 6, 5, 64, "random")

    with pytest.raises(TypeError):
        # passing in float for n_channels
        model = CNN(100, 6.0, 5, 64, "random")

    with pytest.raises(TypeError):
        # passing in float for layers
        model = CNN(100, 6, layers=5.0)

    with pytest.raises(TypeError):
        # passing in float for batch size
        model = CNN(100, 6, 5, 6.4, "random")

    with pytest.raises(TypeError):
        # passing in string for batch size
        model = CNN(100, 6, 5, "64", "random")

    with pytest.raises(TypeError):
        # passing in an integer for test_size
        model = CNN(100, 6, test_size=20)

    with pytest.raises(TypeError):
        # passing in an integer for impute_strategy (this should be a string).
        model = CNN(100, 6, impute_strategy=20)

    with pytest.raises(ValueError):
        # passing in the wrong string for impute_strategy:
        model = CNN(100, 6, impute_strategy="foo")

    with pytest.raises(TypeError):
        # passing in a string for random_state (should be int or RandomState).
        model = CNN(100, 6, random_state="foo")
