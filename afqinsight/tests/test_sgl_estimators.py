import pytest

from afqinsight.sgl_estimators import SGLEstimator
from afqinsight import SGLClassifier
from afqinsight import SGLRegressor

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._testing import assert_array_almost_equal


@pytest.mark.parametrize("Estimator", [SGLEstimator, SGLRegressor, SGLClassifier])
def test_all_estimators(Estimator):
    return check_estimator(Estimator)


# When alpha=1, SGL should behave like the lasso. These next tests
# replicate the unit testing for sklearn.linear_model.Lasso
def test_lasso_toy():
    # Test on a toy example for various values of alpha.
    # When validating this against glmnet notice that glmnet divides it
    # against n_obs.

    X = [[-1], [0], [1]]
    y = [-1, 0, 1]  # just a straight line
    T = [[2], [3], [4]]  # test sample

    clf = SGLRegressor(lambd=1e-8)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])

    clf = SGLRegressor(lambd=0.1)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])

    clf = SGLRegressor(lambd=0.5)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])

    clf = SGLRegressor(lambd=1)
    clf.fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])
