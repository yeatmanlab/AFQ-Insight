import pytest

from afqinsight.sgl import SGLBaseEstimator
from afqinsight import LogisticSGL, SGL, SGLCV

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._testing import assert_array_almost_equal


@pytest.mark.parametrize("Estimator", [SGLBaseEstimator, SGL, LogisticSGL, SGLCV])
def test_all_estimators(Estimator):
    return check_estimator(Estimator())


def test_sgl_input_validation():
    X = [[0], [0], [0]]
    y = [0, 0, 0]

    with pytest.raises(ValueError):
        SGL(l1_ratio=1.0, alpha=0.1, warm_start="error").fit(X, y)

    with pytest.raises(ValueError):
        SGL(l1_ratio=1.0, alpha=0.1).fit(X, y, loss="error")


def test_sgl_zero():
    # Check that SGL can handle zero data without crashing
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = SGL(l1_ratio=1.0, alpha=0.1).fit(X, y)
    pred = clf.predict([[1], [2], [3]])
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])


# When l1_ratio=1, SGL should behave like the lasso. These next tests
# replicate the unit testing for sklearn.linear_model.Lasso
@pytest.mark.parametrize("loss", ["squared_loss", "huber"])
def test_sgl_toy(loss):
    # Test on a toy example for various values of l1_ratio.
    # When validating this against glmnet notice that glmnet divides it
    # against n_obs.

    X = [[-1], [0], [1]]
    y = [-1, 0, 1]  # just a straight line
    T = [[2], [3], [4]]  # test sample

    clf = SGL(alpha=1e-8)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])

    clf = SGL(alpha=0.1)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])

    clf = SGL(alpha=0.5)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])

    clf = SGL(alpha=1)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])


def test_alpha_grid():
    # TODO: Run _alpha_grid on a few problems
    # TODO: Confirm that the highest alpha yields coef_ == 0
    # TODO: Smoke test that n_alphas and eps gives the expected output
    pass
