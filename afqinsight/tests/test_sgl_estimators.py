import pytest

from sklearn.utils.estimator_checks import check_estimator

from afqinsight.sgl_estimators import SGLEstimator
from afqinsight import SGLClassifier
from afqinsight import SGLRegressor


@pytest.mark.parametrize("Estimator", [SGLEstimator, SGLRegressor, SGLClassifier])
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
