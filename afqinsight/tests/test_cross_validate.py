import os
import pytest
import shutil
import tempfile

from sklearn import datasets, linear_model
from sklearn.model_selection import KFold
from afqinsight import cross_validate_checkpoint


@pytest.mark.parametrize("return_estimator", [True, False])
@pytest.mark.parametrize("serialize_cv", [True, False])
def test_cross_validate(return_estimator, serialize_cv):
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()

    with pytest.raises(ValueError):
        cross_validate_checkpoint(lasso, X, y, cv=3, checkpoint=True)

    tempdir = tempfile.mkdtemp()

    lasso = linear_model.Lasso()
    cv = KFold(n_splits=3)
    cross_validate_checkpoint(
        lasso,
        X,
        y,
        cv=cv,
        checkpoint=True,
        workdir=tempdir,
        return_estimator=return_estimator,
        serialize_cv=serialize_cv,
    )
    cv_files_1 = os.listdir(tempdir)

    lasso = linear_model.Lasso()
    cv = KFold(n_splits=3)
    cross_validate_checkpoint(
        lasso,
        X,
        y,
        cv=cv,
        checkpoint=True,
        workdir=tempdir,
        return_estimator=return_estimator,
    )
    cv_files_2 = os.listdir(tempdir)

    assert set(cv_files_1) == set(cv_files_2)
    shutil.rmtree(tempdir)
