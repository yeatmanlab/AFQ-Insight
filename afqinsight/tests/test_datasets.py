import numpy as np
import os.path as op
import pandas as pd
import pytest

import afqinsight as afqi
from afqinsight.datasets import load_afq_data

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_load_afq_data():
    X, y, groups, feature_names, subjects, classes = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
    )

    nodes = pd.read_csv(op.join(test_data_path, "nodes.csv"))
    X_ref = np.load(op.join(test_data_path, "test_transform_x.npy"))
    y_ref = np.load(op.join(test_data_path, "test_data_y.npy"))
    groups_ref = np.load(op.join(test_data_path, "test_transform_groups.npy"))
    cols_ref = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]

    assert np.allclose(X, X_ref, equal_nan=True)  # nosec
    assert np.allclose(y, y_ref)  # nosec
    assert np.allclose(groups, groups_ref)  # nosec
    assert feature_names == cols_ref  # nosec
    assert set(subjects) == set(nodes.subjectID.unique())  # nosec
    assert all(classes["test_class"] == np.array(["c0", "c1"]))

    with pytest.raises(ValueError):
        load_afq_data(
            workdir=test_data_path,
            target_cols=["test_class"],
            label_encode_cols=["test_class", "error"],
        )
