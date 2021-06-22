import numpy as np
import os.path as op
import pandas as pd
import pytest
import tempfile

import afqinsight as afqi
from afqinsight.datasets import load_afq_data, fetch_sarica, fetch_weston_havens

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_fetch():
    X, y, groups, feature_names, group_names, subjects, _ = fetch_sarica()
    assert X.shape == (48, 3600)
    assert y.shape == (48,)
    assert len(groups) == 36
    assert len(feature_names) == 3600
    assert len(group_names) == 36
    assert len(subjects) == 48
    assert op.isfile(op.join(afqi.datasets.DATA_DIR, "sarica_data", "nodes.csv"))
    assert op.isfile(op.join(afqi.datasets.DATA_DIR, "sarica_data", "subjects.csv"))

    X, y, groups, feature_names, group_names, subjects = fetch_weston_havens()
    assert X.shape == (77, 3600)
    assert y.shape == (77,)
    assert len(groups) == 36
    assert len(feature_names) == 3600
    assert len(group_names) == 36
    assert len(subjects) == 77
    assert op.isfile(op.join(afqi.datasets.DATA_DIR, "weston_havens_data", "nodes.csv"))
    assert op.isfile(
        op.join(afqi.datasets.DATA_DIR, "weston_havens_data", "subjects.csv")
    )

    with tempfile.TemporaryDirectory() as td:
        _ = fetch_sarica(data_home=td)
        _ = fetch_weston_havens(data_home=td)
        assert op.isfile(op.join(td, "sarica_data", "nodes.csv"))
        assert op.isfile(op.join(td, "sarica_data", "subjects.csv"))
        assert op.isfile(op.join(td, "weston_havens_data", "nodes.csv"))
        assert op.isfile(op.join(td, "weston_havens_data", "subjects.csv"))


def test_load_afq_data_smoke():
    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
    )
    assert len(output) == 7  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        return_sessions=True,
    )
    assert len(output) == 8  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
    )
    assert len(output) == 5  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
        return_sessions=True,
    )
    assert len(output) == 6  # nosec


def test_load_afq_data():
    X, y, groups, feature_names, group_names, subjects, classes = load_afq_data(
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
    assert group_names == [tup[0:2] for tup in cols_ref if tup[2] == 0]  # nosec
    assert set(subjects) == set(nodes.subjectID.unique())  # nosec
    assert all(classes["test_class"] == np.array(["c0", "c1"]))

    with pytest.raises(ValueError):
        load_afq_data(
            workdir=test_data_path,
            target_cols=["test_class"],
            label_encode_cols=["test_class", "error"],
        )
