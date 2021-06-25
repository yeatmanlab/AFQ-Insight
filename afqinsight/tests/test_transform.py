import json
import numpy as np
import os.path as op
import pandas as pd

import afqinsight as afqi
from afqinsight import AFQDataFrameMapper
from afqinsight import multicol2dicts, multicol2sets
from afqinsight.transform import isiterable

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_AFQDataFrameMapper():
    n_nodes_path = op.join(test_data_path, "n_nodes.csv")
    n_nodes = pd.read_csv(n_nodes_path)
    transformer = AFQDataFrameMapper()
    X = transformer.fit_transform(n_nodes)
    groups = transformer.groups_
    cols = transformer.feature_names_
    n_subjects = transformer.n_subjects_

    X_ref = np.load(op.join(test_data_path, "test_transform_x.npy"))
    groups_ref = np.load(op.join(test_data_path, "test_transform_groups.npy"))
    cols_ref = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]

    assert np.allclose(X, X_ref, equal_nan=True)  # nosec
    assert np.allclose(groups, groups_ref)  # nosec
    assert cols == cols_ref  # nosec
    assert set(n_subjects) == set(n_nodes.subjectID.unique())  # nosec


def test_AFQDataFrameMapper_fit_transform():
    n_nodes_path = op.join(test_data_path, "n_nodes.csv")
    n_nodes = pd.read_csv(n_nodes_path)
    transformer = AFQDataFrameMapper(pd_interpolate_kwargs={"method": "cubic"})
    X_ft = transformer.fit_transform(n_nodes)
    groups_ft = transformer.groups_
    cols_ft = transformer.feature_names_
    n_subjects_ft = transformer.n_subjects_

    transformer = AFQDataFrameMapper(pd_interpolate_kwargs={"method": "cubic"})
    transformer.fit(n_nodes)
    X_t = transformer.transform(n_nodes)
    groups_t = transformer.groups_
    cols_t = transformer.feature_names_
    n_subjects_t = transformer.n_subjects_

    assert np.allclose(X_ft, X_t, equal_nan=True)  # nosec
    assert np.allclose(groups_ft, groups_t)  # nosec
    assert cols_ft == cols_t  # nosec
    assert n_subjects_ft == n_subjects_t  # nosec


def test_isiterable():
    assert isiterable(range(10))  # nosec
    assert not isiterable(5)  # nosec
    assert isiterable(np.arange(10))  # nosec


def test_multicol_utils():
    cols = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]
    label_sets = multicol2sets(
        pd.MultiIndex.from_tuples(cols, names=["metric", "tractID", "nodeID"])
    )
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )
    assert np.all(label_sets == label_sets_ref)  # nosec

    label_dicts = multicol2dicts(
        pd.MultiIndex.from_tuples(cols, names=["metric", "tractID", "nodeID"])
    )
    with open(
        op.join(test_data_path, "test_multicol2dicts_label_dicts.json"), "r"
    ) as fp:
        label_dicts_ref = json.load(fp)

    assert label_dicts == label_dicts_ref  # nosec
