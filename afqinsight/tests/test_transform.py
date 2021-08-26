import json
import numpy as np
import os.path as op
import pandas as pd
import pytest

import afqinsight as afqi
from afqinsight import AFQDataFrameMapper
from afqinsight import multicol2dicts, multicol2sets
from afqinsight.transform import isiterable

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


@pytest.mark.parametrize("concat_subject_session", [True, False])
def test_AFQDataFrameMapper(concat_subject_session):
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    transformer = AFQDataFrameMapper(concat_subject_session=concat_subject_session)
    X = transformer.fit_transform(nodes)
    groups = transformer.groups_
    cols = transformer.feature_names_
    subjects = transformer.subjects_

    X_ref = np.load(op.join(test_data_path, "test_transform_x.npy"))
    groups_ref = np.load(op.join(test_data_path, "test_transform_groups.npy"))
    cols_ref = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]

    assert np.allclose(X, X_ref, equal_nan=True)  # nosec
    assert np.allclose(groups, groups_ref)  # nosec
    assert cols == cols_ref  # nosec
    print("subjectID")
    print(nodes.subjectID.unique())
    print("sessionID")
    print(nodes.sessionID.unique())
    print("concat")
    print((nodes.subjectID + nodes.sessionID).unique())
    print("subjects")
    print(set(subjects))
    if concat_subject_session:
        assert set(subjects) == set(
            (nodes.subjectID + nodes.sessionID).unique()
        )  # nosec
    else:
        assert set(subjects) == set(nodes.subjectID.unique())  # nosec


@pytest.mark.parametrize("concat_subject_session", [True, False])
def test_AFQDataFrameMapper_mean(concat_subject_session):
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    transformer = AFQDataFrameMapper(
        bundle_agg_func="mean", concat_subject_session=concat_subject_session
    )
    X = transformer.fit_transform(nodes)
    groups = transformer.groups_
    cols = transformer.feature_names_
    subjects = transformer.subjects_

    X_ref = (
        nodes.groupby(["subjectID", "tractID"])
        .agg("mean")
        .drop("nodeID", axis="columns")
        .unstack("tractID")
        .to_numpy()
    )
    groups_ref = [np.array([idx]) for idx in range(X.shape[1])]
    cols_ref = set(
        [
            tuple([item[0], item[1]])
            for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
        ]
    )

    assert np.allclose(groups, groups_ref)  # nosec
    assert set(cols) == cols_ref  # nosec
    if concat_subject_session:
        assert set(subjects) == set(
            (nodes.subjectID + nodes.sessionID).unique()
        )  # nosec
    else:
        assert set(subjects) == set(nodes.subjectID.unique())  # nosec
    assert np.allclose(X, X_ref, equal_nan=True)  # nosec


@pytest.mark.parametrize("bundle_agg_func", [None, "mean"])
def test_AFQDataFrameMapper_fit_transform(bundle_agg_func):
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    transformer = AFQDataFrameMapper(
        bundle_agg_func=bundle_agg_func, pd_interpolate_kwargs={"method": "cubic"}
    )
    X_ft = transformer.fit_transform(nodes)
    groups_ft = transformer.groups_
    cols_ft = transformer.feature_names_
    subjects_ft = transformer.subjects_

    transformer = AFQDataFrameMapper(
        bundle_agg_func=bundle_agg_func, pd_interpolate_kwargs={"method": "cubic"}
    )
    transformer.fit(nodes)
    X_t = transformer.transform(nodes)
    groups_t = transformer.groups_
    cols_t = transformer.feature_names_
    subjects_t = transformer.subjects_

    assert np.allclose(X_ft, X_t, equal_nan=True)  # nosec
    assert np.allclose(groups_ft, groups_t)  # nosec
    assert cols_ft == cols_t  # nosec
    assert subjects_ft == subjects_t  # nosec


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
