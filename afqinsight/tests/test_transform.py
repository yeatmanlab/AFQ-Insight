import json
import numpy as np
import os.path as op
import pandas as pd
import pytest

import afqinsight as afqi
from afqinsight import AFQDataFrameMapper, GroupExtractor
from afqinsight import remove_group, remove_groups
from afqinsight import select_group, select_groups
from afqinsight import shuffle_group
from afqinsight import multicol2dicts, multicol2sets
from afqinsight.transform import isiterable
from sklearn.utils.estimator_checks import check_estimator

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_AFQDataFrameMapper():
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    transformer = AFQDataFrameMapper()
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
    assert set(subjects) == set(nodes.subjectID.unique())  # nosec


def test_AFQDataFrameMapper_fit_transform():
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    transformer = AFQDataFrameMapper(pd_interpolate_kwargs={"method": "cubic"})
    X_ft = transformer.fit_transform(nodes)
    groups_ft = transformer.groups_
    cols_ft = transformer.feature_names_
    subjects_ft = transformer.subjects_

    transformer = AFQDataFrameMapper(pd_interpolate_kwargs={"method": "cubic"})
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


def test_value_errors():
    X = np.load(op.join(test_data_path, "test_transform_x.npy"))
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )

    X = X[:, :, np.newaxis]
    with pytest.raises(ValueError):
        remove_group(X, ("Callosum Forceps Major",), label_sets_ref)

    with pytest.raises(ValueError):
        remove_groups(X, ("Callosum Forceps Major",), label_sets_ref)

    with pytest.raises(ValueError):
        select_group(X, ("Callosum Forceps Major",), label_sets_ref)

    with pytest.raises(ValueError):
        select_groups(X, ("Callosum Forceps Major",), label_sets_ref)


@pytest.mark.parametrize("flatten", [True, False])
def test_remove_group(flatten):
    X = np.load(op.join(test_data_path, "test_transform_x.npy"))
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )
    X_ref = np.load(op.join(test_data_path, "test_remove_group_x.npy"))

    if flatten:
        idx = np.random.randint(0, X.shape[0])
        X = np.squeeze(X[idx, :])
        X_ref = np.squeeze(X_ref[idx, :])

    X_removed = remove_group(X, ("Callosum Forceps Major",), label_sets_ref)
    assert np.allclose(X_removed, X_ref, equal_nan=True)  # nosec


@pytest.mark.parametrize("flatten", [True, False])
def test_remove_groups(flatten):
    X = np.load(op.join(test_data_path, "test_transform_x.npy"))
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )
    X_ref = np.load(op.join(test_data_path, "test_remove_groups_x.npy"))

    if flatten:
        idx = np.random.randint(0, X.shape[0])
        X = np.squeeze(X[idx, :])
        X_ref = np.squeeze(X_ref[idx, :])

    X_removed = remove_groups(
        X, [("Callosum Forceps Major",), ("Uncinate",), ("fa",)], label_sets_ref
    )
    assert np.allclose(X_removed, X_ref, equal_nan=True)  # nosec


@pytest.mark.parametrize("flatten", [True, False])
def test_select_group(flatten):
    X = np.load(op.join(test_data_path, "test_transform_x.npy"))
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )
    X_ref = np.load(op.join(test_data_path, "test_select_group_x.npy"))

    if flatten:
        idx = np.random.randint(0, X.shape[0])
        X = np.squeeze(X[idx, :])
        X_ref = np.squeeze(X_ref[idx, :])

    X_select = select_group(X, ("Callosum Forceps Major",), label_sets_ref)
    assert np.allclose(X_select, X_ref, equal_nan=True)  # nosec


@pytest.mark.parametrize("flatten", [True, False])
def test_select_groups(flatten):
    X = np.load(op.join(test_data_path, "test_transform_x.npy"))
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )
    X_ref = np.load(op.join(test_data_path, "test_select_groups_x.npy"))

    if flatten:
        idx = np.random.randint(0, X.shape[0])
        X = np.squeeze(X[idx, :])
        X_ref = np.squeeze(X_ref[idx, :])

    X_select = select_groups(
        X, [("Callosum Forceps Major",), ("Uncinate",), ("fa",)], label_sets_ref
    )
    assert np.allclose(X_select, X_ref, equal_nan=True)  # nosec


def test_shuffle_group():
    X = np.load(op.join(test_data_path, "test_transform_x.npy"))
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )
    X_ref = np.load(op.join(test_data_path, "test_shuffle_group_x.npy"))

    X_shuffle = shuffle_group(X, ("Corticospinal",), label_sets_ref, random_seed=42)
    assert np.allclose(X_shuffle, X_ref, equal_nan=True)  # nosec


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


def test_GroupExtractor():
    X = np.array([list(range(10))] * 10)
    groups = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]

    extract = 2
    ge = GroupExtractor(groups=groups, extract=extract)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, groups[extract]], X_tr)  # nosec

    extract = [0, 3]
    ge = GroupExtractor(groups=groups, extract=extract)
    X_tr = ge.fit_transform(X)
    idx = np.concatenate([groups[e] for e in extract])
    assert np.allclose(X[:, idx], X_tr)  # nosec

    ge = GroupExtractor()
    X_tr = ge.fit_transform(X)
    assert np.allclose(X_tr, X)  # nosec

    with pytest.raises(ValueError):
        GroupExtractor(extract="error").fit_transform(X)


@pytest.mark.parametrize("Transformer", [GroupExtractor])
def test_all_estimators(Transformer):
    return check_estimator(Transformer())
