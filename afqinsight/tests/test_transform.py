import numpy as np
import os.path as op
import pandas as pd
import pytest

import afqinsight as afqi
from afqinsight import AFQDataFrameMapper
from afqinsight import remove_group, remove_groups
from afqinsight import select_group, select_groups
from afqinsight import shuffle_group
from afqinsight import multicol2sets
from afqinsight.transform import isiterable

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_AFQFeatureTransformer():
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    transformer = AFQDataFrameMapper()
    X = transformer.fit_transform(nodes)
    groups = transformer.groups_
    cols = transformer.feature_names_
    subjects = transformer.subjects_

    X_ref = np.load(op.join(test_data_path, "test_transform_x.npy"))
    groups_ref = np.load(op.join(test_data_path, "test_transform_groups.npy"))
    cols_ref = pd.read_hdf(
        op.join(test_data_path, "test_transform_cols.h5"), key="cols"
    ).index

    assert np.allclose(X, X_ref, equal_nan=True)
    assert np.allclose(groups, groups_ref)
    assert cols == cols_ref.tolist()
    assert set(subjects) == set(nodes.subjectID.unique())


def test_AFQFeatureTransformer_fit_transform():
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    transformer = AFQDataFrameMapper()
    X_ft = transformer.fit_transform(nodes)
    groups_ft = transformer.groups_
    cols_ft = transformer.feature_names_
    subjects_ft = transformer.subjects_

    transformer = AFQDataFrameMapper()
    transformer.fit(nodes)
    X_t = transformer.transform(nodes)
    groups_t = transformer.groups_
    cols_t = transformer.feature_names_
    subjects_t = transformer.subjects_

    assert np.allclose(X_ft, X_t, equal_nan=True)
    assert np.allclose(groups_ft, groups_t)
    assert cols_ft == cols_t
    assert subjects_ft == subjects_t


def test_isiterable():
    assert isiterable(range(10))
    assert not isiterable(5)
    assert isiterable(np.arange(10))


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

    assert np.allclose(X_removed, X_ref, equal_nan=True)


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
    assert np.allclose(X_removed, X_ref, equal_nan=True)


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
    assert np.allclose(X_select, X_ref, equal_nan=True)


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
    assert np.allclose(X_select, X_ref, equal_nan=True)


def test_shuffle_group():
    X = np.load(op.join(test_data_path, "test_transform_x.npy"))
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )
    X_ref = np.load(op.join(test_data_path, "test_shuffle_group_x.npy"))

    X_shuffle = shuffle_group(X, ("Corticospinal",), label_sets_ref, random_seed=42)
    assert np.allclose(X_shuffle, X_ref, equal_nan=True)


def test_multicol2sets():
    cols = pd.read_hdf(
        op.join(test_data_path, "test_transform_cols.h5"), key="cols"
    ).index

    label_sets = multicol2sets(cols)
    label_sets_ref = np.load(
        op.join(test_data_path, "test_multicol2sets_label_sets.npy"), allow_pickle=True
    )

    assert np.all(label_sets == label_sets_ref)
