import json
import numpy as np
import os.path as op
import pandas as pd
import pytest

import afqinsight as afqi
from afqinsight import AFQDataFrameMapper
from afqinsight import (
    multicol2dicts,
    multicol2sets,
    sort_features,
    beta_hat_by_groups,
    unfold_beta_hat_by_metrics,
)
from afqinsight.transform import isiterable

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


@pytest.mark.parametrize("concat_subject_session", [True, False])
@pytest.mark.parametrize("set_ses_to_int", [True, False])
def test_AFQDataFrameMapper(concat_subject_session, set_ses_to_int):
    nodes_path = op.join(test_data_path, "nodes.csv")
    nodes = pd.read_csv(nodes_path)
    if set_ses_to_int:
        nodes["sessionID"] = 1
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
    if concat_subject_session:
        assert set(subjects) == set(
            (nodes.subjectID + nodes.sessionID.astype(str)).unique()
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


def _is_lateral_bundle(bundle):
    return "left" in bundle.lower() or "right" in bundle.lower()


def _drop_left_right(bundle):
    return (
        bundle.replace("left", "")
        .replace("right", "")
        .replace("Left", "")
        .replace("Right", "")
        .strip(" ")
    )


def _relateralize_set(_set):
    lateral_list = list(filter(_is_lateral_bundle, _set))
    if lateral_list:
        return _set - set([_drop_left_right(lateral_list[0])])
    else:
        return _set


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

    label_sets = multicol2sets(
        pd.MultiIndex.from_tuples(cols, names=["metric", "tractID", "nodeID"]),
        tract_symmetry=False,
    )
    label_sets_ref = np.array([_relateralize_set(_set) for _set in label_sets_ref])
    assert np.all(label_sets == label_sets_ref)  # nosec

    label_dicts = multicol2dicts(
        pd.MultiIndex.from_tuples(cols, names=["metric", "tractID", "nodeID"])
    )
    with open(
        op.join(test_data_path, "test_multicol2dicts_label_dicts.json"), "r"
    ) as fp:
        label_dicts_ref = json.load(fp)

    assert label_dicts == label_dicts_ref  # nosec

    label_dicts = multicol2dicts(
        pd.MultiIndex.from_tuples(cols, names=["metric", "tractID", "nodeID"]),
        tract_symmetry=False,
    )
    label_dicts_ref = [
        {key: val for key, val in _dict.items() if key != "symmetrized_tractID"}
        for _dict in label_dicts_ref
    ]
    assert label_dicts == label_dicts_ref  # nosec

    scores = np.arange(len(label_dicts))
    sorted_labels = sort_features(label_dicts, scores)
    assert sorted_labels == list(zip(reversed(label_dicts), reversed(scores)))  # nosec

    sorted_labels = sort_features(label_dicts, reversed(scores))
    assert sorted_labels == list(zip(label_dicts, reversed(scores)))  # nosec


def test_beta_hat_transforms():
    cols = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]
    multicols = pd.MultiIndex.from_tuples(cols, names=["metric", "tractID", "nodeID"])
    beta_hat = np.random.rand(len(cols))
    result = beta_hat_by_groups(beta_hat, multicols)
    reshaped_beta = beta_hat.reshape(2, 3, -1)

    for bundle_idx, bundle in enumerate(result.keys()):
        for metric_idx, metric in enumerate(result[bundle].keys()):
            assert np.allclose(
                reshaped_beta[metric_idx, bundle_idx], result[bundle][metric]
            )

    result = unfold_beta_hat_by_metrics(
        beta_hat, multicols, tract_names=list(result.keys())
    )
    reshaped_beta = beta_hat.reshape(2, -1)
    for metric_idx, metric in enumerate(result.keys()):
        assert np.allclose(reshaped_beta[metric_idx], result[metric])

    beta_hat[0:100] = 0.0
    beta_hat[300:400] = 0.0
    result = beta_hat_by_groups(beta_hat, multicols, drop_zeros=True)
    reshaped_beta = beta_hat.reshape(2, 3, -1)

    for bundle_idx, bundle in enumerate(result.keys()):
        for metric_idx, metric in enumerate(result[bundle].keys()):
            assert np.allclose(
                reshaped_beta[metric_idx, bundle_idx + 1], result[bundle][metric]
            )
