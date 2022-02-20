"""Test functions in the match module"""

import afqinsight.match as aim
import numpy as np
import pandas as pd
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import mahalanobis
import pytest


test_features = np.asarray([[1, 2, 3, 4], [4, 2, 1, 3], [1, 4, 5, 6]])


ctrl_features = np.asarray(
    [
        [3, 2, 3, 4],
        [4, 5, 1, 3],
        [1, 10, 5, 6],
        [1, 2, 2, 4],
        [4, 2, 0, 3],
        [1, 4, 9, 6],
    ]
)


data = pd.DataFrame(
    {
        "eid": [34, 35, 36, 37, 38, 39, 40, 41, 42],
        "age": [12, 345, 63, 4, 5, 6, 89, 88, 10],
        "status": [1, 1, 1, 0, 0, 0, 0, 0, 0],
    }
)
for ii in range(4):
    data[f"feature_{ii}"] = [*test_features[:, ii], *ctrl_features[:, ii]]


def test_mahalonobis_dist():
    nbrs = aim._mahalonobis_dist(test_features, ctrl_features)

    v_inv = np.linalg.inv(
        np.cov(np.concatenate((test_features, ctrl_features), axis=0).T, ddof=0)
    )
    nbrs_scipy = np.zeros((test_features.shape[0], ctrl_features.shape[0]))
    for ii, test_sub in enumerate(test_features):
        for jj, ctrl_sub in enumerate(ctrl_features):
            nbrs_scipy[ii, jj] = mahalanobis(test_sub, ctrl_sub, v_inv)

    assert_array_almost_equal(nbrs, nbrs_scipy)


def test_mahalonobis_dist_match_df():
    matched_df = aim.mahalonobis_dist_match(
        data=data,
        status_col="status",
        feature_cols=["feature_0", "feature_1", "feature_2", "feature_3"],
        threshold=1,
    )

    assert_array_equal(
        matched_df.loc[matched_df.status.astype(bool)].eid.to_numpy(), [34, 35]
    )
    assert_array_equal(
        matched_df.loc[~matched_df.status.astype(bool)].eid.to_numpy(), [40, 41]
    )


def test_mahalonobis_dist_match_df_err():
    with pytest.raises(ValueError):  # no status column
        aim.mahalonobis_dist_match(data=data)
    with pytest.raises(ValueError):  # status has more than two unique values
        aim.mahalonobis_dist_match(data=data, status_col="feature_0")


def test_mahalonobis_dist_match_df_feauture_none():
    data_wo_extras = data.drop(columns=["eid", "age"])
    matched_df = aim.mahalonobis_dist_match(
        data=data_wo_extras,
        status_col="status",
        threshold=1,
    )

    assert_array_equal(
        matched_df.loc[matched_df.status.astype(bool)].feature_2.to_numpy(), [3, 1]
    )
    assert_array_equal(
        matched_df.loc[~matched_df.status.astype(bool)].feature_2.to_numpy(), [2, 0]
    )


def test_mahalonobis_dist_match():
    matched_df = aim.mahalonobis_dist_match(
        test=test_features, ctrl=ctrl_features, threshold=1
    )

    assert_array_equal(
        matched_df.loc[matched_df.status.astype(bool)].feature_2.to_numpy(), [3, 1]
    )
    assert_array_equal(
        matched_df.loc[~matched_df.status.astype(bool)].feature_2.to_numpy(), [2, 0]
    )
    assert np.sum(matched_df.status.to_numpy() == 1) == np.sum(
        matched_df.status.to_numpy() == 0
    )


def test_mahalonobis_dist_match_err():
    with pytest.raises(ValueError):  # test if nan error is raised
        test_features_with_nans = test_features.copy()
        test_features_with_nans[0, 2] = np.nan
        aim.mahalonobis_dist_match(
            test=test_features_with_nans, ctrl=ctrl_features, threshold=1
        )
