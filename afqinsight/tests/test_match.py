"""Test functions in the match module"""

import afqinsight.match as aim
import numpy as np
import pandas as pd
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_array_equal)
from scipy.spatial.distance import mahalanobis


test_features = np.asarray([
    [1, 2, 3, 4], [4, 2, 1, 3], [1, 4, 5, 6]])


ctrl_features = np.asarray([
    [3, 2, 3, 4], [4, 5, 1, 3], [1, 10, 5, 6],
    [1, 2, 2, 4], [4, 2, 0, 3], [1, 4, 9, 6]])


def test_mahalonobis_dist_arr():
    nbrs = aim.mahalonobis_dist_arr(test_features, ctrl_features)

    v_inv = np.linalg.inv(np.cov(
        np.concatenate((test_features, ctrl_features), axis=0).T, ddof=0))
    nbrs_scipy = np.zeros((
        test_features.shape[0],
        ctrl_features.shape[0]))
    for ii, test_sub in enumerate(test_features):
        for jj, ctrl_sub in enumerate(ctrl_features):
            nbrs_scipy[ii, jj] = mahalanobis(test_sub, ctrl_sub, v_inv)

    assert_array_almost_equal(nbrs, nbrs_scipy)


def test_mahalonobis_dist_match_df():
    data = pd.DataFrame({
        "eid": [34, 35, 36, 37, 38, 39, 40, 41, 42],
        "age": [12, 345, 63, 4, 5, 6, 89, 88, 10],
    })
    for ii in range(4):
        data[f"feature_{ii}"] = [*test_features[:, ii], *ctrl_features[:, ii]]
    filtered_test_id, filtered_ctrl_id = aim.mahalonobis_dist_match_df(
        data,
        "eid",
        [34, 35, 36],
        [37, 38, 39, 40, 41, 42],
        ["feature_0", "feature_1", "feature_2", "feature_3"],
        threshold=1)
    assert_array_equal(filtered_test_id, [34, 35])
    assert_array_equal(filtered_ctrl_id, [40, 41])


def test_mahalonobis_dist_match():
    filtered_test_ind, filtered_ctrl_ind = aim.mahalonobis_dist_match(
        test_features, ctrl_features, threshold=1)
    assert_array_equal(filtered_test_ind, [0, 1])
    assert_array_equal(filtered_ctrl_ind, [3, 4])
    assert len(filtered_test_ind) == len(filtered_ctrl_ind)
