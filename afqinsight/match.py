"""Perform statistical matching comparing test and control units."""

import numpy as np
from scipy.optimize import linear_sum_assignment


__all__ = [
    "mahalonobis_dist_match_df", "mahalonobis_dist_match",
    "mahalonobis_dist_arr"]


def mahalonobis_dist_match_df(data, id_column, test_ids, ctrl_ids,
                              feature_columns, threshold=0.2):
    """
    Perform Mahalanobis distance matching (MDM) on a pandas
    dataframe.

    Parameters
    ----------
    data : pandas Dataframe
        The feature samples for test subjects.
    id_column : str
        Column with unique subject identifiers.
    test_ids : list of str
        List of subject identifiers for test subjects.
    ctrl_ids : list of str
        List of subject identifiers for control subjects.
    feature_columns : list of str
        List of columns to use as features for MDM.
    threshold : float
        Mahalnobis distance beyond which matches are not accepted.

    Returns
    -------
    filtered_test_ids : array-like
        List of subject identifiers for successfully matched test subjects
    filtered_ctrl_ids : array-like
        List of subject identifiers for matched control subjects,
        this array will be the same length as filtered_test_ids
    """
    test_X = data[data[id_column].isin(test_ids)][feature_columns].to_numpy()
    ctrl_X = data[data[id_column].isin(ctrl_ids)][feature_columns].to_numpy()
    filtered_test_ind, filtered_ctrl_ind = mahalonobis_dist_match(
        test_X, ctrl_X, threshold=threshold)
    test_ids = np.asarray(test_ids)
    ctrl_ids = np.asarray(ctrl_ids)
    return test_ids[filtered_test_ind], ctrl_ids[filtered_ctrl_ind]


def mahalonobis_dist_match(test_X, ctrl_X, threshold=0.2):
    """
    Perform Mahalanobis distance matching (MDM) between two arrays.

    Parameters
    ----------
    test_X : array-like of shape (n_samples1, n_features)
        The feature samples for test subjects.
    ctrl_X : array-like of shape (n_samples2, n_features)
        Feature samples for control subjects from which
        matches will be selected.
    threshold : float
        Mahalnobis distance beyond which matches are not accepted.

    Returns
    -------
    filtered_test_ind : array-like
        Indices into test_X of successfully matched test subjects
    filtered_ctrl_ind : array-like
        Indices into ctrl_X of matched control subjects,
        this array will be the same length as filtered_test_ind
    """
    test_X = np.asarray(test_X)
    ctrl_X = np.asarray(ctrl_X)

    if np.any(np.isnan(test_X)) or np.any(np.isnan(ctrl_X)):
        raise ValueError((
            "There are NaNs in test_X or ctrl_X. "
            "Please replace these NaNs using interpolation or by removing "
            "the subjects with NaNs before calling mahalonobis_dist_match. "))

    # calculate Mahalonobis distance between test and control
    nbrs = mahalonobis_dist_arr(test_X, ctrl_X)

    # assign neighbors using Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(nbrs)

    # remove matches that are too bad
    filtered_test_ind = []
    filtered_ctrl_ind = []
    for row_idx, col_idx in zip(row_ind, col_ind):
        if nbrs[row_idx, col_idx] <= threshold:
            filtered_test_ind.append(row_idx)
            filtered_ctrl_ind.append(col_idx)

    return filtered_test_ind, filtered_ctrl_ind


def mahalonobis_dist_arr(arr1, arr2):
    """
    Calculate the Mahalonobis distance between two
    2-dimensional arrays along the first axis.

    Parameters
    ----------
    arr1 : array-like of shape (n_samples1, n_features)
        First input array.
    arr2 : array-like of shape (n_samples2, n_features)
        Second input array.

    Returns
    -------
    nbrs : array-like of shape (n_samples1, n_samples2)
        Mahalonobis distance between each sample in the
        two input arrays.
    """
    v_inv = np.linalg.inv(np.cov(
        np.concatenate((arr1, arr2), axis=0).T, ddof=0))
    diff = arr1[:, np.newaxis] - arr2[np.newaxis, :]
    nbrs = np.sqrt(np.sum((diff @ v_inv) * diff, axis=2))
    return nbrs
