"""Perform statistical matching comparing test and control units."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


__all__ = ["mahalonobis_dist_match"]


def mahalonobis_dist_match(
    data=None, test=None, ctrl=None, status_col=None, feature_cols=None, threshold=0.2
):
    """
    Perform Mahalanobis distance matching (MDM) on a pandas dataframe or 2d array.

    If data is None, then test and ctrl must be arrays.
    If data is provided, then status_col and feature_cols are used.

    Parameters
    ----------
    data : pandas Dataframe, optional
        The feature samples for test subjects.
        Default: None
    test : array-like of shape (n_samples1, n_features), optional
        The feature samples for test subjects.
        Default: None
    ctrl : array-like of shape (n_samples2, n_features), optional
        Feature samples for control subjects from which
        matches will be selected.
        Default: None
    status_col : str, optional
        Column with subject status. Must have only two unique values,
        evaluating to True and False, where True indicates a test subject
        and False indicates a control.
        Default: None
    feature_cols : list of str, optional
        List of columns to use as features for MDM.
        If feature_columns is None then we use all of the columns
        except status_col as the features.
        Default: None
    threshold : float, optional
        Mahalnobis distance beyond which matches are not accepted.
        Default: None

    Returns
    -------
    selected_data : pandas Dataframe
        If data is None, returns a Dataframe with columns named
        'feature_0', 'feature_1', etc. and a 'status' column
        where 1 corresponds to test and 0 corresponds to control.
        If data is not None, returns data with only the selected rows.
    """
    if data is not None:
        if status_col is None:
            raise ValueError(
                ("If data is provided," " status_col must also be provided")
            )

        if len(data[status_col].unique()) != 2:
            raise ValueError(
                ("status_col in data must " "contain only 2 unique values")
            )

        if feature_cols is None:
            feature_cols = data.columns.drop(status_col)

        df_test_rows = data[status_col].astype(bool)
        df_ctrl_rows = ~data[status_col].astype(bool)
        test = data.loc[df_test_rows, feature_cols].to_numpy()
        ctrl = data.loc[df_ctrl_rows, feature_cols].to_numpy()
    else:
        test = np.asarray(test)
        ctrl = np.asarray(ctrl)

    if np.any(np.isnan(test)) or np.any(np.isnan(ctrl)):
        raise ValueError(
            (
                "There are NaNs in test or ctrl data. "
                "Please replace these NaNs using interpolation or by removing "
                "the subjects with NaNs before calling mahalonobis_dist_match. "
            )
        )

    # calculate Mahalonobis distance between test and control
    nbrs = _mahalonobis_dist(test, ctrl)

    # assign neighbors using Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(nbrs)

    # remove matches that are too bad
    filtered_test_ind = []
    filtered_ctrl_ind = []
    for row_idx, col_idx in zip(row_ind, col_ind):
        if nbrs[row_idx, col_idx] <= threshold:
            filtered_test_ind.append(row_idx)
            filtered_ctrl_ind.append(col_idx)

    # contstruct dataframe if no dataframe is provided
    if data is None:
        data = {}
        filtered_test = test[filtered_test_ind]
        filtered_ctrl = ctrl[filtered_ctrl_ind]
        for i in range(test.shape[1]):
            data[f"feature_{i}"] = np.concatenate(
                (filtered_test[:, i], filtered_ctrl[:, i]), axis=0
            )
        data["status"] = np.concatenate(
            (np.ones(len(filtered_test_ind)), np.zeros(len(filtered_ctrl_ind))), axis=0
        )
        return pd.DataFrame(data)
    else:
        # determine selected rows
        df_test_idx = np.argwhere(df_test_rows.to_numpy()).flatten()
        df_ctrl_idx = np.argwhere(df_ctrl_rows.to_numpy()).flatten()
        all_idx = np.concatenate(
            (df_test_idx[filtered_test_ind], df_ctrl_idx[filtered_ctrl_ind]), axis=0
        )
        return data.iloc[all_idx]


def _mahalonobis_dist(arr1, arr2):
    """
    Calculate the Mahalonobis distance between two 2d arrays along the first axis.

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
    v_inv = np.linalg.inv(np.cov(np.concatenate((arr1, arr2), axis=0).T, ddof=0))
    diff = arr1[:, np.newaxis] - arr2[np.newaxis, :]
    nbrs = np.sqrt(np.sum((diff @ v_inv) * diff, axis=2))
    return nbrs
