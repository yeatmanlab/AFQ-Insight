import numpy as np
from scipy.optimize import linear_sum_assignment


def mahalonobis_dist_match(X, test_idx, threshold=0.2):
    """
    Perform Mahalanobis distance matching (MDM).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The feature samples.
    test_idx : array-like
        indicies in the first dimension of X which correspond to test
        subjects. All other subjects will be control subjects.
    threshold : float
        Mahalnobis distance beyond which matches are not accepted.

    Returns
    -------
    filtered_test_ind : array-like
        Indices into X of successfully matched test subjects
    filtered_ctrl_ind : array-like
        Indices into X of matched control subjects,
        this array will be the same length as filtered_test_ind
    """

    # calculate mahalonobis distance between all subjects
    v_inv = np.linalg.inv(np.cov(X.T, ddof=0))
    test_X = X[test_idx]
    ctrl_X = np.delete(X, test_idx, axis=0)
    diff = test_X[:, np.newaxis] - ctrl_X[np.newaxis, :]
    nbrs = np.sqrt(np.sum((diff @ v_inv) * diff, axis=2))

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
