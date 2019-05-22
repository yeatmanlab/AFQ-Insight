from __future__ import absolute_import, division, print_function

import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

from .transform import shuffle_group

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
def get_random_forest_group_scores(
    x,
    y,
    group_labels,
    all_label_sets,
    _type="classifier",
    n_splits=1000,
    test_size=0.3,
    ss_random_state=None,
    rf_random_state=None,
    rf_n_estimators=100,
    rf_criterion=None,
    rf_max_depth=None,
):
    """Get scores for each group using a form of feature elimination

    For `n_split` trials, we will train a random forest on all features.
    We will record a "score" based on this fit. Then, we will shuffle each
    feature group and record the relative difference in score between the
    shuffled and un-shuffled feature matrices. A larger difference in score
    indicates that the group has a larger importance. For regression, "score"
    is the MSE and an increase in score after shuffling indicates that a
    feature group was more important. For classification, "score" is the
    ROC AUC and a decrease in score after shuffling indicates that a feature
    group was more important.

    After all trials, the relative differences in scores are averaged and
    sorted before output.

    Parameters
    ----------
    x : numpy.ndarray

    y : numpy.ndarray

    group_labels : sequence of tuples
        tuples of feature labels at any level(s) of the MultiIndex for `x`

    all_label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    _type : 'classifier' or 'regressor'
        Type of random forest to use: classifier or regressor
        Default: 'classifier'

    n_splits : int
        Number of test/train splits to use to average scores over
        Default: 100

    test_size : float
        Test size for the test/train splits
        Default: 0.3

    ss_random_state : int or None
        Random state for the test/train shuffle splits
        Default: None

    rf_random_state : int or None
        Random state for the random forests
        Default: None

    rf_n_estimators : int
        Number of trees in the random forest
        Default: 100

    rf_criterion : string
        Splitting criterion to use in random forest
        Default: 'gini' if _type is 'classifier', 'mse' if _type is 'regressor'

    rf_max_depth : int
        Maximum depth of a tree in the random forest
        Default: None

    Returns
    -------
    importance : list of two-tuples
        A list of two-tuples, where the elements of each tuple are the
        feature labels and their associated scores. The list is sorted
        in descending order by score.
    """
    if rf_criterion is None:
        rf_criterion = "gini" if _type == "classifier" else "mse"

    if _type == "classifier":
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            criterion=rf_criterion,
            max_depth=rf_max_depth,
            random_state=rf_random_state,
        )

        get_score = roc_auc_score
    elif _type == "regressor":
        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            criterion=rf_criterion,
            max_depth=rf_max_depth,
            random_state=rf_random_state,
        )

        get_score = mean_squared_error
    else:
        raise ValueError('`_type` must be either "classifier" or "regressor".')

    ss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=ss_random_state
    )

    scores = defaultdict(list)

    # crossvalidate the scores on a number of different random splits of data
    for train_idx, test_idx in tqdm(ss.split(x, y), total=ss.get_n_splits()):
        train_idx_bs = np.random.choice(train_idx, size=len(train_idx))
        test_idx_bs = np.random.choice(test_idx, size=len(test_idx))
        x_train, x_test = x[train_idx_bs], x[test_idx_bs]
        y_train, y_test = y[train_idx_bs], y[test_idx_bs]
        rf.fit(x_train, y_train)
        score = get_score(y_test, rf.predict(x_test))
        for label in group_labels:
            x_shuffled = shuffle_group(x_test, label, all_label_sets)
            shuffled_score = get_score(y_test, rf.predict(x_shuffled))
            if _type == "classifier":
                scores[label].append((score - shuffled_score) / score)
            else:
                scores[label].append((shuffled_score - score) / score)

    importance = sorted(
        [(feat, np.mean(score)) for feat, score in scores.items()],
        key=lambda s: s[1],
        reverse=True,
    )

    return importance
