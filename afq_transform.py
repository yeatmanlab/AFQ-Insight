import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import ShuffleSplit

class AFQFeatureTransformer(object):
    """Transforms AFQ data from an input dataframe into a feature matrix

    Using an object interface for eventual inclusion into sklearn Pipelines
    """
    def __init__(self):
        pass

    def transform(self, df, extrapolate=False):
        """Transforms an AFQ dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            input AFQ dataframe

        extrapolate : boolean
            If True, use column-wise linear interpolation/extrapolation
            for missing metric values. If False, use pandas built-in
            `interpolate` method, which uses interpolation for interior points
            and forward(back)-fill for exterior points.

        Returns
        -------
        X : numpy.ndarray
            feature matrix

        groups : numpy.ndarray
            group membership for each feature (column) of X

        columns : pandas.MultiIndex
            multi-indexed columns of X
        """
        # We'd like to interpolate the missing values, but first we need to
        # structure the data frame so that it does not interpolate from other
        # subjects, tracts, or metrics. It should only interpolate from nearby
        # nodes. So we want the nodeID as the row index and all the other
        # stuff as columns . After that we can interpolate along each column.
        by_node_idx = pd.pivot_table(
            data=df.melt(
                id_vars=['subjectID', 'tractID', 'nodeID'],
                var_name='metric'
            ),
            index='nodeID',
            columns=['metric', 'tractID', 'subjectID'],
            values='value'
        )

        if not extrapolate:
            # We could use the built-in `.interpolate` method. This has some
            # unexpected behavior when the NaN values are at the beginning or
            # end of a series. For NaN values at the end of the series, it
            # forward fills the most recent valid value. And for NaN values
            # at the beginning of the series, it back fills the next valid
            # value. For now, we accept this behavior because the execution
            # time is much, much faster than doing column-wise linear
            # extrapolation
            interpolated = by_node_idx.interpolate(
                method='linear', limit_direction='both'
            )
        else:
            # Instead, we may want to interpolate NaN values with
            # extrapolation at the end of the node range. But, pandas does
            # not currently support extrapolation
            # See this issue:
            # https://github.com/pandas-dev/pandas/issues/16284
            # And this stalled PR:
            # https://github.com/pandas-dev/pandas/pull/16513
            # Until that's fixed, we can perform the interpolation column by
            # column using the apply method. This is SLOW, but it does the job
            def interp_linear_with_extrap(series):
                """Linearly interpolate a series with extrapolation...

                ...outside the series range
                """
                x = series[~series.isnull()].index.values
                y = series[~series.isnull()].values
                f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                return f(series.index)

            # Apply the interpolation across all columns
            interpolated = by_node_idx.apply(interp_linear_with_extrap)

        # Now we have the NaN values filled in, we want to structure the nodes
        # dataframe as a feature matrix with one row per subject and one
        # column for each combination of metric, tractID, and nodeID
        features = interpolated.stack(
            ['subjectID', 'tractID', 'metric']
        ).unstack(
            ['metric', 'tractID', 'nodeID']
        )

        # We're almost there. It'd be nice if the multi-indexed columns were
        # ordered well. So let's reorder the columns
        new_columns = pd.MultiIndex.from_product(
            features.columns.levels,
            names=['metric', 'tractID', 'nodeID']
        )

        features = features.loc[:, new_columns]

        # Lastly, there may still be some nan values. After interpolating
        # above, the only NaN values left should be the one created after
        # stacking and unstacking due to a subject missing an entire tract.
        # In this case, for each missing column, we take the median value
        # of all other subjects as the fillna value
        features.fillna(features.median(), inplace=True)

        # Construct bundle group membership
        metric_level = features.columns.names.index('metric')
        tract_level = features.columns.names.index('tractID')
        n_tracts = len(features.columns.levels[tract_level])
        bundle_group_membership = np.array(
            features.columns.labels[metric_level].astype(np.int64) * n_tracts
            + features.columns.labels[tract_level].astype(np.int64),
            dtype=np.int64
        )

        return features.values, bundle_group_membership, features.columns


def isiterable(obj):
    """Return True if obj is an iterable, False otherwise."""
    try:
        _ = iter(obj)
    except TypeError:
        return False
    else:
        return True


class GroupExtractor(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group extractor

    Given a sequence of all group indices and a subsequence of desired
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that are in the desired subgroups.

    Parameters
    ----------
    extract : numpy.ndarray or int, optional
        subsequence of desired groups to extract from feature matrix

    groups : numpy.ndarray or int, optional
        all group indices for feature matrix

    Note
    ----
    Following
    http://scikit-learn.org/dev/developers/contributing.html
    We do not do have any parameter validation in __init__. All logic behind
    estimator parameters is done in transform.
    """
    def __init__(self, extract=None, groups=None):
        self.extract = extract
        self.groups = groups

    def transform(self, X, *_):
        if self.groups is not None and self.extract is not None:
            if isiterable(self.extract):
                extract = np.array(self.extract)
            else:
                extract = np.array([self.extract])

            if isiterable(self.groups):
                groups = np.array(self.groups)
            else:
                groups = np.array([self.groups])

            try:
                mask = np.isin(groups, extract)
            except AttributeError:
                mask = np.array([item in extract for item in groups])

            return X[:, mask]
        else:
            return X

    def fit(self, *_):
        return self


def remove_group(x, remove_label, label_sets):
    """Remove all columns for group `remove_idx`

    Parameters
    ----------
    x : ndarray
        Feature matrix

    remove_label : string or sequence
        label for any level of the MultiIndex columns of `x`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    Returns
    -------
    ndarray
        new feature matrix with group represented by `remove_label` removed
    """
    mask = not set(remove_label) <= label_sets
    return np.copy(x[:, mask])


def remove_groups(x, remove_labels, label_sets):
    """Remove all columns for group `remove_idx`

    Parameters
    ----------
    x : ndarray
        Feature matrix

    remove_labels : sequence
        labels for any level of the MultiIndex columns of `x`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    Returns
    -------
    ndarray
        new feature matrix with group represented by `remove_label` removed
    """
    mask = np.zeros_like(label_sets, dtype=np.bool)
    for label in remove_labels:
        mask = np.logical_or(mask, not set(label) <= label_sets)

    return np.copy(x[:, mask])


def select_group(x, select_label, label_sets):
    """Select all columns for group `remove_idx`

    Parameters
    ----------
    x : ndarray
        Feature matrix

    select_label : string or sequence
        label for any level of the MultiIndex columns of `x`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    Returns
    -------
    ndarray
        new feature matrix with only the group represented by `select_label`
    """
    mask = set(select_label) <= label_sets
    return np.copy(x[:, mask])


def select_groups(x, select_labels, label_sets):
    """Select all columns for group `remove_idx`

    Parameters
    ----------
    x : ndarray
        Feature matrix

    select_labels : sequence
        labels for any level of the MultiIndex columns of `x`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    Returns
    -------
    ndarray
        new feature matrix with only the group represented by `select_label`
    """
    mask = np.zeros_like(label_sets, dtype=np.bool)
    for label in select_labels:
        mask = np.logical_or(mask, set(label) <= label_sets)

    return np.copy(x[:, mask])


def shuffle_group(x, label, label_sets):
    """Shuffle all elements for group `remove_idx`

    Parameters
    ----------
    x : ndarray
        Feature matrix

    label : string or sequence
        label for any level of the MultiIndex columns of `x`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    Returns
    -------
    ndarray
        new feature matrix with all elements of group `shuffle_idx` permuted
    """
    out = np.copy(x)
    mask = set(label) <= label_sets
    section = out[:, mask]
    section_shape = section.shape
    section = section.flatten()
    np.random.shuffle(section)
    out[:, mask] = section.reshape(section_shape)
    return out


def multicol2sets(columns, tract_symmetry=True):
    """Convert a pandas MultiIndex to an array of sets

    Parameters
    ----------
    columns : pandas.MultiIndex
        multi-indexed columns used to generate the result

    tract_symmetry : boolean, optional
        If True, then another tract item will be added to each set
        if the set contains a tract containing "Left" or "Right."
        The added tract will be the more general (i.e. symmetrized) name.
        Default: True

    Returns
    -------
    col_sets : numpy.ndarray
        An array of sets containing the tuples of the input MultiIndex
    """
    col_vals = columns.get_values()

    if tract_symmetry:
        tract_idx = columns.names.index('tractID')

        bilateral_symmetry = {
            tract: tract.replace('Left ', '').replace('Right ', '')
            for tract in columns.levels[tract_idx]
        }

        col_vals = np.array([
            x + (bilateral_symmetry[x[tract_idx]], )
            for x in col_vals
        ])

    col_sets = np.array(list(map(
        lambda x: set(x),
        col_vals
    )))

    return col_sets


def get_random_forest_group_scores(
        x, y, group_labels, all_label_sets,
        type='classifier',
        n_splits=100, test_size=0.3,
        ss_random_state=None, rf_random_state=None,
        rf_n_estimators=1000, rf_criterion=None, rf_max_depth=None
):
    """Get scores for each group using a form of feature elimination

    Parameters
    ----------
    x : numpy.ndarray

    y : numpy.ndarray

    group_labels : sequence of tuples
        tuples of feature labels at any level(s) of the MultiIndex for `x`

    all_label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    type : 'classifier' or 'regressor'
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
        Default: 'gini' if type is 'classifier', 'mse' if type is 'regressor'

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
        rf_criterion = 'gini' if type == 'classifier' else 'mse'

    if type == 'classifier':
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            criterion=rf_criterion,
            max_depth=rf_max_depth,
            random_state=rf_random_state
        )

        get_score = roc_auc_score
    elif type == 'regressor':
        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            criterion=rf_criterion,
            max_depth=rf_max_depth,
            random_state=rf_random_state
        )

        get_score = mean_squared_error
    else:
        raise ValueError('`type` must be either "classifier" or "regressor".')

    ss = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=ss_random_state
    )

    scores = defaultdict(list)

    # crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in tqdm(ss.split(x), total=ss.get_n_splits()):
        train_idx_bs = np.random.choice(train_idx, size=len(train_idx))
        test_idx_bs = np.random.choice(test_idx, size=len(test_idx))
        x_train, x_test = x[train_idx_bs], x[test_idx_bs]
        y_train, y_test = y[train_idx_bs], y[test_idx_bs]
        rf.fit(x_train, y_train)
        score = get_score(y_test, rf.predict(x_test))
        for label in group_labels:
            x_shuffled = shuffle_group(x_test, label, all_label_sets)
            shuffled_score = get_score(y_test, rf.predict(x_shuffled))
            if type == 'classifier':
                scores[label].append((score - shuffled_score) / score)
            else:
                scores[label].append((shuffled_score - score) / score)

    importance = sorted(
        [(feat, np.mean(score)) for
         feat, score in scores.items()],
        key=lambda x: x[1],
        reverse=True)

    return importance


class TopNGroupsExtractor(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group extractor

    Given a sequence of all group indices and a subsequence of desired
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that are in the desired subgroups.

    Parameters
    ----------
    extract : sequence of tuples of labels
        labels for any level of the MultiIndex columns of `X`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `X`

    Note
    ----
    Following
    http://scikit-learn.org/dev/developers/contributing.html
    We do not do have any parameter validation in __init__. All logic behind
    estimator parameters is done in transform.
    """
    def __init__(self, top_n=10, labels_by_importance=None, all_labels=None):
        self.top_n = top_n
        self.labels_by_importance = labels_by_importance
        self.all_labels = all_labels

    def transform(self, X, *_):
        input_provided = (
            self.labels_by_importance is not None
            and self.all_labels is not None
        )

        if input_provided:
            out = select_groups(
                X, self.labels_by_importance[:self.top_n], self.all_labels
            )
            return out
        else:
            return X

    def fit(self, *_):
        return self
