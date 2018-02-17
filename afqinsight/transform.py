from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
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
        _ = iter(obj)  # noqa F841
    except TypeError:
        return False
    else:
        return True


@registered
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

    def transform(self, x, *_):
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

            return x[:, mask]
        else:
            return x

    def fit(self, *_):
        return self


@registered
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
    mask = np.logical_not(set(remove_label) <= label_sets)
    return np.copy(x[:, mask])


@registered
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
        mask = np.logical_or(mask, np.logical_not(set(label) <= label_sets))

    return np.copy(x[:, mask])


@registered
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


@registered
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


@registered
def shuffle_group(x, label, label_sets, random_seed=None):
    """Shuffle all elements for group `remove_idx`

    Parameters
    ----------
    x : ndarray
        Feature matrix

    label : string or sequence
        label for any level of the MultiIndex columns of `x`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `x`

    random_seed : int, optional
        Random seed for group shuffling
        Default: None

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

    random_state = None
    if random_seed is not None:
        # Save random state to restore later
        random_state = np.random.get_state()
        # Set the random seed
        np.random.seed(random_seed)

    np.random.shuffle(section)

    if random_seed is not None:
        # Restore random state after shuffling
        np.random.set_state(random_state)

    out[:, mask] = section.reshape(section_shape)
    return out


@registered
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
        lambda c: set(c),
        col_vals
    )))

    return col_sets


@registered
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

    def transform(self, x, *_):
        input_provided = (
            self.labels_by_importance is not None
            and self.all_labels is not None
        )

        if input_provided:
            out = select_groups(
                x, self.labels_by_importance[:self.top_n], self.all_labels
            )
            return out
        else:
            return x

    def fit(self, *_):
        return self
