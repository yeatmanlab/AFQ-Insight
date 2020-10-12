"""Extract, transform, select, and shuffle AFQ data."""
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas import DataFrameMapper

from .utils import CANONICAL_TRACT_NAMES

__all__ = [
    "AFQDataFrameMapper",
    "GroupExtractor",
    "remove_group",
    "remove_groups",
    "select_group",
    "select_groups",
    "shuffle_group",
    "multicol2sets",
    "multicol2dicts",
    "sort_features",
    "TopNGroupsExtractor",
    "beta_hat_by_groups",
    "unfold_beta_hat_by_metrics",
]


class AFQDataFrameMapper(DataFrameMapper):
    """Map pandas dataframe to sklearn feature matrix.

    This object first converts an AFQ nodes.csv dataframe into a feature
    matrix with rows corresponding to subjects and columns corresponding to
    tract profile values. It interpolates along tracts to fill missing values
    and then maps the dataframe onto a 2D feature matrix for ingestion into
    sklearn-compatible estimators. It also maintains attributes for the
    subject index, feature names, and groups of features.

    Parameters
    ----------
    df_mapper_params : kwargs, default=dict(features=[], default=None)
        Keyword arguments passed to sklearn_pandas.DataFrameMapper. You will
        probably not need to change these defaults.

    pd_interpolate_params : kwargs, default=dict(method="linear", limit_direction="both", limit_area="inside")
        Keyword arguments passed to pandas.DataFrame.interpolate. Missing
        values are interpolated within the tract profile so that no data is
        used from other subjects, tracts, or metrics, minimizing the chance
        of train/test leakage. You will probably not need to change these
        defaults.

    Attributes
    ----------
    subjects_ : list
        List of subject IDs retrieved from pandas dataframe index.

    groups_ : list of numpy.ndarray
        List of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``.

    feature_names_ : list of tuples
        List of feature column names.
    """

    def __init__(
        self,
        dataframe_mapper_kwargs={"features": [], "default": None},
        pd_interpolate_kwargs={
            "method": "linear",
            "limit_direction": "both",
            "limit_area": "inside",
        },
    ):
        self.subjects_ = []
        self.groups_ = []
        self.pd_interpolate_kwargs = pd_interpolate_kwargs
        super().__init__(**dataframe_mapper_kwargs)

    def _preprocess(self, X, set_attributes=True):
        # We'd like to interpolate the missing values, but first we need to
        # structure the data frame so that it does not interpolate from other
        # subjects, tracts, or metrics. It should only interpolate from nearby
        # nodes. So we want the nodeID as the row index and all the other
        # stuff as columns . After that we can interpolate along each column.
        by_node_idx = pd.pivot_table(
            data=X.melt(id_vars=["subjectID", "tractID", "nodeID"], var_name="metric"),
            index="nodeID",
            columns=["metric", "tractID", "subjectID"],
            values="value",
        )

        # Interpolate the missing values, using self.pd_interpolate_kwargs
        interpolated = by_node_idx.interpolate(**self.pd_interpolate_kwargs)

        # Now we have the NaN values filled in, we want to structure the nodes
        # dataframe as a feature matrix with one row per subject and one
        # column for each combination of metric, tractID, and nodeID
        features = interpolated.stack(["subjectID", "tractID", "metric"]).unstack(
            ["metric", "tractID", "nodeID"]
        )

        # We're almost there. It'd be nice if the multi-indexed columns were
        # ordered well. So let's reorder the columns
        new_columns = pd.MultiIndex.from_product(
            features.columns.levels, names=["metric", "tractID", "nodeID"]
        )

        features = features.loc[:, new_columns]

        # Lastly, there may still be some NaN values. After interpolating
        # above, the only NaN values left should be the ones created after
        # stacking and unstacking due to a subject missing an entire tract. In
        # this case, we do not fill these values and instead recommend that
        # users use the sklearn.impute.SimpleImputer

        if set_attributes:
            # Construct bundle group membership
            metric_level = features.columns.names.index("metric")
            tract_level = features.columns.names.index("tractID")
            n_tracts = len(features.columns.levels[tract_level])
            bundle_group_membership = np.array(
                features.columns.codes[metric_level].astype(np.int64) * n_tracts
                + features.columns.codes[tract_level].astype(np.int64),
                dtype=np.int64,
            )

            # Done, now let's extract the subject IDs from the index
            self.subjects_ = features.index.tolist()

            self.groups_ = [
                np.where(bundle_group_membership == gid)[0]
                for gid in np.unique(bundle_group_membership)
            ]

        return features

    def fit(self, X, y=None):
        """Fit a transform from the given dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to fit

        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target values. Unused in this transformer
        """
        features = self._preprocess(X, set_attributes=True)
        return super().fit(features, y)

    def transform(self, X):
        """Transform the input data.

        This assumes that ``fit`` or ``fit_transform`` has already been called.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to transform
        """
        features = self._preprocess(X, set_attributes=False)
        return super().transform(features)

    def fit_transform(self, X, y=None):
        """Fit a transform from the given dataframe and apply directly to given data.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to fit

        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target values. Unused in this transformer
        """
        features = self._preprocess(X, set_attributes=True)
        return super().fit_transform(features, y)

    @property
    def feature_names_(self):
        """Return the feature names."""
        return self.transformed_names_


def isiterable(obj):
    """Return True if obj is an iterable, False otherwise."""
    try:
        _ = iter(obj)  # noqa F841
    except TypeError:
        return False
    else:
        return True


class GroupExtractor(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group extractor.

    Given a sequence of all group indices and a subsequence of desired
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that are in the desired subgroups.

    Parameters
    ----------
    extract : numpy.ndarray or int, optional
        subsequence of desired groups to extract from feature matrix

    groups : numpy.ndarray or int, optional
        all group indices for feature matrix
    """

    def __init__(self, extract=None, groups=None):
        self.extract = extract
        self.groups = groups

    def transform(self, X, *_):
        """Transform the input data.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix
        """
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
        """Fit a transform from the given data. This is a no-op."""
        return self


def remove_group(X, remove_label, label_sets):
    """Remove all columns for group specified by ``remove_label``.

    Parameters
    ----------
    X : ndarray
        Feature matrix

    remove_label : string or sequence
        label for any level of the MultiIndex columns of `X`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `X`

    Returns
    -------
    ndarray
        new feature matrix with group represented by `remove_label` removed

    See Also
    --------
    multicol2sets
        function to convert a pandas MultiIndex into the sequence of sets
        expected for the parameter `label_sets`
    """
    mask = np.logical_not(set(remove_label) <= label_sets)
    if len(X.shape) == 2:
        return np.copy(X[:, mask])
    elif len(X.shape) == 1:
        return np.copy(X[mask])
    else:
        raise ValueError("`X` must be a one- or two-dimensional ndarray.")


def remove_groups(X, remove_labels, label_sets):
    """Remove all columns for groups specified by ``remove_labels``.

    Parameters
    ----------
    X : ndarray
        Feature matrix

    remove_labels : sequence
        labels for any level of the MultiIndex columns of `X`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `X`

    Returns
    -------
    ndarray
        new feature matrix with group represented by `remove_label` removed

    See Also
    --------
    multicol2sets
        function to convert a pandas MultiIndex into the sequence of sets
        expected for the parameter `label_sets`
    """
    mask = np.zeros_like(label_sets, dtype=np.bool)
    for label in remove_labels:
        mask = np.logical_or(mask, np.logical_not(set(label) <= label_sets))

    if len(X.shape) == 2:
        return np.copy(X[:, mask])
    elif len(X.shape) == 1:
        return np.copy(X[mask])
    else:
        raise ValueError("`X` must be a one- or two-dimensional ndarray.")


def select_group(X, select_label, label_sets):
    """Select all columns for group specified by ``select_label``.

    Parameters
    ----------
    X : ndarray
        Feature matrix

    select_label : string or sequence
        label for any level of the MultiIndex columns of `X`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `X`

    Returns
    -------
    ndarray
        new feature matrix with only the group represented by `select_label`

    See Also
    --------
    multicol2sets
        function to convert a pandas MultiIndex into the sequence of sets
        expected for the parameter `label_sets`
    """
    mask = set(select_label) <= label_sets
    if len(X.shape) == 2:
        return np.copy(X[:, mask])
    elif len(X.shape) == 1:
        return np.copy(X[mask])
    else:
        raise ValueError("`X` must be a one- or two-dimensional ndarray.")


def select_groups(X, select_labels, label_sets):
    """Select all columns for groups specified by ``select_labels``.

    Parameters
    ----------
    X : ndarray
        Feature matrix

    select_labels : sequence
        labels for any level of the MultiIndex columns of `X`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `X`

    Returns
    -------
    ndarray
        new feature matrix with only the group represented by `select_label`

    See Also
    --------
    multicol2sets
        function to convert a pandas MultiIndex into the sequence of sets
        expected for the parameter `label_sets`
    """
    mask = np.zeros_like(label_sets, dtype=np.bool)
    for label in select_labels:
        mask = np.logical_or(mask, set(label) <= label_sets)

    if len(X.shape) == 2:
        return np.copy(X[:, mask])
    elif len(X.shape) == 1:
        return np.copy(X[mask])
    else:
        raise ValueError("`X` must be a one- or two-dimensional ndarray.")


def shuffle_group(X, label, label_sets, random_seed=None):
    """Shuffle all elements for group specified by ``label``.

    Parameters
    ----------
    X : ndarray
        Feature matrix

    label : string or sequence
        label for any level of the MultiIndex columns of `X`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `X`

    random_seed : int, optional
        Random seed for group shuffling
        Default: None

    Returns
    -------
    ndarray
        new feature matrix with all elements of group `shuffle_idx` permuted
    """
    out = np.copy(X)
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


def multicol2sets(columns, tract_symmetry=True):
    """Convert a pandas MultiIndex to an array of sets.

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
    col_vals = columns.to_numpy()

    if tract_symmetry:
        tract_idx = columns.names.index("tractID")

        bilateral_symmetry = {
            tract: tract.replace("Left ", "").replace("Right ", "")
            for tract in columns.levels[tract_idx]
        }

        col_vals = np.array([x + (bilateral_symmetry[x[tract_idx]],) for x in col_vals])

    col_sets = np.array(list(map(lambda c: set(c), col_vals)))

    return col_sets


def multicol2dicts(columns, tract_symmetry=True):
    """Convert a pandas MultiIndex to an array of dicts.

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
    col_dicts : numpy.ndarray
        An array of dicts containing the tuples of the input MultiIndex
    """
    col_vals = columns.to_numpy()
    col_names = columns.names

    if tract_symmetry:
        tract_idx = columns.names.index("tractID")

        bilateral_symmetry = {
            tract: tract.replace("Left ", "").replace("Right ", "")
            for tract in columns.levels[tract_idx]
        }

        col_vals = np.array([x + (bilateral_symmetry[x[tract_idx]],) for x in col_vals])

        col_names = list(col_names) + ["symmetrized_tractID"]

    col_dicts = np.array([dict(zip(col_names, vals)) for vals in col_vals])

    return col_dicts


def sort_features(features, scores):
    """Sort features by importance.

    Parameters
    ----------
    features : sequence of features
        Sequence of features, can be the returned values from multicol2sets
        or multicol2dicts

    scores : sequence of scores
        importance scores for each feature

    Returns
    -------
    list
        Sorted list of columns and scores
    """
    res = sorted(
        [(feat, score) for feat, score in zip(features, scores)],
        key=lambda s: np.abs(s[1]),
        reverse=True,
    )

    return res


class TopNGroupsExtractor(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group extractor.

    Given a sequence of all group indices and a subsequence of desired
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that are in the desired subgroups.

    Parameters
    ----------
    extract : sequence of tuples of labels
        labels for any level of the MultiIndex columns of `X`

    label_sets : ndarray of sets
        Array of sets of labels for each column of `X`
    """

    def __init__(self, top_n=10, labels_by_importance=None, all_labels=None):
        self.top_n = top_n
        self.labels_by_importance = labels_by_importance
        self.all_labels = all_labels

    def transform(self, X, *_):
        """Transform the input data.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix
        """
        input_provided = (
            self.labels_by_importance is not None and self.all_labels is not None
        )

        if input_provided:
            out = select_groups(
                X, self.labels_by_importance[: self.top_n], self.all_labels
            )
            return out
        else:
            return X

    def fit(self, *_):
        """Fit a transform from the given data. This is a no-op."""
        return self


def beta_hat_by_groups(beta_hat, columns, drop_zeros=False):
    """Transform one-dimensional beta_hat array into OrderedDict.

    Organize by tract-metric groups

    Parameters
    ----------
    beta_hat : np.ndarray
        one-dimensional array of feature coefficients

    columns : pd.MultiIndex
        MultiIndex columns of the feature matrix

    drop_zeros : bool, default=False
        If True, only include betas for which there are non-zero values

    Returns
    -------
    OrderedDict
        Two-level ordered dict with beta_hat coefficients, ordered first
        by tract and then by metric

    See Also
    --------
    AFQFeatureTransformer
        Transforms AFQ csv files into feature matrix. Use this to create
        the `columns` input.
    """
    betas = OrderedDict()

    label_sets = multicol2sets(columns, tract_symmetry=False)

    for tract in columns.levels[columns.names.index("tractID")]:
        all_metrics = select_group(
            x=beta_hat, select_label=(tract,), label_sets=label_sets
        )
        if not drop_zeros or any(all_metrics != 0):
            betas[tract] = OrderedDict()
            for metric in columns.levels[columns.names.index("metric")]:
                x = select_group(
                    x=beta_hat, select_label=(tract, metric), label_sets=label_sets
                )
                if not drop_zeros or any(x != 0):
                    betas[tract][metric] = x

    return betas


def unfold_beta_hat_by_metrics(beta_hat, columns, tract_names=None):
    """Transform one-dimensional beta_hat array into OrderedDict.

    Organize by tract-metric groups

    Parameters
    ----------
    beta_hat : np.ndarray
        one-dimensional array of feature coefficients

    columns : pd.MultiIndex
        MultiIndex columns of the feature matrix

    tract_names : list or None, default=None
        Names of the tracts. If None, use utils.CANONICAL_TRACT_NAMES

    Returns
    -------
    OrderedDict
        Single-level ordered dict with beta_hat coefficients.
        The keys are the metrics and the values are the unfolded beta_hat
        coefficients

    See Also
    --------
    AFQFeatureTransformer
        Transforms AFQ csv files into feature matrix. Use this to create
        the `columns` input.

    beta_hat_by_groups
        Returns a two-level ordered dict instead of "unfolding" the tracts
    """
    betas = OrderedDict()

    betas_by_groups = beta_hat_by_groups(beta_hat, columns, drop_zeros=False)

    if tract_names is not None:
        tracts = tract_names
    else:
        tracts = CANONICAL_TRACT_NAMES

    for metric in columns.levels[columns.names.index("metric")]:
        betas[metric] = []
        for tract in tracts:
            betas[metric].append(betas_by_groups[tract][metric])

        betas[metric] = np.concatenate(betas[metric])

    return betas
