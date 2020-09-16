"""
Generate samples of synthetic data sets or extract AFQ data
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import os.path as op
import pandas as pd
from collections import namedtuple
from shutil import copyfile
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle

from .transform import AFQFeatureTransformer

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
def load_afq_data(
    workdir,
    target_cols,
    binary_positives=None,
    fn_nodes="nodes.csv",
    fn_subjects="subjects.csv",
    scale_x=False,
    add_bias_feature=True,
):
    """Load AFQ data from CSV, transform it, return feature matrix and target

    Parameters
    ----------
    workdir : str
        Directory in which to find the AFQ csv files

    target_cols : list of strings
        List of column names in subjects csv file to use as target variables

    binary_positives : list or dict of string values, or None, default=None
        If supplied, use these values as the positive value in a binary
        classification problem. If this is a list, it must have the same
        length as `target cols`. If this is a dict, it must have a key
        for each item in `target_cols`. If None, do not use a binary mapping
        (e.g. for a regression problem).

    fn_nodes : str, default='nodes.csv'
        Filename for the nodes csv file.

    fn_subjects : str, default='subjects.csv'
        Filename for the subjects csv file.

    scale_x : bool, default=False
        If True, center each feature to have zero mean and scale it to have
        unit variance.

    add_bias_feature : bool, default=True
        If True, add a bias (i.e. intercept) feature to the feature matrix
        and return the bias index with the results.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        x - the feature matrix
        y - the target array
        groups - group indices for each feature group
        columns - multi-indexed columns of x
        bias_index - index of the bias feature column in x

    See Also
    --------
    transform.FeatureTransformer
    """
    workdir = op.abspath(workdir)
    fn_nodes = op.join(workdir, fn_nodes)
    fn_subjects = op.join(workdir, fn_subjects)

    nodes = pd.read_csv(fn_nodes)
    targets = pd.read_csv(fn_subjects, index_col="subjectID").drop(
        ["Unnamed: 0"], axis="columns"
    )

    y = targets[target_cols]

    if binary_positives is not None:
        if not isinstance(binary_positives, dict):
            binary_positives = {
                key: val for key, val in zip(target_cols, binary_positives)
            }

        for col in y.columns:
            y.loc[:, col] = y[col].map(lambda c: int(c == binary_positives[col])).values

    transformer = AFQFeatureTransformer()
    x, groups, columns, bias_index = transformer.transform(
        nodes, add_bias_feature=add_bias_feature
    )

    if scale_x:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x[:, bias_index] = 1.0

    AFQData = namedtuple("AFQData", "x y groups columns bias_index")

    return AFQData(x=x, y=y, groups=groups, columns=columns, bias_index=bias_index)


def make_sparse_group_classification(
    n_samples=100,
    n_groups=20,
    n_informative_groups=2,
    n_features_per_group=20,
    n_informative_per_group=2,
    n_redundant_per_group=2,
    n_repeated_per_group=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    useful_indices=False,
    random_state=None,
):
    """Generate a random n-class sparse group classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Prior to shuffling, ``X`` stacks a number of these primary "informative"
    features, "redundant" linear combinations of these, "repeated" duplicates
    of sampled features, and arbitrary noise for and remaining features.
    This method uses sklearn.datasets.make_classification to construct a
    giant unshuffled classification problem of size
    ``n_groups * n_features_per_group`` and then distributes the returned
    features to each group. It then optionally shuffles each group.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_groups : int, optional (default=10)
        The number of feature groups.

    n_informative_groups : int, optional (default=2)
        The total number of informative groups. All other groups will be
        just noise.

    n_features_per_group : int, optional (default=20)
        The total number of features_per_group. These comprise `n_informative`
        informative features, `n_redundant` redundant features, `n_repeated`
        duplicated features and `n_features-n_informative-n_redundant-
        n_repeated` useless features drawn at random.

    n_informative_per_group : int, optional (default=2)
        The number of informative features_per_group. Each class is composed
        of a number of gaussian clusters each located around the vertices of a
        hypercube in a subspace of dimension `n_informative_per_group`. For
        each cluster, informative features are drawn independently from
        N(0, 1) and then randomly linearly combined within each cluster in
        order to add covariance. The clusters are then placed on the vertices
        of the hypercube.

    n_redundant_per_group : int, optional (default=2)
        The number of redundant features per group. These features are
        generated as random linear combinations of the informative features.

    n_repeated_per_group : int, optional (default=0)
        The number of duplicated features per group, drawn randomly from the
        informative and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if `len(weights) == n_classes - 1`,
        then the last class weight is automatically inferred.
        More than `n_samples` samples may be returned if the sum of `weights`
        exceeds 1.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged. Larger
        values introduce noise in the labels and make the classification
        task harder.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    useful_indices : boolean, optional (default=False)
        If True, a boolean array indicating useful features is returned

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for class membership of each sample.
    groups : array of shape [n_features]
        The group number for each feature
    indices : array of shape [n_features]
        A boolean array indicating which features are useful. Returned only if `useful_indices` is True.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See also
    --------
    make_classification: non-group-sparse version
    make_blobs: simplified variant
    make_multilabel_classification: unrelated generator for multilabel tasks
    """
    generator = check_random_state(random_state)

    total_features = n_groups * n_features_per_group
    total_informative = n_informative_groups * n_informative_per_group
    total_redundant = n_informative_groups * n_redundant_per_group
    total_repeated = n_informative_groups * n_repeated_per_group

    # Generate a big classification problem for the total number of features
    # The `shuffle` argument is False so that the feature matrix X has
    # features stacked in the order: informative, redundant, repeated, useless
    X, y = make_classification(
        n_samples=n_samples,
        n_features=total_features,
        n_informative=total_informative,
        n_redundant=total_redundant,
        n_repeated=total_repeated,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        weights=weights,
        flip_y=flip_y,
        class_sep=class_sep,
        hypercube=hypercube,
        shift=shift,
        scale=scale,
        shuffle=False,
        random_state=generator,
    )

    total_useful = total_informative + total_redundant + total_repeated
    idx = np.arange(total_features) < total_useful

    # Evenly distribute the first `n_informative_groups * n_features_per_group`
    # features into the first `n_informative_groups` groups
    n_info_grp_features = n_informative_groups * n_features_per_group
    idx_range = np.arange(n_info_grp_features)

    idx_map_consolidated_2_grouped = (
        np.concatenate(
            [np.arange(0, n_info_grp_features, n_informative_groups)]
            * n_informative_groups
        )
        + idx_range // n_features_per_group
    )

    X = np.concatenate(
        [X[:, idx_map_consolidated_2_grouped], X[:, n_info_grp_features:]], axis=1
    )

    if useful_indices:
        idx = np.concatenate(
            [idx[idx_map_consolidated_2_grouped], idx[n_info_grp_features:]]
        )

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Permute the groups, maintaining the order within them group_idx_map
        # maps feature indices to group indices. The group order is random
        # but all features in a single group are adjacent
        group_idx_map = np.concatenate(
            [
                np.ones(n_features_per_group, dtype=np.int32) * i
                for i in np.random.choice(
                    np.arange(n_groups), size=n_groups, replace=False
                )
            ]
        )

        permute_group_map = (
            np.concatenate(
                [
                    np.random.choice(
                        np.arange(n_features_per_group),
                        size=n_features_per_group,
                        replace=False,
                    )
                    for _ in range(n_groups)
                ]
            )
            + group_idx_map * n_features_per_group
        )

        X = X[:, permute_group_map]

        if useful_indices:
            idx = idx[permute_group_map]

    if useful_indices:
        return X, y, group_idx_map, idx
    else:
        return X, y, group_idx_map


@registered
def output_beta_to_afq(
    beta_hat,
    columns,
    workdir_in,
    workdir_out,
    fn_nodes_in="nodes.csv",
    fn_subjects_in="subjects.csv",
    fn_nodes_out="nodes.csv",
    fn_subjects_out="subjects.csv",
    scale_beta=False,
):
    """Load AFQ data from CSV, transform it, return feature matrix and target

    Parameters
    ----------
    workdir_in : str
        Directory in which to find the input AFQ csv files

    workdir_out : str
        Directory in which to save the output AFQ csv files

    fn_nodes_in : str, default='nodes.csv'
        Filename for the input nodes csv file.

    fn_subjects_in : str, default='subjects.csv'
        Filename for the input subjects csv file.

    fn_nodes_out : str, default='nodes.csv'
        Filename for the output nodes csv file.

    fn_subjects_out : str, default='subjects.csv'
        Filename for the output subjects csv file.

    scale_beta : bool, default=False
        If True, scale the beta coefficients to have the same mean and
        variance as other values for the same metric and tract.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        nodes_file - output nodes csv file path
        subjects_file - output subjects csv file path
    """
    workdir_in = op.abspath(workdir_in)
    fn_nodes_in = op.join(workdir_in, fn_nodes_in)
    fn_subjects_in = op.join(workdir_in, fn_subjects_in)

    workdir_out = op.abspath(workdir_out)
    fn_nodes_out = op.join(workdir_out, fn_nodes_out)
    fn_subjects_out = op.join(workdir_out, fn_subjects_out)

    if op.samefile(workdir_in, workdir_out):
        raise ValueError(
            "output directory equals input directory, please "
            "output to a different directory to avoid "
            "overwriting your files."
        )

    df_nodes = pd.read_csv(fn_nodes_in)

    df_beta = pd.DataFrame(columns=columns)
    df_beta.loc["value"] = beta_hat
    df_beta = df_beta.transpose()
    df_beta = df_beta.unstack(level="metric")
    df_beta.columns = [t[-1] for t in df_beta.columns]
    df_beta.reset_index(inplace=True)
    df_beta["subjectID"] = "beta_hat"
    df_beta = df_beta[df_nodes.columns]

    if scale_beta:
        # For each tract-metric scale the beta_hat values to have the
        # same mean and standard deviation as the subjects' tract-metric
        # values. Suppose we have beta values given by x_i with mean `b_mean`
        # and standard deviation `b_std` and we want to arrive at a similar
        # set with mean `f_mean` and standard deviation `f_std`:
        #     y_i = f_mean + (x_i - b_mean) * f_std / b_std
        for tract in df_nodes["tractID"].unique():
            f_mean = (
                df_nodes.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_nodes["tractID"] == tract]
                .mean()
            )

            f_std = (
                df_nodes.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_nodes["tractID"] == tract]
                .std()
            )

            b_mean = (
                df_beta.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_beta["tractID"] == tract]
                .mean()
            )

            b_std = (
                df_beta.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_beta["tractID"] == tract]
                .std()
            )

            metrics = b_mean.index
            df_beta.loc[df_beta["tractID"] == tract, metrics] = f_mean + (
                df_beta.loc[df_beta["tractID"] == tract, metrics] - b_mean
            ) * f_std.divide(b_std).replace([np.inf, -np.inf], 1)

    df_nodes = pd.concat([df_nodes, df_beta], axis="rows", ignore_index=True)
    df_nodes.to_csv(fn_nodes_out, index=False)

    df_subjects = pd.read_csv(fn_subjects_in, index_col=0)
    subject_row = {key: "" for key in df_subjects.columns}
    subject_row["subjectID"] = "beta_hat"
    df_subjects.loc[len(df_subjects)] = subject_row
    df_subjects.to_csv(fn_subjects_out, index=True)

    fn_streamlines_in = op.join(workdir_in, "streamlines.json")
    fn_streamlines_out = op.join(workdir_out, "streamlines.json")
    copyfile(fn_streamlines_in, fn_streamlines_out)

    fn_params_in = op.join(workdir_in, "params.json")
    fn_params_out = op.join(workdir_out, "params.json")
    copyfile(fn_params_in, fn_params_out)

    OutputFiles = namedtuple("OutputFiles", "nodes_file subjects_file")

    return OutputFiles(nodes_file=fn_nodes_out, subjects_file=fn_subjects_out)
