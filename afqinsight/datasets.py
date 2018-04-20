"""
Generate samples of synthetic data sets.
"""

import numpy as np
import os.path as op
import pandas as pd

from collections import namedtuple
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.preprocessing import StandardScaler

from .transform import AFQFeatureTransformer

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
def load_afq_data(workdir, target_col, binary_positive=None,
                  fn_nodes='nodes.csv', fn_subjects='subjects.csv',
                  scale_x=True):
    """Load AFQ data from CSV, transform it, return feature matrix and target

    Parameters
    ----------
    workdir : str
        Directory in which to find the AFQ csv files

    target_col : str
        Name of column in the subjects csv file to use as the target variable

    binary_positive : str or None, default=None
        If supplied, use this value as the positive value in a binary
        classification problem. If None, do not use a binary mapping (e.g. for
        a regression problem).

    fn_nodes : str, default='nodes.csv'
        Filename for the nodes csv file.

    fn_subjects : str, default='subjects.csv'
        Filename for the subjects csv file.

    scale_x : bool, default=True
        If True, center each feature to have zero mean and scale it to have
        unit variance.

    Returns
    -------
    collections.namedtuple
        namedtuple with field:
        x - the feature matrix
        y - the target array
        groups - group indices for each feature group
        cols - multi-indexed columns of x

    See Also
    --------
    transform.FeatureTransformer
    """
    workdir = op.abspath(workdir)
    fn_nodes = op.join(workdir, fn_nodes)
    fn_subjects = op.join(workdir, fn_subjects)

    nodes = pd.read_csv(fn_nodes)
    targets = pd.read_csv(
        fn_subjects, index_col='subjectID'
    ).drop(['Unnamed: 0'], axis='columns')

    y = targets[target_col]

    if binary_positive is not None:
        y = y.map(lambda c: int(c == binary_positive)).values

    transformer = AFQFeatureTransformer()
    x, groups, cols = transformer.transform(nodes)

    if scale_x:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    AFQData = namedtuple('AFQData', 'x y groups cols')

    return AFQData(x=x, y=y, groups=groups, cols=cols)


def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions
    """
    if dimensions > 30:
        return np.hstack([rng.randint(2, size=(samples, dimensions - 30)),
                          _generate_hypercube(samples, 30, rng)])
    out = sample_without_replacement(2 ** dimensions, samples,
                                     random_state=rng).astype(dtype='>u4',
                                                              copy=False)
    out = np.unpackbits(out.view('>u1')).reshape((-1, 32))[:, -dimensions:]
    return out


@registered
def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, useful_indices=False, random_state=None):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an `n_informative`-dimensional hypercube with sides of
    length `2*class_sep` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Prior to shuffling, `X` stacks a number of these primary "informative"
    features, "redundant" linear combinations of these, "repeated" duplicates
    of sampled features, and arbitrary noise for and remaining features.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features. These comprise `n_informative`
        informative features, `n_redundant` redundant features, `n_repeated`
        duplicated features and `n_features-n_informative-n_redundant-
        n_repeated` useless features drawn at random.

    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension `n_informative`. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

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

    useful_indices : array of shape [n_features], optional
        A boolean array indicating the usefulness of each feature. An element
        in this array is True if the corresponding feature is either
        informative, redundant, or repeated. It is returned only if indices
        is True.

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
    make_blobs: simplified variant
    make_multilabel_classification: unrelated generator for multilabel tasks
    """
    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated "
                         "features must sum to less than the number of total"
                         " features")
    if 2 ** n_informative < n_classes * n_clusters_per_class:
        raise ValueError("n_classes * n_clusters_per_class must"
                         " be smaller or equal 2 ** n_informative")
    if weights and len(weights) not in [n_classes, n_classes - 1]:
        raise ValueError("Weights specified but incompatible with number "
                         "of classes.")

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    if weights and len(weights) == (n_classes - 1):
        weights = weights + [1.0 - sum(weights)]

    if weights is None:
        weights = [1.0 / n_classes] * n_classes
        weights[-1] = 1.0 - sum(weights[:-1])

    # Distribute samples among clusters by weight
    n_samples_per_cluster = []
    for k in range(n_clusters):
        n_samples_per_cluster.append(int(n_samples * weights[k % n_classes]
                                         / n_clusters_per_class))
    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative,
                                    generator).astype(float)
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.rand(n_clusters, 1)
        centroids *= generator.rand(1, n_informative)

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.randn(n_samples, n_informative)

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster

        A = 2 * generator.rand(n_informative, n_informative) - 1
        X_k[...] = np.dot(X_k, A)  # introduce random covariance

        X_k += centroid  # shift the cluster to a vertex

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = generator.randn(n_samples, n_useless)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.rand(n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.rand(n_features)
    X *= scale

    indices = np.arange(n_features)
    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Randomly permute features
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    if useful_indices:
        n_useful = n_informative + n_redundant + n_repeated
        return X, y, indices < n_useful
    else:
        return X, y


@registered
def make_sparse_group_classification(
        n_samples=100, n_groups=20, n_informative_groups=2,
        n_features_per_group=20, n_informative_per_group=2,
        n_redundant_per_group=2, n_repeated_per_group=0,
        n_classes=2, n_clusters_per_class=2,
        weights=None, flip_y=0.01,
        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
        shuffle=True, useful_indices=False, random_state=None):
    """Generate a random n-class sparse group classification problem.
    This initially creates clusters of points normally distributed (std=1)
    about vertices of an `n_informative`-dimensional hypercube with sides of
    length `2*class_sep` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.
    Prior to shuffling, `X` stacks a number of these primary "informative"
    features, "redundant" linear combinations of these, "repeated" duplicates
    of sampled features, and arbitrary noise for and remaining features.
    This method uses sklearn.datasets.make_classification to construct a
    giant unshuffled classification problem of size
    `n_groups * n_features_per_group` and then distributes the returned
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
    consolidated_class_output = make_classification(
        n_samples=n_samples, n_features=total_features,
        n_informative=total_informative, n_redundant=total_redundant,
        n_repeated=total_repeated, n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class, weights=weights,
        flip_y=flip_y, class_sep=class_sep, hypercube=hypercube, shift=shift,
        scale=scale, shuffle=False, useful_indices=useful_indices,
        random_state=generator
    )

    if useful_indices:
        X, y, idx = consolidated_class_output
    else:
        X, y = consolidated_class_output

    # Evenly distribute the first `n_informative_groups * n_features_per_group`
    # features into the first `n_informative_groups` groups
    n_info_grp_features = n_informative_groups * n_features_per_group
    idx_range = np.arange(n_info_grp_features)

    idx_map_consolidated_2_grouped = np.concatenate([
        np.arange(0, n_info_grp_features, n_informative_groups)
    ] * n_informative_groups) + idx_range // n_features_per_group

    X = np.concatenate([
        X[:, idx_map_consolidated_2_grouped],
        X[:, n_info_grp_features:]
    ], axis=1)

    if useful_indices:
        idx = np.concatenate([
            idx[idx_map_consolidated_2_grouped],
            idx[n_info_grp_features:]
        ])

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Permute the groups, maintaining the order within them group_idx_map
        # maps feature indices to group indices. The group order is random
        # but all features in a single group are adjacent
        group_idx_map = np.concatenate([
            np.ones(n_features_per_group, dtype=np.int32) * i
            for i in np.random.choice(
                np.arange(n_groups), size=n_groups, replace=False
            )
        ])

        permute_group_map = np.concatenate([
            np.random.choice(
                np.arange(n_features_per_group),
                size=n_features_per_group,
                replace=False
            ) for _ in range(n_groups)
        ]) + group_idx_map * n_features_per_group

        X = X[:, permute_group_map]

        if useful_indices:
            idx = idx[permute_group_map]

    if useful_indices:
        return X, y, idx
    else:
        return X, y
