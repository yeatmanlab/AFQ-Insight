"""
Define custom proximal operators for use with copt package
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from itertools import compress

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
class SparseGroupL1(object):
    """Sparse group lasso penalty class for use with openopt/copt package.

    Implements the sparse group lasso penalty [1]_

    .. math::
        (1 - \alpha) * \lambda \displaystyle \sum_{g \in G} || \beta_g ||_2
        + \alpha * \lambda || \beta ||_1

    where :math:`G` is a partition of the features into groups.

    Parameters
    ----------
    alpha : float
        Combination between group lasso and lasso. alpha = 0 gives the group
        lasso and alpha = 1 gives the lasso.

    lambda_ : float
        Regularization parameter, overall strength of regularization.

    groups : numpy.ndarray
        array of non-overlapping indices for each group. For example, if nine
        features are grouped into equal contiguous groups of three, then groups
        would be an nd.array like [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If the
        feature matrix contains a bias or intercept feature, do not include it
        as a group.
        TODO: Change to [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]

    bias_index : int or None, default=None
        If None, regularize all coefficients. Otherwise, this is the index
        of the bias (i.e. intercept) feature, which should not be regularized.
        Exclude this index from the penalty term. And force the proximal
        operator for this index to return the result of the identity function.

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250
    """  # noqa: W605

    def __init__(self, alpha, lambda_, groups, bias_index=None):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.groups = groups
        self.feature2group_map = np.concatenate(
            [
                [grp_idx] * feature_indices.size
                for grp_idx, feature_indices in enumerate(groups)
            ]
        )
        self.sqrt_group_lengths = np.sqrt([grp.size for grp in groups])
        self.bias_index = bias_index

    def __call__(self, x):
        penalty = (
            (1.0 - self.alpha)
            * self.lambda_
            * np.sum([np.linalg.norm(x[g]) for g in self.groups])
        )

        ind = np.ones(len(x), bool)
        if self.bias_index is not None:
            ind[self.bias_index] = False

        penalty += self.alpha * self.lambda_ * np.abs(x[ind]).sum()
        return penalty

    def prox(self, x, step_size):
        """Return the proximal operator of the sparse group lasso penalty

        For the sparse group lasso, we can decompose the penalty into

        .. math::
            P(\beta) = P_1(\beta) + P_2(\beta)

        where :math:`P_2 = \alpha \lambda || \beta ||_1` is the lasso
        penalty and :math:`P_1 = (1 - \alpha) \lambda \displaystyle
        \sum_{g \in G} || \beta_g ||_2` is the group lasso penalty.

        Then the proximal operator is given by

        .. math::
            \text{prox}_{\sigma P_1 + \sigma P_2} (u) = \left(
            \text{prox}_{\sigma P_2} \circ \text{prox}_{\sigma P_1}
            \right)(u)

        where :math:`sigma` is a step size

        Parameters
        ----------
        x : np.ndarray
            Argument for proximal operator.

        step_size : float
            Step size for proximal operator

        Returns
        -------
        np.ndarray
            proximal operator of sparse group lasso penalty evaluated on
            input `x` with step size `step_size`
        """  # noqa: W605
        l1_prox = np.fmax(x - self.alpha * self.lambda_ * step_size, 0) - np.fmax(
            -x - self.alpha * self.lambda_ * step_size, 0
        )
        out = l1_prox.copy()

        if self.bias_index is not None:
            out[self.bias_index] = x[self.bias_index]

        norms = (
            np.array([np.linalg.norm(l1_prox[grp]) for grp in self.groups])
            / self.sqrt_group_lengths
        )

        norm_mask = norms > (1.0 - self.alpha) * self.lambda_ * step_size
        all_norm = all(norm_mask)
        if not all_norm:
            idx_true = np.array([], dtype=int)
        else:
            idx_true = np.concatenate(list(compress(self.groups, norm_mask)))

        if all_norm:
            idx_false = np.array([], dtype=int)
        else:
            idx_false = np.concatenate(
                list(compress(self.groups, np.logical_not(norm_mask)))
            )

        out[idx_true] -= (
            step_size
            * (1.0 - self.alpha)
            * self.lambda_
            * out[idx_true]
            / norms[self.feature2group_map][idx_true]
        )
        out[idx_false] = 0.0

        return out
