from __future__ import absolute_import, division, print_function

import copt as cp
import numpy as np
from collections import namedtuple
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

from .prox import SparseGroupL1

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
def pgd_classifier(x_train, y_train, x_test, y_test, groups,
                   beta0=None, alpha1=0.0, alpha2=0.0, max_iter=5000,
                   tol=1e-6, verbose=0, cb_trace=False, accelerate=False):
    """Find solution to sparse group lasso problem by proximal gradient descent

    Solve sparse group lasso [1]_ problem for feature matrix `x_train` and
    target vector `y_train` with features partitioned into groups. Solve using
    the proximal gradient descent (PGD) algorithm. Compute accuracy and ROC AUC
    using `x_test` and `y_test`.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training feature matrix

    y_train : numpy.ndarray
        Training target array

    x_test : numpy.ndarray
        Testing feature matrix

    y_test : numpy.ndarray
        Testing target array

    groups : numpy.ndarray
        Array of non-overlapping indices for each group. For example, if nine
        features are grouped into equal contiguous groups of three, then groups
        would be an nd.array like [[0, 1, 2], [3, 4, 5], [6, 7, 8]].

    beta0 : numpy.ndarray
        Initial guess for coefficient array

    alpha1 : float, default=0.0
        Group lasso regularization parameter. This encourages groupwise
        sparsity.

    alpha2 : float, default=0.0
        Lasso regularization parameter. This encourages within group sparsity.

    max_iter : int, default=5000
        Maximum number of iterations for PGD algorithm.

    tol : float, default=1e-6
        Convergence tolerance for PGD algorithm.

    verbose : int, default=0
        Verbosity flag for PGD algorithm.

    cb_trace : bool, default=False
        If True, include copt.utils.Trace() object in return

    accelerate : bool, default=False
        If True, use accelerated PGD algorithm, otherwise use standard PGD.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        beta_hat - estimate of the optimal beta,
        accuracy - test set accuracy,
        roc_auc - test set ROC AUC,
        trace - copt.utils.Trace object if cv_trace is True, None otherwise

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250
    """
    n_features = x_train.shape[1]

    if beta0 is None:
        beta0 = np.zeros(n_features)

    sg1 = SparseGroupL1(alpha1, alpha2, groups)

    step_size = 1. / cp.utils.get_lipschitz(x_train, 'logloss')
    f_grad = cp.utils.LogLoss(x_train, y_train).func_grad

    if cb_trace:
        cb_tos = cp.utils.Trace()
        cb_tos(beta0)
    else:
        cb_tos = None

    if accelerate:
        minimizer = cp.minimize_APGD
    else:
        minimizer = cp.minimize_PGD

    pgd = minimizer(
        f_grad, beta0, sg1.prox, step_size=step_size,
        max_iter=max_iter, tol=tol, verbose=verbose,
        callback=cb_tos)

    beta_hat = np.copy(pgd.x)
    y_pred = 1.0 / (1.0 - np.exp(-x_test.dot(beta_hat)))
    y_pred = y_pred > 0.5

    acc = accuracy_score(y_test, y_pred > 0.5)
    auc = roc_auc_score(y_test, y_pred > 0.5)

    Result = namedtuple('Result',
                        'beta_hat accuracy roc_auc trace')
    return Result(beta_hat=beta_hat, accuracy=acc,
                  roc_auc=auc, trace=cb_tos)


@registered
def pgd_classifier_cv(x, y, groups, beta0=None, alpha1=0.0, alpha2=0.0,
                      max_iter=5000, tol=1e-6, verbose=0, cb_trace=False,
                      accelerate=False, n_splits=3, n_repeats=1,
                      random_state=None):
    """Find solution to sparse group lasso problem by proximal gradient descent

    Solve sparse group lasso [1]_ problem for feature matrix `x_train` and
    target vector `y_train` with features partitioned into groups. Solve using
    the proximal gradient descent (PGD) algorithm. Compute accuracy and ROC AUC
    using `x_test` and `y_test`.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix

    y : numpy.ndarray
        Target array

    groups : numpy.ndarray
        Array of non-overlapping indices for each group. For example, if nine
        features are grouped into equal contiguous groups of three, then groups
        would be an nd.array like [[0, 1, 2], [3, 4, 5], [6, 7, 8]].

    beta0 : numpy.ndarray
        Initial guess for coefficient array

    alpha1 : float, default=0.0
        Group lasso regularization parameter. This encourages group-wise
        sparsity.

    alpha2 : float, default=0.0
        Lasso regularization parameter. This encourages within group sparsity.

    max_iter : int, default=5000
        Maximum number of iterations for PGD algorithm.

    tol : float, default=1e-6
        Convergence tolerance for PGD algorithm.

    verbose : int, default=0
        Verbosity flag for PGD algorithm.

    cb_trace : bool, default=False
        If True, include copt.utils.Trace() object in return.

    accelerate : bool, default=False
        If True, use accelerated PGD algorithm, otherwise use standard PGD.

    n_splits : int, default=3
        Number of folds. Must be at least 2.

    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.

    random_state : None, int or RandomState, default=None
        Random state to be used to generate random state for each repetition.


    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        beta_hat - list of estimates of the optimal beta,
        accuracy - list of test set accuracies,
        roc_auc - list of test set ROC AUCs,
        trace - list of copt.utils.Trace objects if cv_trace is True

    See Also
    --------
    pgd_classify
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    clf_results = []

    if verbose:
        verbose -= 1
        splitter = tqdm(rskf.split(x, y), total=rskf.get_n_splits())
    else:
        splitter = rskf.split(x, y)

    for train_idx, test_idx in splitter:
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        res = pgd_classifier(
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            groups=groups,
            beta0=beta0, alpha1=alpha1, alpha2=alpha2,
            max_iter=max_iter, tol=tol,
            verbose=verbose, cb_trace=cb_trace,
            accelerate=accelerate
        )

        clf_results.append(res)
        beta0 = res.beta_hat

    CVResults = namedtuple('CVResults',
                           'beta_hat accuracy roc_auc trace')

    return CVResults(
        beta_hat=[res.beta_hat for res in clf_results],
        accuracy=[res.accuracy for res in clf_results],
        roc_auc=[res.roc_auc for res in clf_results],
        trace=[res.trace for res in clf_results]
    )


@registered
def fit_hyperparams(x, y, groups, max_evals=100,
                    mongo_handle=None, mongo_exp_key=None):
    """Find the best hyperparameters for sparse group lasso using hyperopt.fmin

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix

    y : numpy.ndarray
        Target array

    groups : numpy.ndarray
        Array of non-overlapping indices for each group. For example, if nine
        features are grouped into equal contiguous groups of three, then groups
        would be an nd.array like [[0, 1, 2], [3, 4, 5], [6, 7, 8]].

    max_evals : int, default=100
        Maximum allowed function evaluations for fmin

    mongo_handle : str, hyperopt.mongoexp.MongoJobs, or None, default=None
        If not None, the connection string for the mongodb jobs database or a
        MongoJobs instance. If None, fmin will not parallelize its search
        using mongodb.

    mongo_exp_key : str or None, default=None
        Experiment key for this search if using mongodb to parallelize fmin.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        best_fit - the best fit from hyperopt.fmin
        trials - trials object from hyperopt.fmin
    """
    hp_space = {
        'alpha1': hp.loguniform('alpha1', -4, 2),
        'alpha2': hp.loguniform('alpha2', -4, 2),
    }

    def hp_objective(params):
        hp_cv_partial = partial(pgd_classifier_cv, x=x, y=y, groups=groups)
        cv_results = hp_cv_partial(**params)
        auc = np.array(cv_results.roc_auc)
        return {
            'loss': -np.mean(auc),
            'loss_variance': np.var(auc),
            'status': STATUS_OK
        }

    if mongo_handle is not None:
        trials = MongoTrials(mongo_handle, exp_key=mongo_exp_key)
    else:
        trials = Trials()

    best = fmin(hp_objective, hp_space, algo=tpe.suggest,
                max_evals=max_evals, trials=trials)

    HPResults = namedtuple('HPResults', 'best_fit trials')

    return HPResults(
        best_fit=best,
        trials=trials,
    )
