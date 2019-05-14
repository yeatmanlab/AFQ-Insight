from __future__ import absolute_import, division, print_function

import configparser
import contextlib
import copt as cp
import numpy as np
import os
import os.path as op
import pickle
import warnings
from collections import namedtuple
from functools import partial
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .prox import SparseGroupL1

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@registered
def target_transformation(y, eta=1.0, transform_type="power", direction="forward"):
    r"""Tranform target variables according to
    .. math::

        y_t = \begin{cases}
            y^\eta & y > 0 and transform_type == "power" \\
            \eta^y & y > 0 and transform_type == "exponentiation" \\
            0 & y <= 0
        \end{cases}

    Parameters
    ----------
    y : np.ndarray
        Target variables

    eta : float, default=1.0
        Transformation parameter

    transform_type : ["power", "exponentiation"]
        Type of transformation, see above equation

    direction : ["forward", "inverse"], default="forward"
        If "forward", do the transform listed above. If "inverse", do the
        inverse transform

    Returns
    -------
    y_t : mp.ndarray
        Transformed target variables
    """
    if direction not in ["forward", "inverse"]:
        raise ValueError("'direction' must be either 'forward' or 'inverse'")

    if transform_type not in ["power", "exponentiation"]:
        raise ValueError("'transform_type' must be either 'power' or 'exponentiation'")

    y_t = np.zeros_like(y)
    if direction == "forward":
        if transform_type == "power":
            y_t[y > 0] = np.power(y[y > 0], eta)
        else:
            y_t[y > 0] = np.power(eta, y[y > 0])
    else:
        if transform_type == "power":
            y_t[y > 0] = np.power(y[y > 0], 1.0 / eta)
        else:
            y_t[y > 0] = np.log(y[y > 0]) / np.log(eta)

    return y_t


@registered
def classification_scores(x, y, beta_hat, clf_threshold=0.5):
    """Return classification accuracy and ROC AUC scores

    Parameters
    ----------
    x : np.ndarray
        Feature matrix

    y : np.ndarray
        Target array

    beta_hat : np.ndarray
        Estimate of coefficient array

    clf_threshold : float, default=0.5
        Decision threshold for binary classification

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        accuracy - accuracy score
        auc - ROC AUC
        avg_precision - average precision score
        f1_score - F1 score
    """
    y_pred = _sigmoid(x.dot(beta_hat))

    acc = accuracy_score(y, y_pred > clf_threshold)
    auc = roc_auc_score(y, y_pred)
    aps = average_precision_score(y, y_pred)

    with warnings.catch_warnings():
        # For some metaparameters, we might not predict all of the true labels
        # If that's the case, f1_score will raise a warning to tell us that
        # the F1 score is being set to zero. This is nice to know but it's
        # exactly what we want so we can suppress the warning here.
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        f1 = f1_score(y, y_pred > clf_threshold)

    Scores = namedtuple("Scores", "x y accuracy auc avg_precision f1")
    return Scores(x=x, y=y, accuracy=acc, auc=auc, avg_precision=aps, f1=f1)


@registered
def regression_scores(x, y, beta_hat, eta=1.0, transform_type="power"):
    """Return regression scores

    Parameters
    ----------
    x : np.ndarray
        Feature matrix

    y : np.ndarray
        Target array

    beta_hat : np.ndarray
        Estimate of coefficient array

    eta : float, default=1.0
        Target variable transformation parameter

    transform_type : ["power", "exponentiation"]
        Type of transformation, see above equation

    Returns
    -------
    collections.namedtuple
        namedtuple with field:
        rmse - RMSE score
        r2 - R^2 score, coefficient of determination
        medae - The median absolute error
    """
    y_pred = x.dot(beta_hat)
    y_pred = target_transformation(
        y=y_pred, eta=eta, transform_type=transform_type, direction="forward"
    )

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    medae = median_absolute_error(y, y_pred)

    Scores = namedtuple("Scores", "x y rmse r2 medae")

    return Scores(x=x, y=y, rmse=rmse, r2=r2, medae=medae)


SGLResult = namedtuple(
    "SGLResult",
    ["alpha1", "alpha2", "eta", "transform_type", "beta_hat", "test", "train", "trace"],
)


@registered
def sgl_estimator(
    x_train,
    y_train,
    x_test,
    y_test,
    groups,
    bias_index=None,
    beta0=None,
    alpha1=0.0,
    alpha2=0.0,
    eta=1.0,
    transform_type="power",
    max_iter=5000,
    tol=1e-6,
    verbose=0,
    suppress_warnings=True,
    cb_trace=False,
    accelerate=False,
    loss_type="logloss",
    clf_threshold=0.5,
):
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

    bias_index : int or None, default=None
        the index of the bias feature in x_train and x_test. If None, assume
        no bias feature.

    beta0 : numpy.ndarray
        Initial guess for coefficient array

    alpha1 : float, default=0.0
        Group lasso regularization parameter. This encourages groupwise
        sparsity.

    alpha2 : float, default=0.0
        Lasso regularization parameter. This encourages within group sparsity.

    eta : float, default=1.0
        Target variable transformation parameter.

    transform_type : ["power", "exponentiation"]
        Type of transformation, see above equation

    max_iter : int, default=5000
        Maximum number of iterations for PGD algorithm.

    tol : float, default=1e-6
        Convergence tolerance for PGD algorithm.

    verbose : int, default=0
        Verbosity flag for PGD algorithm.

    suppress_warnings : bool, default=True
        If True, suppress convergence warnings from PGD algorithm.
        This is useful for hyperparameter tuning when some combinations
        of hyperparameters may not converge.

    cb_trace : bool, default=False
        If True, include copt.utils.Trace() object in return

    accelerate : bool, default=False
        If True, use accelerated PGD algorithm, otherwise use standard PGD.

    loss_type : {'logloss', 'square', 'huber'}
        The type of loss function to use. If 'logloss', treat this problem as
        a binary classification problem using logistic regression. Otherwise,
        treat this problem as a regression problem using either the mean
        square error or the Huber loss.

    clf_threshold : float, default=0.5
        Decision threshold for binary classification

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        alpha1 - group lasso regularization parameter,
        alpha2 - lasso regularization parameter,
        eta - target variable transformation parameter,
        beta_hat - estimate of the optimal beta,
        test - scores namedtuple for test set,
        train - scores namedtuple for train set,
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

    sg1 = SparseGroupL1(alpha1, alpha2, groups, bias_index=bias_index)

    if loss_type not in ["logloss", "square", "huber"]:
        raise ValueError("loss_type must be one of " "['logloss', 'square', 'huber'].")

    if loss_type == "logloss" and eta != 1.0:
        raise ValueError("If loss_type is 'logloss', then eta must be one.")

    ind = np.ones(x_train.shape[1], bool)
    if bias_index is not None:
        ind[bias_index] = False

    # Inverse transform target variables
    y_train = target_transformation(
        y=y_train, eta=eta, transform_type=transform_type, direction="inverse"
    )

    if loss_type == "logloss":
        f = cp.utils.LogLoss(x_train, y_train)
    elif loss_type == "huber":
        f = cp.utils.HuberLoss(x_train, y_train)
    else:
        f = cp.utils.SquareLoss(x_train, y_train)

    step_size = 1.0 / f.lipschitz

    if cb_trace:
        cb_tos = cp.utils.Trace()
        cb_tos(beta0)
    else:
        cb_tos = None

    if suppress_warnings:
        ctx_mgr = warnings.catch_warnings()
    else:
        ctx_mgr = contextlib.suppress()

    with ctx_mgr:
        # For some metaparameters, minimize_PGD or minimize_APGD might not
        # reach the desired tolerance level. This might be okay during
        # hyperparameter optimization. So ignore the warning if the user
        # specifies suppress_warnings=True
        if suppress_warnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
        pgd = cp.minimize_proximal_gradient(
            f.f_grad,
            beta0,
            sg1.prox,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            callback=cb_tos,
            accelerated=accelerate,
        )

    beta_hat = np.copy(pgd.x)

    # Transform the target variables back to original
    y_train = target_transformation(
        y=y_train, eta=eta, transform_type=transform_type, direction="forward"
    )

    if loss_type == "logloss":
        train = classification_scores(
            x=x_train, y=y_train, beta_hat=beta_hat, clf_threshold=clf_threshold
        )
        test = classification_scores(
            x=x_test, y=y_test, beta_hat=beta_hat, clf_threshold=clf_threshold
        )
    else:
        train = regression_scores(
            x=x_train,
            y=y_train,
            beta_hat=beta_hat,
            eta=eta,
            transform_type=transform_type,
        )
        test = regression_scores(
            x=x_test,
            y=y_test,
            beta_hat=beta_hat,
            eta=eta,
            transform_type=transform_type,
        )

    return SGLResult(
        alpha1=alpha1,
        alpha2=alpha2,
        eta=eta,
        transform_type=transform_type,
        beta_hat=beta_hat,
        test=test,
        train=train,
        trace=cb_tos,
    )


@registered
def sgl_estimator_cv(
    x,
    y,
    groups,
    bias_index=None,
    beta0=None,
    alpha1=0.0,
    alpha2=0.0,
    eta=1.0,
    transform_type="power",
    max_iter=5000,
    tol=1e-6,
    verbose=0,
    cb_trace=False,
    accelerate=False,
    loss_type="logloss",
    n_splits=3,
    n_repeats=1,
    random_state=None,
):
    """Find solution to sparse group lasso problem by proximal gradient descent

    Solve sparse group lasso [1]_ problem for feature matrix `x_train` and
    target vector `y_train` with features partitioned into groups. Solve using
    the proximal gradient descent (PGD) algorithm. Compute accuracy, ROC AUC,
    and average precision using `x_test` and `y_test`.

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

    bias_index : int or None, default=None
        the index of the bias feature in x_train and x_test. If None, assume
        no bias feature.

    beta0 : numpy.ndarray
        Initial guess for coefficient array

    alpha1 : float, default=0.0
        Group lasso regularization parameter. This encourages group-wise
        sparsity.

    alpha2 : float, default=0.0
        Lasso regularization parameter. This encourages within group sparsity.

    eta : float, default=1.0
        Target variable transformation parameter.

    transform_type : ["power", "exponentiation"]
        Type of transformation, see above equation

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

    loss_type : {'logloss', 'square', 'huber'}
        The type of loss function to use. If 'logloss', treat this problem as
        a binary classification problem using logistic regression. Otherwise,
        treat this problem as a regression problem using either the mean
        square error or the Huber loss.

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
        alpha1 - group lasso regularization parameter,
        alpha2 - lasso regularization parameter,
        eta - target variable transformation parameter,
        beta_hat - list of estimates of the optimal beta,
        test - list of scores namedtuples for test set,
        train - list of scores namedtuples for train set,
        trace - list of copt.utils.Trace objects if cv_trace is True

    See Also
    --------
    pgd_classify
    """
    if loss_type == "logloss":
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        cv = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    clf_results = []

    if verbose:
        verbose -= 1
        splitter = tqdm(cv.split(x, y), total=cv.get_n_splits())
    else:
        splitter = cv.split(x, y)

    scaler = StandardScaler()

    if 0.0 <= eta < 1.0:
        raise ValueError("eta must satisfy 0.0 <= eta < 1.0")

    for train_idx, test_idx in splitter:
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train[:, bias_index] = 1.0
        x_test[:, bias_index] = 1.0

        res = sgl_estimator(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            groups=groups,
            bias_index=bias_index,
            beta0=beta0,
            alpha1=alpha1,
            alpha2=alpha2,
            eta=eta,
            transform_type=transform_type,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            cb_trace=cb_trace,
            accelerate=accelerate,
            loss_type=loss_type,
        )

        clf_results.append(res)
        beta0 = res.beta_hat

    return SGLResult(
        alpha1=alpha1,
        alpha2=alpha2,
        eta=eta,
        transform_type=transform_type,
        beta_hat=[res.beta_hat for res in clf_results],
        test=[res.test for res in clf_results],
        train=[res.train for res in clf_results],
        trace=[res.trace for res in clf_results],
    )


@registered
def fit_hyperparams(
    x,
    y,
    groups,
    bias_index=None,
    max_evals=100,
    beta0=None,
    loss_type="logloss",
    score="roc_auc",
    trials=None,
    mongo_handle=None,
    mongo_exp_key=None,
    save_trials_pickle=None,
):
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

    bias_index : int or None, default=None
        the index of the bias feature in x_train and x_test. If None, assume
        no bias feature.

    max_evals : int, default=100
        Maximum allowed function evaluations for fmin

    beta0 : numpy.ndarray
        Initial guess for coefficient array

    loss_type : {'logloss', 'square', 'huber'}
        The type of loss function to use. If 'logloss', treat this problem as
        a binary classification problem using logistic regression. Otherwise,
        treat this problem as a regression problem using either the mean
        square error or the Huber loss.

    score : {'roc_auc', 'accuracy', 'avg_precision', 'r2', 'rmse', 'medae'},
        default='roc_auc'. scoring metric to be optimized

    trials : hyperopt Trials or MongoTrials object or None, default=None
        Pre-existing Trials or MongoTrials object to continue a previous
        hyperopt search.

    mongo_handle : str, hyperopt.mongoexp.MongoJobs, or None, default=None
        If not None, the connection string for the mongodb jobs database or a
        MongoJobs instance. If None, fmin will not parallelize its search
        using mongodb.

    mongo_exp_key : str or None, default=None
        Experiment key for this search if using mongodb to parallelize fmin.

    save_trials_pickle : str or None, default=None
        If not None, save trials dictionary to pickle file using this filename.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        best_fit - the best fit from hyperopt.fmin
        trials - trials object from hyperopt.fmin
    """
    allowed_clf_scores = ["roc_auc", "accuracy", "avg_precision"]
    allowed_rgs_scores = ["r2", "rmse", "medae"]
    if loss_type == "logloss" and score not in allowed_clf_scores:
        raise ValueError(
            "For classification problems `score` must be one of "
            "{scores!s}.".format(scores=allowed_clf_scores)
        )
    elif loss_type in ["square", "huber"] and score not in allowed_rgs_scores:
        raise ValueError(
            "For regression problems `score` must be one of "
            "{scores!s}.".format(scores=allowed_rgs_scores)
        )

    # Define the search space
    hp_space = {
        "alpha1": hp.uniform("alpha1", 0.0, 1.0),
        "alpha2": hp.loguniform("alpha2", -7, 3),
        "eta": hp.uniform("eta", 1.0, 3.0),
        "transform_type": hp.choice("transform_type", ["power", "exponentiation"]),
    }

    if trials is None:
        if mongo_handle is not None:
            trials = MongoTrials(mongo_handle, exp_key=mongo_exp_key)
        else:
            trials = Trials()

    # Define the objective function for hyperopt to minimize
    def hp_objective(params):
        hp_cv_partial = partial(
            sgl_estimator_cv,
            x=x,
            y=y,
            groups=groups,
            loss_type=loss_type,
            bias_index=bias_index,
            beta0=beta0,
        )
        cv_results = hp_cv_partial(**params)
        if loss_type == "logloss":
            auc = np.array([test.auc for test in cv_results.test])
            acc = np.array([test.accuracy for test in cv_results.test])
            aps = np.array([test.avg_precision for test in cv_results.test])
            if score == "roc_auc":
                loss = -auc
            elif score == "accuracy":
                loss = -acc
            else:
                loss = -aps

            result_dict = {
                "loss": np.mean(loss),
                "loss_variance": np.var(loss),
                "status": STATUS_OK,
                "accuracy_mean": np.mean(acc),
                "accuracy_variance": np.var(acc),
                "roc_auc_mean": np.mean(auc),
                "roc_auc_variance": np.var(auc),
                "avg_precision_mean": np.mean(aps),
                "avg_precision_variance": np.var(aps),
            }
        else:
            r2 = np.array([test.r2 for test in cv_results.test])
            rmse = np.array([test.rmse for test in cv_results.test])
            medae = np.array([test.medae for test in cv_results.test])
            if score == "rmse":
                loss = rmse
            elif score == "medae":
                loss = medae
            else:
                loss = -r2
            result_dict = {
                "loss": np.mean(loss),
                "loss_variance": np.var(loss),
                "status": STATUS_OK,
                "r2_mean": np.mean(r2),
                "r2_variance": np.var(r2),
                "rmse_mean": np.mean(rmse),
                "rmse_variance": np.var(rmse),
                "medae_mean": np.mean(medae),
                "medae_variance": np.var(medae),
            }

        return result_dict

    best = fmin(
        hp_objective, hp_space, algo=tpe.suggest, max_evals=max_evals, trials=trials
    )

    if mongo_handle is None and save_trials_pickle is not None:
        with open(save_trials_pickle, "wb") as fp:
            pickle.dump(trials, fp)

    HPResults = namedtuple("HPResults", "best_fit trials space")

    return HPResults(best_fit=best, trials=trials, space=hp_space)


@registered
def fit_hyperparams_cv(
    x,
    y,
    groups,
    bias_index=None,
    beta0=None,
    n_splits=10,
    n_repeats=1,
    max_evals_per_cv=100,
    loss_type="logloss",
    score="roc_auc",
    trials_pickle_dir=None,
    mongo_handle=None,
    random_state=None,
    verbose=0,
    clf_threshold=0.5,
):
    """Run fit_hyperparams over K-fold cross validation

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

    bias_index : int or None, default=None
        the index of the bias feature in x_train and x_test. If None, assume
        no bias feature.

    beta0 : numpy.ndarray
        Initial guess for coefficient array

    n_splits : int, default=10
        Number of folds. Must be at least 2.

    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.

    max_evals_per_cv : int, default=100
        Maximum allowed function evaluations for fmin

    loss_type : {'logloss', 'square', 'huber'}
        The type of loss function to use. If 'logloss', treat this problem as
        a binary classification problem using logistic regression. Otherwise,
        treat this problem as a regression problem using either the mean
        square error or the Huber loss.

    score : {'roc_auc', 'accuracy', 'avg_precision', 'r2', 'rmse', 'medae'},
        default='roc_auc'. scoring metric to be optimized

    trials_pickle_dir : str or None, default=None
        Directory to store/retrieve pickled trials

    mongo_handle : str, hyperopt.mongoexp.MongoJobs, or None, default=None
        If not None, the connection string for the mongodb jobs database or a
        MongoJobs instance. If None, fmin will not parallelize its search
        using mongodb.

    random_state : None, int or RandomState, default=None
        Random state to be used to generate random state for each repetition.

    verbose : int, default=0
        Verbosity flag for CV loop

    clf_threshold : float, default=0.5
        Decision threshold for binary classification

    Returns
    -------
    """
    allowed_clf_scores = ["roc_auc", "accuracy", "avg_precision"]
    allowed_rgs_scores = ["r2", "rmse", "medae"]
    if loss_type == "logloss" and score not in allowed_clf_scores:
        raise ValueError(
            "For classification problems `score` must be one of "
            "{scores!s}.".format(scores=allowed_clf_scores)
        )
    elif loss_type in ["square", "huber"] and score not in allowed_rgs_scores:
        raise ValueError(
            "For regression problems `score` must be one of "
            "{scores!s}.".format(scores=allowed_rgs_scores)
        )

    if trials_pickle_dir is not None:
        os.makedirs(op.abspath(trials_pickle_dir), exist_ok=True)
        configfile = op.join(op.abspath(trials_pickle_dir), "params.ini")
        config = configparser.ConfigParser()
        if op.isfile(configfile):
            # Check that existing params equal input params
            config.read(configfile)
            params = config["params"]
            if not all(
                [
                    params.getint("n_splits") == n_splits,
                    params.getint("n_repeats") == n_repeats,
                    params.getint("random_state") == random_state,
                    params["loss_type"] == loss_type,
                    params["score"] == score,
                ]
            ):
                raise ValueError(
                    "Stored trial parameters do not match input parameters. "
                    "This could contaminate the train/test split for previous "
                    "trials. Either set n_splits={ns:s}, random_state={rs:s}, "
                    "score={score:s}, loss_type={loss_type:s} or specify a "
                    "new trials directory".format(
                        ns=params["n_splits"],
                        nr=params["n_repeats"],
                        rs=params["random_state"],
                        score=params["score"],
                        loss_type=params["loss_type"],
                    )
                )
        else:
            # Write input params to file
            config["params"] = {
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "random_state": random_state,
                "loss_type": loss_type,
                "score": score,
            }
            with open(configfile, "w") as cfile:
                config.write(cfile)

    if loss_type == "logloss":
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        cv = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    if verbose:
        splitter = tqdm(enumerate(cv.split(x, y)), total=cv.get_n_splits())
    else:
        splitter = enumerate(cv.split(x, y))

    cv_results = []

    scaler = StandardScaler()

    for cv_idx, (train_idx, test_idx) in splitter:
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if mongo_handle is None:
            mongo_exp_key = None
            if trials_pickle_dir is not None:
                pickle_name = op.join(
                    op.abspath(trials_pickle_dir),
                    "cv{i:03d}_trials.pkl".format(i=cv_idx),
                )

                try:
                    with open(pickle_name, "rb") as fp:
                        trials = pickle.load(fp)
                except IOError:
                    trials = None
            else:
                pickle_name = None
                trials = None
        else:
            mongo_exp_key = "cv_{i:03d}".format(i=cv_idx)
            pickle_name = None
            trials = None

        hp_res = fit_hyperparams(
            x_train,
            y_train,
            groups,
            bias_index=bias_index,
            beta0=beta0,
            max_evals=max_evals_per_cv,
            loss_type=loss_type,
            score=score,
            trials=trials,
            mongo_handle=mongo_handle,
            mongo_exp_key=mongo_exp_key,
            save_trials_pickle=pickle_name,
        )

        alpha1 = hp_res.best_fit["alpha1"]
        alpha2 = hp_res.best_fit["alpha2"]
        eta = hp_res.best_fit.get("eta", 1.0)
        if loss_type != "logloss":
            transform_type = space_eval(hp_res.space, hp_res.best_fit)["transform_type"]
        else:
            transform_type = "power"

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train[:, bias_index] = 1.0
        x_test[:, bias_index] = 1.0

        sgl = sgl_estimator(
            x_train,
            y_train,
            x_test,
            y_test,
            groups,
            alpha1=alpha1,
            alpha2=alpha2,
            eta=eta,
            transform_type=transform_type,
            loss_type=loss_type,
            clf_threshold=clf_threshold,
        )

        beta0 = sgl.beta_hat

        cv_results.append(sgl)

    return cv_results
