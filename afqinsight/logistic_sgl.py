import numpy as np

# from joblib import Parallel, delayed, effective_n_jobs
from scipy import sparse
from tqdm.auto import tqdm

from sklearn.base import TransformerMixin
from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
from sklearn.linear_model._coordinate_descent import _path_residuals
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from .sgl import SGLBaseEstimator
from .utils import check_groups

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
class LogisticSGL(SGLBaseEstimator, LinearClassifierMixin):
    """
    An sklearn compatible sparse group lasso classifier.

    This solves the sparse group lasso [1]_ problem for a feature matrix
    partitioned into groups using the proximal gradient descent (PGD)
    algorithm.

    Parameters
    ----------
    l1_ratio : float, default=1.0
        Hyper-parameter : Combination between group lasso and lasso. l1_ratio=0
        gives the group lasso and l1_ratio=1 gives the lasso.

    alpha : float, default=0.0
        Hyper-parameter : overall regularization strength.

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group. We set groups in ``__init__`` so
        that it can be reused in model selection and CV routines.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=1000
        Maximum number of iterations for PGD solver.

    tol : float, default=1e-7
        Stopping criterion. Convergence tolerance for PGD algorithm.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_``.

    verbose : int, default=0
        Verbosity flag for PGD solver. Any positive integer will produce
        verbose output

    suppress_solver_warnings : bool, default=True
        If True, suppress convergence warnings from PGD solver.
        This is useful for hyperparameter tuning when some combinations
        of hyperparameters may not converge.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`).

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.

    Examples
    --------

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250

    """

    def fit(self, X, y):  # pylint: disable=arguments-differ
        """Fit a linear model using the sparse group lasso

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        return super().fit(X=X, y=y, loss="log")

    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse="csr")

        n_features = self.coef_.size
        if X.shape[1] != n_features:
            raise ValueError(
                "X has %d features per sample; expecting %d" % (X.shape[1], n_features)
            )

        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return scores

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)

        return self.classes_[indices]

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the label of classes.

        Else use a one-vs-rest approach, i.e calculate the probability of
        each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        return super()._predict_proba_lr(X)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"binary_only": True, "requires_y": True}


# def _logistic_regression_path(
#     X,
#     y,
#     pos_class=None,
#     alphas=10,
#     fit_intercept=True,
#     max_iter=1000,
#     tol=1e-4,
#     verbose=0,
#     coef=None,
#     class_weight=None,
#     random_state=None,
#     check_input=True,
#     l1_ratio=None,
# ):
#     """Compute a Logistic SGL model for a list of regularization parameters.

#     This is an implementation that uses the result of the previous model
#     to speed up computations along the set of solutions, making it faster
#     than sequentially calling LogisticSGL for the different parameters.

#     Parameters
#     ----------
#     X : {array-like, sparse matrix} of shape (n_samples, n_features)
#         Input data.

#     y : array-like of shape (n_samples,) or (n_samples, n_targets)
#         Input data, target values.

#     pos_class : int, default=None
#         The class with respect to which we perform a one-vs-all fit.
#         If None, then it is assumed that the given problem is binary.

#     alphas : int or array-like of shape (n_cs,), default=10
#         List of values for the regularization parameter or integer specifying
#         the number of regularization parameters that should be used. In this
#         case, the parameters will be chosen in a logarithmic scale between
#         1e-4 and 1e4.

#     fit_intercept : bool, default=True
#         Whether to fit an intercept for the model. In this case the shape of
#         the returned array is (n_cs, n_features + 1).

#     max_iter : int, default=1000
#         Maximum number of iterations for the solver.

#     tol : float, default=1e-4
#         Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
#         will stop when ``max{|g_i | i = 1, ..., n} <= tol``
#         where ``g_i`` is the i-th component of the gradient.

#     verbose : int, default=0
#         For the liblinear and lbfgs solvers set verbose to any positive
#         number for verbosity.

#     coef : array-like of shape (n_features,), default=None
#         Initialization value for coefficients of logistic regression.

#     random_state : int, RandomState instance, default=None
#         Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
#         data. See :term:`Glossary <random_state>` for details.

#     check_input : bool, default=True
#         If False, the input arrays X and y will not be checked.

#     l1_ratio : float, default=None
#         The group lasso / lasso mixing parameter, with ``0 <= l1_ratio <= 1``.
#         Setting ``l1_ratio=0`` is equivalent to pure group lasso, while setting
#         ``l1_ratio=1`` is equivalent to pure lasso.

#     Returns
#     -------
#     coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
#         List of coefficients for the Logistic Regression model. If
#         fit_intercept is set to True then the second dimension will be
#         n_features + 1, where the last item represents the intercept. For
#         ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
#         n_features) or (n_classes, n_cs, n_features + 1).

#     alphas : ndarray
#         Grid of alphas used for cross-validation.

#     n_iter : array of shape (n_cs,)
#         Actual number of iteration for each Cs.
#     """
#     if isinstance(alphas, numbers.Integral):
#         alphas = np.logspace(-4, 4, alphas)

#     # Preprocessing.
#     if check_input:
#         X = check_array(X, accept_sparse="csr", dtype=np.float64)
#         y = check_array(y, ensure_2d=False, dtype=None)
#         check_consistent_length(X, y)

#     _, n_features = X.shape

#     classes = np.unique(y)
#     random_state = check_random_state(random_state)

#     if pos_class is None and multi_class != "multinomial":
#         if classes.size > 2:
#             raise ValueError("To fit OvR, use the pos_class argument")
#         # np.unique(y) gives labels in sorted order.
#         pos_class = classes[1]

#     # If class_weights is a dict (provided by the user), the weights
#     # are assigned to the original labels. If it is "balanced", then
#     # the class_weights are assigned after masking the labels with a OvR.
#     le = LabelEncoder()
#     if isinstance(class_weight, dict) or multi_class == "multinomial":
#         class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
#         sample_weight *= class_weight_[le.fit_transform(y)]

#     # For doing a ovr, we need to mask the labels first. for the
#     # multinomial case this is not necessary.
#     if multi_class == "ovr":
#         w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
#         mask_classes = np.array([-1, 1])
#         mask = y == pos_class
#         y_bin = np.ones(y.shape, dtype=X.dtype)
#         y_bin[~mask] = -1.0
#         # for compute_class_weight

#         if class_weight == "balanced":
#             class_weight_ = compute_class_weight(
#                 class_weight, classes=mask_classes, y=y_bin
#             )
#             sample_weight *= class_weight_[le.fit_transform(y_bin)]

#     else:
#         if solver not in ["sag", "saga"]:
#             lbin = LabelBinarizer()
#             Y_multi = lbin.fit_transform(y)
#             if Y_multi.shape[1] == 1:
#                 Y_multi = np.hstack([1 - Y_multi, Y_multi])
#         else:
#             # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
#             le = LabelEncoder()
#             Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

#         w0 = np.zeros(
#             (classes.size, n_features + int(fit_intercept)), order="F", dtype=X.dtype
#         )

#     if coef is not None:
#         # it must work both giving the bias term and not
#         if multi_class == "ovr":
#             if coef.size not in (n_features, w0.size):
#                 raise ValueError(
#                     "Initialization coef is of shape %d, expected shape "
#                     "%d or %d" % (coef.size, n_features, w0.size)
#                 )
#             w0[: coef.size] = coef
#         else:
#             # For binary problems coef.shape[0] should be 1, otherwise it
#             # should be classes.size.
#             n_classes = classes.size
#             if n_classes == 2:
#                 n_classes = 1

#             if coef.shape[0] != n_classes or coef.shape[1] not in (
#                 n_features,
#                 n_features + 1,
#             ):
#                 raise ValueError(
#                     "Initialization coef is of shape (%d, %d), expected "
#                     "shape (%d, %d) or (%d, %d)"
#                     % (
#                         coef.shape[0],
#                         coef.shape[1],
#                         classes.size,
#                         n_features,
#                         classes.size,
#                         n_features + 1,
#                     )
#                 )

#             if n_classes == 1:
#                 w0[0, : coef.shape[1]] = -coef
#                 w0[1, : coef.shape[1]] = coef
#             else:
#                 w0[:, : coef.shape[1]] = coef

#     if multi_class == "multinomial":
#         # scipy.optimize.minimize and newton-cg accepts only
#         # ravelled parameters.
#         if solver in ["lbfgs", "newton-cg"]:
#             w0 = w0.ravel()
#         target = Y_multi
#         if solver == "lbfgs":

#             def func(x, *args):
#                 return _multinomial_loss_grad(x, *args)[0:2]

#         elif solver == "newton-cg":

#             def func(x, *args):
#                 return _multinomial_loss(x, *args)[0]

#             def grad(x, *args):
#                 return _multinomial_loss_grad(x, *args)[1]

#             hess = _multinomial_grad_hess
#         warm_start_sag = {"coef": w0.T}
#     else:
#         target = y_bin
#         if solver == "lbfgs":
#             func = _logistic_loss_and_grad
#         elif solver == "newton-cg":
#             func = _logistic_loss

#             def grad(x, *args):
#                 return _logistic_loss_and_grad(x, *args)[1]

#             hess = _logistic_grad_hess
#         warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

#     coefs = list()
#     n_iter = np.zeros(len(Cs), dtype=np.int32)
#     for i, C in enumerate(Cs):
#         if solver == "lbfgs":
#             iprint = [-1, 50, 1, 100, 101][
#                 np.searchsorted(np.array([0, 1, 2, 3]), verbose)
#             ]
#             opt_res = optimize.minimize(
#                 func,
#                 w0,
#                 method="L-BFGS-B",
#                 jac=True,
#                 args=(X, target, 1.0 / C, sample_weight),
#                 options={"iprint": iprint, "gtol": tol, "maxiter": max_iter},
#             )
#             n_iter_i = _check_optimize_result(
#                 solver,
#                 opt_res,
#                 max_iter,
#                 extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
#             )
#             w0, loss = opt_res.x, opt_res.fun
#         elif solver == "newton-cg":
#             args = (X, target, 1.0 / C, sample_weight)
#             w0, n_iter_i = _newton_cg(
#                 hess, func, grad, w0, args=args, maxiter=max_iter, tol=tol
#             )
#         elif solver == "liblinear":
#             coef_, intercept_, n_iter_i, = _fit_liblinear(
#                 X,
#                 target,
#                 C,
#                 fit_intercept,
#                 intercept_scaling,
#                 None,
#                 penalty,
#                 dual,
#                 verbose,
#                 max_iter,
#                 tol,
#                 random_state,
#                 sample_weight=sample_weight,
#             )
#             if fit_intercept:
#                 w0 = np.concatenate([coef_.ravel(), intercept_])
#             else:
#                 w0 = coef_.ravel()

#         elif solver in ["sag", "saga"]:
#             if multi_class == "multinomial":
#                 target = target.astype(X.dtype, copy=False)
#                 loss = "multinomial"
#             else:
#                 loss = "log"
#             # alpha is for L2-norm, beta is for L1-norm
#             if penalty == "l1":
#                 alpha = 0.0
#                 beta = 1.0 / C
#             elif penalty == "l2":
#                 alpha = 1.0 / C
#                 beta = 0.0
#             else:  # Elastic-Net penalty
#                 alpha = (1.0 / C) * (1 - l1_ratio)
#                 beta = (1.0 / C) * l1_ratio

#             w0, n_iter_i, warm_start_sag = sag_solver(
#                 X,
#                 target,
#                 sample_weight,
#                 loss,
#                 alpha,
#                 beta,
#                 max_iter,
#                 tol,
#                 verbose,
#                 random_state,
#                 False,
#                 max_squared_sum,
#                 warm_start_sag,
#                 is_saga=(solver == "saga"),
#             )

#         else:
#             raise ValueError(
#                 "solver must be one of {'liblinear', 'lbfgs', "
#                 "'newton-cg', 'sag'}, got '%s' instead" % solver
#             )

#         if multi_class == "multinomial":
#             n_classes = max(2, classes.size)
#             multi_w0 = np.reshape(w0, (n_classes, -1))
#             if n_classes == 2:
#                 multi_w0 = multi_w0[1][np.newaxis, :]
#             coefs.append(multi_w0.copy())
#         else:
#             coefs.append(w0.copy())

#         n_iter[i] = n_iter_i

#     return np.array(coefs), np.array(Cs), n_iter


# # helper function for LogisticSGLCV
# def _log_reg_scoring_path(
#     X,
#     y,
#     train,
#     test,
#     pos_class=None,
#     Cs=10,
#     scoring=None,
#     fit_intercept=False,
#     max_iter=100,
#     tol=1e-4,
#     class_weight=None,
#     verbose=0,
#     solver="lbfgs",
#     penalty="l2",
#     dual=False,
#     intercept_scaling=1.0,
#     multi_class="auto",
#     random_state=None,
#     l1_ratio=None,
# ):
#     """Computes scores across logistic SGL path

#     Parameters
#     ----------
#     X : {array-like, sparse matrix} of shape (n_samples, n_features)
#         Training data.

#     y : array-like of shape (n_samples,) or (n_samples, n_targets)
#         Target labels.

#     train : list of indices
#         The indices of the train set.

#     test : list of indices
#         The indices of the test set.

#     pos_class : int, default=None
#         The class with respect to which we perform a one-vs-all fit.
#         If None, then it is assumed that the given problem is binary.

#     alphas : int or list of floats, default=10
#         Each of the values in Cs describes the inverse of
#         regularization strength. If Cs is as an int, then a grid of Cs
#         values are chosen in a logarithmic scale between 1e-4 and 1e4.
#         If not provided, then a fixed set of values for Cs are used.

#     scoring : callable, default=None
#         A string (see model evaluation documentation) or
#         a scorer callable object / function with signature
#         ``scorer(estimator, X, y)``. For a list of scoring functions
#         that can be used, look at :mod:`sklearn.metrics`. The
#         default scoring option used is accuracy_score.

#     fit_intercept : bool, default=False
#         If False, then the bias term is set to zero. Else the last
#         term of each coef_ gives us the intercept.

#     max_iter : int, default=1000
#         Maximum number of iterations for the solver.

#     tol : float, default=1e-4
#         Tolerance for stopping criteria.

#     verbose : int, default=0
#         For the liblinear and lbfgs solvers set verbose to any positive
#         number for verbosity.

#     random_state : int, RandomState instance, default=None
#         Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
#         data. See :term:`Glossary <random_state>` for details.

#     l1_ratio : float, default=None
#         The group lasso / lasso mixing parameter, with ``0 <= l1_ratio <= 1``.
#         Setting ``l1_ratio=0`` is equivalent to pure group lasso, while setting
#         ``l1_ratio=1`` is equivalent to pure lasso.

#     Returns
#     -------
#     coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
#         List of coefficients for the Logistic Regression model. If
#         fit_intercept is set to True then the second dimension will be
#         n_features + 1, where the last item represents the intercept.

#     alphas : ndarray
#         Grid of alphas used for cross-validation.

#     scores : ndarray of shape (n_cs,)
#         Scores obtained for each Cs.

#     n_iter : ndarray of shape(n_cs,)
#         Actual number of iteration for each Cs.
#     """
#     X_train = X[train]
#     X_test = X[test]
#     y_train = y[train]
#     y_test = y[test]

#     if sample_weight is not None:
#         sample_weight = _check_sample_weight(sample_weight, X)
#         sample_weight = sample_weight[train]

#     coefs, Cs, n_iter = _logistic_regression_path(
#         X_train,
#         y_train,
#         Cs=Cs,
#         l1_ratio=l1_ratio,
#         fit_intercept=fit_intercept,
#         solver=solver,
#         max_iter=max_iter,
#         class_weight=class_weight,
#         pos_class=pos_class,
#         multi_class=multi_class,
#         tol=tol,
#         verbose=verbose,
#         dual=dual,
#         penalty=penalty,
#         intercept_scaling=intercept_scaling,
#         random_state=random_state,
#         check_input=False,
#         max_squared_sum=max_squared_sum,
#         sample_weight=sample_weight,
#     )

#     log_reg = LogisticRegression(solver=solver, multi_class=multi_class)

#     # The score method of Logistic Regression has a classes_ attribute.
#     if multi_class == "ovr":
#         log_reg.classes_ = np.array([-1, 1])
#     elif multi_class == "multinomial":
#         log_reg.classes_ = np.unique(y_train)
#     else:
#         raise ValueError(
#             "multi_class should be either multinomial or ovr, " "got %d" % multi_class
#         )

#     if pos_class is not None:
#         mask = y_test == pos_class
#         y_test = np.ones(y_test.shape, dtype=np.float64)
#         y_test[~mask] = -1.0

#     scores = list()

#     scoring = get_scorer(scoring)
#     for w in coefs:
#         if multi_class == "ovr":
#             w = w[np.newaxis, :]
#         if fit_intercept:
#             log_reg.coef_ = w[:, :-1]
#             log_reg.intercept_ = w[:, -1]
#         else:
#             log_reg.coef_ = w
#             log_reg.intercept_ = 0.0

#         if scoring is None:
#             scores.append(log_reg.score(X_test, y_test))
#         else:
#             scores.append(scoring(log_reg, X_test, y_test))

#     return coefs, Cs, np.array(scores), n_iter


# @registered
# class LogisticSGLCV(LinearModel, LinearClassifierMixin, TransformerMixin):
#     """Class for iterative Logistic SGL model fitting along a regularization path

#     Parameters
#     ----------
#     l1_ratio : float or list of float, default=1.0
#         float between 0 and 1 passed to SGL (scaling between group lasso and
#         lasso penalties). For ``l1_ratio = 0`` the penalty is the group lasso
#         penalty. For ``l1_ratio = 1`` it is the lasso penalty. For ``0 <
#         l1_ratio < 1``, the penalty is a combination of group lasso and
#         lasso. This parameter can be a list, in which case the different
#         values are tested by cross-validation and the one giving the best
#         prediction score is used. Note that a good choice of list of values
#         will depend on the problem. For problems where we expect strong
#         overall sparsity and would like to encourage grouping, put more
#         values close to 1 (i.e. Lasso). In contrast, if we expect strong
#         group-wise sparsity, but only mild sparsity within groups, put more
#         values close to 0 (i.e. group lasso).

#     groups : list of numpy.ndarray
#         list of arrays of non-overlapping indices for each group. For
#         example, if nine features are grouped into equal contiguous groups of
#         three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
#         array([6, 7, 8])]``. If the feature matrix contains a bias or
#         intercept feature, do not include it as a group. If None, all
#         features will belong to one group. We set groups in ``__init__`` so
#         that it can be reused in model selection and CV routines.

#     scale_l2_by : ["group_length", None], default="group_length"
#         Scaling technique for the group-wise L2 penalty.
#         By default, ``scale_l2_by="group_length`` and the L2 penalty is
#         scaled by the square root of the group length so that each variable
#         has the same effect on the penalty. This may not be appropriate for
#         one-hot encoded features and ``scale_l2_by=None`` would be more
#         appropriate for that case. ``scale_l2_by=None`` will also reproduce
#         ElasticNet results when all features belong to one group.

#     eps : float, default=1e-3
#         Length of the path. ``eps=1e-3`` means that
#         ``alpha_min / alpha_max = 1e-3``.

#     n_alphas : int, default=100
#         Number of alphas along the regularization path, used for each l1_ratio.

#     alphas : ndarray, default=None
#         List of alphas where to compute the models.
#         If None alphas are set automatically

#     fit_intercept : bool, default=True
#         whether to calculate the intercept for this model. If set
#         to false, no intercept will be used in calculations
#         (i.e. data is expected to be centered).

#     normalize : bool, default=False
#         This parameter is ignored when ``fit_intercept`` is set to False.
#         If True, the regressors X will be normalized before regression by
#         subtracting the mean and dividing by the l2-norm.
#         If you wish to standardize, please use
#         :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
#         on an estimator with ``normalize=False``.

#     max_iter : int, default=1000
#         The maximum number of iterations

#     tol : float, default=1e-7
#         The tolerance for the SGL solver

#     cv : int, cross-validation generator or iterable, default=None
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:

#         - None, to use the default 5-fold cross-validation,
#         - int, to specify the number of folds.
#         - :term:`CV splitter`,
#         - An iterable yielding (train, test) splits as arrays of indices.

#         For int/None inputs, :class:`KFold` is used.

#         Refer to the scikit-learn User Guide for the various
#         cross-validation strategies that can be used here.

#     copy_X : bool, default=True
#         If ``True``, X will be copied; else, it may be overwritten.

#     verbose : bool or int, default=0
#         Amount of verbosity.

#     n_jobs : int, default=None
#         Number of CPUs to use during the cross validation.
#         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#         ``-1`` means using all processors.

#     random_state : int, RandomState instance, default=None
#         The seed of the pseudo random number generator that selects a random
#         feature to update. Used when ``selection`` == 'random'.
#         Pass an int for reproducible output across multiple function calls.

#     Attributes
#     ----------
#     alpha_ : float
#         The amount of penalization chosen by cross validation

#     l1_ratio_ : float
#         The compromise between l1 and l2 penalization chosen by
#         cross validation

#     classes_ : ndarray of shape (n_classes, )
#         A list of class labels known to the classifier.

#     coef_ : array of shape (n_features,)
#         Estimated coefficients for the linear predictor (`X @ coef_ +
#         intercept_`).

#     intercept_ : float
#         Intercept (a.k.a. bias) added to linear predictor.

#     n_iter_ : int
#         Actual number of iterations used in the solver.

#     mse_path_ : ndarray of shape (n_l1_ratio, n_alpha, n_folds)
#         Mean square error for the test set on each fold, varying l1_ratio and
#         alpha.

#     alphas_ : ndarray of shape (n_alphas,) or (n_l1_ratio, n_alphas)
#         The grid of alphas used for fitting, for each l1_ratio.

#     n_iter_ : int
#         number of iterations run by the proximal gradient descent solver to
#         reach the specified tolerance for the optimal alpha.

#     See also
#     --------
#     sgl_path
#     SGL
#     """

#     def __init__(
#         self,
#         l1_ratio=1.0,
#         groups=None,
#         scale_l2_by="group_length",
#         eps=1e-3,
#         n_alphas=100,
#         alphas=None,
#         fit_intercept=True,
#         normalize=False,
#         max_iter=1000,
#         tol=1e-7,
#         copy_X=True,
#         cv=None,
#         verbose=False,
#         n_jobs=None,
#         random_state=None,
#     ):
#         self.l1_ratio = l1_ratio
#         self.groups = groups
#         self.scale_l2_by = scale_l2_by
#         self.eps = eps
#         self.n_alphas = n_alphas
#         self.alphas = alphas
#         self.fit_intercept = fit_intercept
#         self.normalize = normalize
#         self.max_iter = max_iter
#         self.tol = tol
#         self.copy_X = copy_X
#         self.cv = cv
#         self.verbose = verbose
#         self.n_jobs = n_jobs
#         self.random_state = random_state

#     def fit(self, X, y):
#         """Fit logistic sparse group lasso linear model

#         Fit is on grid of alphas and best alpha estimated by cross-validation.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             Training data. Pass directly as Fortran-contiguous data
#             to avoid unnecessary memory duplication. If y is mono-output,
#             X can be sparse.

#         y : array-like of shape (n_samples,) or (n_samples, n_targets)
#             Target values
#         """
#         # This makes sure that there is no duplication in memory.
#         # Dealing right with copy_X is important in the following:
#         # Multiple functions touch X and subsamples of X and can induce a
#         # lot of duplication of memory
#         copy_X = self.copy_X and self.fit_intercept

#         check_y_params = dict(
#             copy=False, dtype=[np.float64, np.float32], ensure_2d=False
#         )

#         if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
#             # Keep a reference to X
#             reference_to_old_X = X
#             # Let us not impose fortran ordering so far: it is
#             # not useful for the cross-validation loop and will be done
#             # by the model fitting itself

#             # Need to validate separately here.
#             # We can't pass multi_ouput=True because that would allow y to be
#             # csr. We also want to allow y to be 64 or 32 but check_X_y only
#             # allows to convert for 64.
#             check_X_params = dict(
#                 accept_sparse="csc", dtype=[np.float64, np.float32], copy=False
#             )
#             X, y = self._validate_data(
#                 X, y, validate_separately=(check_X_params, check_y_params)
#             )
#             if sparse.isspmatrix(X):
#                 if hasattr(reference_to_old_X, "data") and not np.may_share_memory(
#                     reference_to_old_X.data, X.data
#                 ):
#                     # X is a sparse matrix and has been copied
#                     copy_X = False
#             elif not np.may_share_memory(reference_to_old_X, X):
#                 # X has been copied
#                 copy_X = False
#             del reference_to_old_X
#         else:
#             # Need to validate separately here.
#             # We can't pass multi_ouput=True because that would allow y to be
#             # csr. We also want to allow y to be 64 or 32 but check_X_y only
#             # allows to convert for 64.
#             check_X_params = dict(
#                 accept_sparse="csc",
#                 dtype=[np.float64, np.float32],
#                 order="F",
#                 copy=copy_X,
#             )
#             X, y = self._validate_data(
#                 X, y, validate_separately=(check_X_params, check_y_params)
#             )
#             copy_X = False

#         if y.shape[0] == 0:
#             raise ValueError("y has 0 samples: %r" % y)

#         check_classification_targets(y)

#         # Encode for string labels
#         label_encoder = LabelEncoder().fit(y)
#         y = label_encoder.transform(y)

#         # The original class labels
#         classes = self.classes_ = label_encoder.classes_
#         encoded_labels = label_encoder.transform(label_encoder.classes_)

#         model = LogisticSGL()
#         y = column_or_1d(y, warn=True)

#         if X.shape[0] != y.shape[0]:
#             raise ValueError(
#                 "X and y have inconsistent dimensions (%d != %d)"
#                 % (X.shape[0], y.shape[0])
#             )

#         groups = check_groups(self.groups, X, allow_overlap=False, fit_intercept=False)

#         # All SGLCV parameters except "cv" and "n_jobs" are acceptable
#         path_params = self.get_params()
#         path_params.pop("cv", None)
#         path_params.pop("n_jobs", None)

#         l1_ratios = np.atleast_1d(path_params["l1_ratio"])
#         # For the first path, we need to set l1_ratio
#         path_params["l1_ratio"] = l1_ratios[0]

#         alphas = self.alphas
#         n_l1_ratio = len(l1_ratios)
#         if alphas is None:
#             alphas = [
#                 _alpha_grid(
#                     X=X,
#                     y=y,
#                     groups=groups,
#                     scale_l2_by=self.scale_l2_by,
#                     l1_ratio=l1_ratio,
#                     fit_intercept=self.fit_intercept,
#                     eps=self.eps,
#                     n_alphas=self.n_alphas,
#                     normalize=self.normalize,
#                     copy_X=self.copy_X,
#                 )
#                 for l1_ratio in l1_ratios
#             ]
#         else:
#             # Making sure alphas is properly ordered.
#             alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))

#         # We want n_alphas to be the number of alphas used for each l1_ratio.
#         n_alphas = len(alphas[0])
#         path_params.update({"n_alphas": n_alphas})

#         path_params["copy_X"] = copy_X
#         # We are not computing in parallel, we can modify X
#         # inplace in the folds
#         if effective_n_jobs(self.n_jobs) > 1:
#             path_params["copy_X"] = False

#         # "precompute" has no effect but it is expected by _path_residuals
#         path_params["precompute"] = False

#         if isinstance(self.verbose, int):
#             path_params["verbose"] = self.verbose - 1

#         # init cross-validation generator
#         cv = check_cv(self.cv)

#         # Compute path for all folds and compute MSE to get the best alpha
#         folds = list(cv.split(X, y))
#         best_mse = np.inf

#         # We do a double for loop folded in one, in order to be able to
#         # iterate in parallel on l1_ratio and folds
#         jobs = (
#             delayed(_path_residuals)(
#                 X,
#                 y,
#                 train,
#                 test,
#                 sgl_path,
#                 path_params,
#                 alphas=this_alphas,
#                 l1_ratio=this_l1_ratio,
#                 X_order="F",
#                 dtype=X.dtype.type,
#             )
#             for this_l1_ratio, this_alphas in zip(
#                 tqdm(l1_ratios, desc="L1_ratios", total=n_l1_ratio), alphas
#             )
#             for train, test in tqdm(folds, desc="CV folds", total=len(folds))
#         )

#         if isinstance(self.verbose, int):
#             parallel_verbosity = self.verbose - 2
#             if parallel_verbosity < 0:
#                 parallel_verbosity = 0
#         else:
#             parallel_verbosity = self.verbose

#         mse_paths = Parallel(
#             n_jobs=self.n_jobs,
#             verbose=parallel_verbosity,
#             **_joblib_parallel_args(prefer="threads")
#         )(jobs)

#         mse_paths = np.reshape(mse_paths, (n_l1_ratio, len(folds), -1))
#         mean_mse = np.mean(mse_paths, axis=1)
#         self.mse_path_ = np.squeeze(np.rollaxis(mse_paths, 2, 1))

#         for l1_ratio, l1_alphas, mse_alphas in zip(l1_ratios, alphas, mean_mse):
#             i_best_alpha = np.argmin(mse_alphas)
#             this_best_mse = mse_alphas[i_best_alpha]
#             if this_best_mse < best_mse:
#                 best_alpha = l1_alphas[i_best_alpha]
#                 best_l1_ratio = l1_ratio
#                 best_mse = this_best_mse

#         self.l1_ratio_ = best_l1_ratio
#         self.alpha_ = best_alpha

#         if self.alphas is None:
#             self.alphas_ = np.asarray(alphas)
#             if n_l1_ratio == 1:
#                 self.alphas_ = self.alphas_[0]
#         # Remove duplicate alphas in case alphas is provided.
#         else:
#             self.alphas_ = np.asarray(alphas[0])

#         # Refit the model with the parameters selected
#         common_params = {
#             name: value
#             for name, value in self.get_params().items()
#             if name in model.get_params()
#         }

#         model.set_params(**common_params)
#         model.alpha = best_alpha
#         model.l1_ratio = best_l1_ratio
#         model.copy_X = copy_X

#         model.fit(X, y)

#         self.coef_ = model.coef_
#         self.intercept_ = model.intercept_
#         self.n_iter_ = model.n_iter_
#         self.is_fitted_ = True
#         return self

#     @property
#     def chosen_features_(self):
#         """An index array of chosen features"""
#         return np.nonzero(self.coef_)[0]

#     @property
#     def sparsity_mask_(self):
#         """A boolean array indicating which features survived regularization"""
#         return self.coef_ != 0

#     def like_nonzero_mask_(self, rtol=1e-8):
#         """A boolean array indicating which features are zero or close to zero

#         Parameters
#         ----------
#         rtol : float
#             Relative tolerance. Any features that are larger in magnitude
#             than ``rtol`` times the mean coefficient value are considered
#             nonzero-like.
#         """
#         mean_abs_coef = abs(self.coef_.mean())
#         return np.abs(self.coef_) > rtol * mean_abs_coef

#     @property
#     def chosen_groups_(self):
#         """A set of the group IDs that survived regularization"""
#         if self.groups is not None:
#             group_mask = [
#                 bool(set(grp).intersection(set(self.chosen_features_)))
#                 for grp in self.groups
#             ]
#             return np.nonzero(group_mask)[0]
#         else:
#             return self.chosen_features_

#     def transform(self, X):
#         """Remove columns corresponding to zeroed-out coefficients"""
#         # Check is fit had been called
#         check_is_fitted(self, "is_fitted_")

#         # Input validation
#         X = check_array(X, accept_sparse=True)

#         # Check that the input is of the same shape as the one passed
#         # during fit.
#         if X.shape[1] != self.coef_.size:
#             raise ValueError("Shape of input is different from what was seen in `fit`")

#         return X[:, self.sparsity_mask_]

#     def _more_tags(self):  # pylint: disable=no-self-use
#         return {"multioutput": False, "requires_y": True}
