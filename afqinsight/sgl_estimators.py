import contextlib
import copt as cp
import numpy as np
import warnings

from sklearn.base import BaseEstimator, RegressorMixin, is_classifier, is_regressor
from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .prox import SparseGroupL1

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


class SGLEstimator(BaseEstimator):
    """
    An sklearn compatible sparse group lasso estimator.

    This solves the sparse group lasso [1]_ problem for a feature matrix
    partitioned into groups using the proximal gradient descent (PGD)
    algorithm.

    Parameters
    ----------
    alpha : float, default=1.0
        Hyper-parameter : Combination between group lasso and lasso. alpha=0
        gives the group lasso and alpha=1 gives the lasso.

    lambd : float, default=1.0
        Hyper-parameter : overall regularization strength.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=5000
        Maximum number of iterations for PGD solver.

    tol : float, default=1e-6
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

    include_solver_trace : bool, default=False
        If True, include copt.utils.Trace() object in the attribue ``solver_trace_``.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`).

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.

    solver_trace_ : copt.utils.Trace
        This object traces convergence of the solver and can be useful for
        debugging. If the ``include_solver_trace`` parameter is False, this
        attribute is ``None``.

    Examples
    --------

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250

    """

    def __init__(
        self,
        alpha=1.0,
        lambd=1.0,
        fit_intercept=True,
        max_iter=5000,
        tol=1e-6,
        warm_start=False,
        verbose=0,
        suppress_solver_warnings=True,
        include_solver_trace=False,
    ):
        self.alpha = alpha
        self.lambd = lambd
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose
        self.suppress_solver_warnings = suppress_solver_warnings
        self.include_solver_trace = include_solver_trace

    def fit(self, X, y, groups=None, loss="squared_loss"):
        """Fit a linear model using the sparse group lasso

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        groups : {array-like}, total number of elements (n_features), default=None
            Array of non-overlapping indices for each group. For example, if
            nine features are grouped into equal contiguous groups of three,
            then groups would be an nd.array like [[0, 1, 2], [3, 4, 5], [6,
            7, 8]]. If None, all features will belong to their own singleton
            group.

        loss : ["squared_loss", "huber", "log"]
            The type of loss function to use in the PGD solver.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(self.warm_start, bool):
            raise ValueError(
                "The argument warm_start must be bool;"
                " got {0}".format(self.warm_start)
            )

        allowed_losses = ["squared_loss", "huber"]
        if is_classifier(self) and loss.lower() != "log":
            raise ValueError(
                "For classification, loss must be 'log';" "got {0}".format(loss)
            )
        elif is_regressor(self) and loss.lower() not in allowed_losses:
            raise ValueError(
                "For regression, the argument loss must be one of {0};"
                "got {1}".format(allowed_losses, loss)
            )

        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            y_numeric=not is_classifier(self),
            multi_output=False,
        )

        if is_classifier(self):
            check_classification_targets(y)
            self.classes_ = np.unique(y)
            y = np.logical_not(y == self.classes_[0]).astype(int)

        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])

        if self.warm_start and hasattr(self, "coef_"):
            if self.fit_intercept:
                coef = np.concatenate((np.array([self.intercept_]), self.coef_))
            else:
                coef = self.coef_
        else:
            if self.fit_intercept:
                coef = np.zeros(n_features + 1)
                # Initial bias condition gives 50/50 for binary classification
                coef[0] = 0.5
            else:
                coef = np.zeros(n_features)

        if loss == "huber":
            f = cp.utils.HuberLoss(X, y)
        elif loss == "log":
            f = cp.utils.LogLoss(X, y)
        else:
            f = cp.utils.SquareLoss(X, y)

        if self.include_solver_trace:
            self.solver_trace_ = cp.utils.Trace(f)
        else:
            self.solver_trace_ = None

        if self.suppress_solver_warnings:
            ctx_mgr = warnings.catch_warnings()
        else:
            ctx_mgr = contextlib.suppress()

        if groups is None:
            # If no groups provided, assign each feature to its own singleton group
            # e.g. for 5 features, groups = array([[0], [1], [2], [3], [4]])
            groups = np.arange(n_features).reshape((-1, 1))

        bias_index = 0 if self.fit_intercept else None
        sg1 = SparseGroupL1(self.alpha, self.lambd, groups, bias_index=bias_index)

        with ctx_mgr:
            # For some metaparameters, minimize_PGD might not reach the desired
            # tolerance level. This might be okay during hyperparameter
            # optimization. So ignore the warning if the user specifies
            # suppress_solver_warnings=True
            if self.suppress_solver_warnings:
                warnings.filterwarnings("ignore", category=RuntimeWarning)

            pgd = cp.minimize_proximal_gradient(
                f.f_grad,
                coef,
                sg1.prox,
                jac=True,
                step="backtracking",
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                callback=self.solver_trace_,
                accelerated=False,
            )

        if self.fit_intercept:
            self.intercept_ = pgd.x[0]
            self.coef_ = pgd.x[1:]
        else:
            # set intercept to zero as the other linear models do
            self.intercept_ = 0.0
            self.coef_ = pgd.x

        self.n_iter_ = pgd.nit

        self.is_fitted_ = True
        return self


@registered
class SGLRegressor(SGLEstimator, RegressorMixin, LinearModel):
    """
    An sklearn compatible sparse group lasso regressor.

    This solves the sparse group lasso [1]_ problem for a feature matrix
    partitioned into groups using the proximal gradient descent (PGD)
    algorithm.

    Parameters
    ----------
    alpha : float, default=1.0
        Hyper-parameter : Combination between group lasso and lasso. alpha=0
        gives the group lasso and alpha=1 gives the lasso.

    lambd : float, default=1.0
        Hyper-parameter : overall regularization strength.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=5000
        Maximum number of iterations for PGD solver.

    tol : float, default=1e-6
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

    def fit(self, X, y, groups=None, loss="squared_loss"):
        """Fit a linear model using the sparse group lasso

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        groups : {array-like}, total number of elements (n_features), default=None
            Array of non-overlapping indices for each group. For example, if
            nine features are grouped into equal contiguous groups of three,
            then groups would be an nd.array like [[0, 1, 2], [3, 4, 5], [6,
            7, 8]]. If None, all features will belong to their own singleton
            group.

        loss : ["squared_loss", "huber"]
            The type of loss function to use in the PGD solver.

        Returns
        -------
        self : object
            Returns self.
        """
        return super().fit(X=X, y=y, groups=groups, loss=loss)

    def predict(self, X):
        """Predict targets for test vectors in ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_


@registered
class SGLClassifier(SGLEstimator, LinearClassifierMixin):
    """
    An sklearn compatible sparse group lasso classifier.

    This solves the sparse group lasso [1]_ problem for a feature matrix
    partitioned into groups using the proximal gradient descent (PGD)
    algorithm.

    Parameters
    ----------
    alpha : float, default=1.0
        Hyper-parameter : Combination between group lasso and lasso. alpha=0
        gives the group lasso and alpha=1 gives the lasso.

    lambd : float, default=1.0
        Hyper-parameter : overall regularization strength.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=5000
        Maximum number of iterations for PGD solver.

    tol : float, default=1e-6
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

    def fit(self, X, y, groups=None):
        """Fit a linear model using the sparse group lasso

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        groups : {array-like}, total number of elements (n_features), default=None
            Array of non-overlapping indices for each group. For example, if
            nine features are grouped into equal contiguous groups of three,
            then groups would be an nd.array like [[0, 1, 2], [3, 4, 5], [6,
            7, 8]]. If None, all features will belong to their own singleton
            group.

        Returns
        -------
        self : object
            Returns self.
        """
        return super().fit(X=X, y=y, groups=groups, loss="log")

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

    def _more_tags(self):
        return {"binary_only": True}
