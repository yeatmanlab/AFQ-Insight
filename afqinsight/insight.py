"""sklearn-compatible pipelines for AFQ data."""
import inspect
import groupyr as gr

from string import Template
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

__all__ = ["AFQClassifierPipeline", "AFQRegressorPipeline"]


class _BaseAFQPipeline(Pipeline):
    """The base AFQ-specific modeling pipeline.

    This class returns an instance of
    :ref:`sklearn:sklearn.pipeline.Pipeline` with the following steps::
        
        [imputer, scaler, power_transformer, estimator]

    where ``imputer`` imputes missing data due to individual subjects missing
    metrics along an entire bundle; ``scaler`` is optional and scales the
    features of the feature matrix; ``power_transformer`` is optional and
    applies a power transform featurewise to make data more Gaussian-like;
    and ``estimator`` is a scikit-learn compatible estimator. The estimator
    may be optionally wrapped in
    ``sklearn:sklearn.compose.TransformedTargetRegressor``, such that

    The computation during ``fit`` is::

        estimator.fit(X, target_transform_func(y))
        
    or::

        estimator.fit(X, target_transformer.transform(y))

    The computation during ``predict`` is::
    
        target_transform_inverse_func(estimator.predict(X))

    or::

        target_transformer.inverse_transform(estimator.predict(X))

    This is a base class on which more specific classifiers and regressors
    will be built.

    Parameters
    ----------
    imputer : "simple", "knn", or sklearn-compatible transformer, default="simple"
        The imputer for missing data. String arguments result in the use of
        transformers from ``sklearn.impute``. "simple" yields the
        ``SimpleImputer``. "knn" yields the ``KNNImputer``. Custom
        transformers are allowed as long as they inherit from
        ``sklearn.base.TransformerMixin``.

    scaler : "standard", "minmax", "maxabs", "robust", or sklearn-compatible transformer, default="standard"
        The scaler to use for the feature matrix. String arguments result in
        the use of transformers from ``sklearn.preprocessing``. "standard"
        yields the ``StandardScalar``. "minmax" yields the ``MinMaxScaler``.
        "maxabs" yields the ``MaxAbsScaler``. "robust" yields the
        ``RobustScaler``. Custom transformers are allowed as long as they
        inherit from ``sklearn.base.TransformerMixin``.

    power_transformer : bool or sklearn-compatible transformer, default=False
        An optional power transformer for use on the feature matrix. If True,
        use :class:`sklearn:sklearn.preprocessing.PowerTransformer`. If
        False, skip this step. Custom transformers are allowed as long as
        they inherit from :class:`sklearn:sklearn.base.TransformerMixin`.

    estimator : sklearn-compatible estimator or None, default=None
        The estimator to use as the last step of the pipeline. If provided,
        it must inherit from ``sklearn.base.BaseEstimator``

    imputer_kwargs : dict, default=None,
        Key-word arguments for the imputer.

    scaler_kwargs : dict, default=None,
        Key-word arguments for the scaler.

    power_transformer_kwargs : dict, default=None,
        Key-word arguments for the power_transformer.

    estimator_kwargs : dict, default=None,
        Key-word arguments for the estimator.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.
    
    target_transform_transformer : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Cannot be
        set at the same time as ``func`` and ``inverse_func``. If
        ``transformer`` is ``None`` as well as ``func`` and ``inverse_func``,
        the transformer will be an identity transformer. Note that the
        transformer will be cloned during fitting. Also, the transformer is
        restricting ``y`` to be a numpy array.

    target_transform_func : function, default=None
        Function to apply to ``y`` before passing to ``fit``. Cannot be set at
        the same time as ``transformer``. The function needs to return a
        2-dimensional array. If ``func`` is ``None``, the function used will be
        the identity function.

    target_transform_inverse_func : function, default=None
        Function to apply to the prediction of the regressor. Cannot be set at
        the same time as ``transformer`` as well. The function needs to return
        a 2-dimensional array. The inverse function is used to return
        predictions to the same space of the original training labels.

    target_transform_check_inverse : bool, default=True
        Whether to check that ``transform`` followed by ``inverse_transform``
        or ``func`` followed by ``inverse_func`` leads to the original targets.

    Attributes
    ----------
    named_steps : `sklearn.utils.Bunch <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn-utils-bunch>`_
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    def __init__(
        self,
        imputer="simple",
        scaler="standard",
        power_transformer=False,
        estimator=None,
        imputer_kwargs=None,
        scaler_kwargs=None,
        power_transformer_kwargs=None,
        estimator_kwargs=None,
        memory=None,
        verbose=False,
        target_transformer=None,
        target_transform_func=None,
        target_transform_inverse_func=None,
        target_transform_check_inverse=True,
    ):
        base_msg = Template(
            "${kw} must be one of ${allowed} or a class that inherits "
            "from sklearn.base.TransformerMixin; got ${input} instead."
        )

        def call_with_kwargs(Transformer, kwargs):
            if kwargs is None:
                return Transformer()
            else:
                return Transformer(**kwargs)

        allowed = ["simple", "knn"]
        err_msg = Template(base_msg.safe_substitute(kw="imputer", allowed=allowed))
        if isinstance(imputer, str):
            if imputer.lower() == "simple":
                pl_imputer = call_with_kwargs(SimpleImputer, imputer_kwargs)
            elif imputer.lower() == "knn":
                pl_imputer = call_with_kwargs(KNNImputer, imputer_kwargs)
            else:
                raise ValueError(err_msg.substitute(input=imputer))
        elif inspect.isclass(imputer):
            if issubclass(imputer, TransformerMixin):
                pl_imputer = call_with_kwargs(imputer, imputer_kwargs)
            else:
                raise ValueError(err_msg.substitute(input=imputer))
        else:
            raise ValueError(err_msg.substitute(input=imputer))

        allowed = ["standard", "minmax", "maxabs", "robust"]
        err_msg = Template(base_msg.safe_substitute(kw="scaler", allowed=allowed))
        if isinstance(scaler, str):
            if scaler.lower() == "standard":
                pl_scaler = call_with_kwargs(StandardScaler, scaler_kwargs)
            elif scaler.lower() == "minmax":
                pl_scaler = call_with_kwargs(MinMaxScaler, scaler_kwargs)
            elif scaler.lower() == "maxabs":
                pl_scaler = call_with_kwargs(MaxAbsScaler, scaler_kwargs)
            elif scaler.lower() == "robust":
                pl_scaler = call_with_kwargs(RobustScaler, scaler_kwargs)
            else:
                raise ValueError(err_msg.substitute(input=scaler))
        elif inspect.isclass(scaler):
            if issubclass(scaler, TransformerMixin):
                pl_scaler = call_with_kwargs(scaler, scaler_kwargs)
            else:
                raise ValueError(err_msg.substitute(input=scaler))
        elif scaler is None:
            pl_scaler = None
        else:
            raise ValueError(err_msg.substitute(input=scaler))

        allowed = [True, False]
        err_msg = Template(
            base_msg.safe_substitute(kw="power_transformer", allowed=allowed)
        )
        if isinstance(power_transformer, bool):
            if power_transformer:
                pl_power_transformer = call_with_kwargs(
                    PowerTransformer, power_transformer_kwargs
                )
            else:
                pl_power_transformer = None
        elif inspect.isclass(power_transformer):
            if issubclass(power_transformer, TransformerMixin):
                pl_power_transformer = call_with_kwargs(
                    power_transformer, power_transformer_kwargs
                )
            else:
                raise ValueError(err_msg.substitute(input=power_transformer))
        else:
            raise ValueError(err_msg.substitute(input=power_transformer))

        if estimator is not None:
            if issubclass(estimator, BaseEstimator):
                pl_estimator = call_with_kwargs(estimator, estimator_kwargs)
            else:
                raise ValueError(
                    "If provided, estimator must inherit from sklearn.base.BaseEstimator; "
                    "got {0} instead.".format(estimator)
                )
        else:
            pl_estimator = None

        # Build the pipeline steps. We will always start with the imputer and end
        # with the estimator. The scaler and power_transform steps are optional.
        pl = [
            ("impute", pl_imputer),
            ("scale", pl_scaler),
            ("power_transform", pl_power_transformer),
            ("estimate", pl_estimator),
        ]

        super().__init__(steps=pl, memory=memory, verbose=verbose)


class AFQClassifierPipeline(_BaseAFQPipeline):
    """The recommended AFQ-specific classification pipeline.

    This class returns a :ref:`Pipeline <sklearn:pipeline>` instance with the
    following steps::
        
        [imputer, scaler, power_transformer, estimator]

    where ``imputer`` imputes missing data due to individual subjects missing
    metrics along an entire bundle; ``scaler`` is optional and scales the
    features of the feature matrix; ``power_transformer`` is optional and
    applies a power transform featurewise to make data more Gaussian-like;
    and ``estimator`` is an instance of
    :class:`groupyr:groupyr.LogisticSGLCV`. The parameters below include the
    keyword arguments passed to :class:`groupyr:groupyr.LogisticSGLCV`. The
    estimator may be optionally wrapped in a
    :class:`sklearn:sklearn.compose.TransformedTargetRegressor` such that
    the computation during ``fit`` is::

        estimator.fit(X, target_transform_func(y))
        
    or::

        estimator.fit(X, target_transformer.transform(y))

    The computation during ``predict`` is::
    
        target_transform_inverse_func(estimator.predict(X))

    or::

        target_transformer.inverse_transform(estimator.predict(X))

    Parameters
    ----------
    imputer : "simple", "knn", or sklearn-compatible transformer, default="simple"
        The imputer for missing data. String arguments result in the use of
        specific imputers/transformers:
        "simple" yields :class:`sklearn:sklearn.impute.SimpleImputer`;
        "knn" yields :class:`sklearn:sklearn.impute.KNNImputer`.
        Custom transformers are
        allowed as long as they inherit from
        :class:`sklearn:sklearn.base.TransformerMixin`.

    scaler : "standard", "minmax", "maxabs", "robust", or sklearn-compatible transformer, default="standard"
        The scaler to use for the feature matrix. String arguments result in
        the use of specific transformers: "standard" yields the
        :class:`sklearn:sklearn.preprocessing.StandardScalar`; "minmax"
        yields the :class:`sklearn:sklearn.preprocessing.MinMaxScaler`;
        "maxabs" yields the
        :class:`sklearn:sklearn.preprocessing.MaxAbsScaler`; "robust" yields
        the :class:`sklearn:sklearn.preprocessing.RobustScaler`. Custom
        transformers are allowed as long as they inherit from
        :class:`sklearn:sklearn.base.TransformerMixin`.

    power_transformer : bool or sklearn-compatible transformer, default=False
        An optional power transformer for use on the feature matrix. If True,
        use :class:`sklearn:sklearn.preprocessing.PowerTransformer`. If
        False, skip this step. Custom transformers are allowed as long as
        they inherit from :class:`sklearn:sklearn.base.TransformerMixin`.

    imputer_kwargs : dict, default=None,
        Key-word arguments for the imputer.

    scaler_kwargs : dict, default=None,
        Key-word arguments for the scaler.

    power_transformer_kwargs : dict, default=None,
        Key-word arguments for the power_transformer.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    pipeline_verbosity : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    l1_ratio : float or list of float, default=1.0
        float between 0 and 1 passed to SGL (scaling between group lasso and
        lasso penalties). For ``l1_ratio = 0`` the penalty is the group lasso
        penalty. For ``l1_ratio = 1`` it is the lasso penalty. For ``0 <
        l1_ratio < 1``, the penalty is a combination of group lasso and
        lasso. This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of values
        will depend on the problem. For problems where we expect strong
        overall sparsity and would like to encourage grouping, put more
        values close to 1 (i.e. Lasso). In contrast, if we expect strong
        group-wise sparsity, but only mild sparsity within groups, put more
        values close to 0 (i.e. group lasso).

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path, used for each l1_ratio.

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically

    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    max_iter : int, default=1000
        The maximum number of iterations

    tol : float, default=1e-7
        The tolerance for the SGL solver

    scoring : callable, default=None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions that can
        be used, look at :mod:`sklearn:sklearn.metrics`. The default scoring
        option used is accuracy_score.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - an sklearn `CV splitter <https://scikit-learn.org/stable/glossary.html#term-cv-splitter>`_,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`sklearn:sklearn.model_selection.StratifiedKFold` is used.

        Refer to the :ref:`scikit-learn User Guide
        <sklearn:cross_validation>` for the various cross-validation
        strategies that can be used here.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=False
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation. ``None`` means 1
        unless in a :obj:`joblib:joblib.parallel_backend` context. ``-1`` means
        using all processors. See :term:`sklearn:n_jobs` for more details.

    Attributes
    ----------
    named_steps : `sklearn.utils.Bunch <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn-utils-bunch>`_
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    def __init__(
        self,
        imputer="simple",
        scaler="standard",
        power_transformer=False,
        imputer_kwargs=None,
        scaler_kwargs=None,
        power_transformer_kwargs=None,
        memory=None,
        pipeline_verbosity=False,
        **logistic_sglcv_kwargs,
    ):
        super().__init__(
            imputer=imputer,
            scaler=scaler,
            power_transformer=power_transformer,
            estimator=gr.LogisticSGLCV,
            imputer_kwargs=imputer_kwargs,
            scaler_kwargs=scaler_kwargs,
            power_transformer_kwargs=power_transformer_kwargs,
            estimator_kwargs=logistic_sglcv_kwargs,
            memory=memory,
            verbose=pipeline_verbosity,
        )


class AFQRegressorPipeline(_BaseAFQPipeline):
    """The recommended AFQ-specific regression pipeline.

    This class returns a :ref:`Pipeline <sklearn:pipeline>` instance with the
    following steps::
        
        [imputer, scaler, power_transformer, estimator]

    where ``imputer`` imputes missing data due to individual subjects missing
    metrics along an entire bundle; ``scaler`` is optional and scales the
    features of the feature matrix; ``power_transformer`` is optional and
    applies a power transform featurewise to make data more Gaussian-like;
    and ``estimator`` is an instance of
    :class:`groupyr:groupyr.SGLCV`. The parameters below include the
    keyword arguments passed to :class:`groupyr:groupyr.SGLCV`. The
    estimator may be optionally wrapped in a
    :class:`sklearn:sklearn.compose.TransformedTargetRegressor` such that
    the computation during ``fit`` is::

        estimator.fit(X, target_transform_func(y))
        
    or::

        estimator.fit(X, target_transformer.transform(y))

    The computation during ``predict`` is::
    
        target_transform_inverse_func(estimator.predict(X))

    or::

        target_transformer.inverse_transform(estimator.predict(X))

    Parameters
    ----------
    imputer : "simple", "knn", or sklearn-compatible transformer, default="simple"
        The imputer for missing data. String arguments result in the use of
        specific imputers/transformers:
        "simple" yields :class:`sklearn:sklearn.impute.SimpleImputer`;
        "knn" yields :class:`sklearn:sklearn.impute.KNNImputer`.
        Custom transformers are
        allowed as long as they inherit from
        :class:`sklearn:sklearn.base.TransformerMixin`.

    scaler : "standard", "minmax", "maxabs", "robust", or sklearn-compatible transformer, default="standard"
        The scaler to use for the feature matrix. String arguments result in
        the use of specific transformers: "standard" yields the
        :class:`sklearn:sklearn.preprocessing.StandardScalar`; "minmax"
        yields the :class:`sklearn:sklearn.preprocessing.MinMaxScaler`;
        "maxabs" yields the
        :class:`sklearn:sklearn.preprocessing.MaxAbsScaler`; "robust" yields
        the :class:`sklearn:sklearn.preprocessing.RobustScaler`. Custom
        transformers are allowed as long as they inherit from
        :class:`sklearn:sklearn.base.TransformerMixin`.

    power_transformer : bool or sklearn-compatible transformer, default=False
        An optional power transformer for use on the feature matrix. If True,
        use :class:`sklearn:sklearn.preprocessing.PowerTransformer`. If
        False, skip this step. Custom transformers are allowed as long as
        they inherit from :class:`sklearn:sklearn.base.TransformerMixin`.

    imputer_kwargs : dict, default=None,
        Key-word arguments for the imputer.

    scaler_kwargs : dict, default=None,
        Key-word arguments for the scaler.

    power_transformer_kwargs : dict, default=None,
        Key-word arguments for the power_transformer.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    pipeline_verbosity : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    l1_ratio : float or list of float, default=1.0
        float between 0 and 1 passed to SGL (scaling between group lasso and
        lasso penalties). For ``l1_ratio = 0`` the penalty is the group lasso
        penalty. For ``l1_ratio = 1`` it is the lasso penalty. For ``0 <
        l1_ratio < 1``, the penalty is a combination of group lasso and
        lasso. This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of values
        will depend on the problem. For problems where we expect strong
        overall sparsity and would like to encourage grouping, put more
        values close to 1 (i.e. Lasso). In contrast, if we expect strong
        group-wise sparsity, but only mild sparsity within groups, put more
        values close to 0 (i.e. group lasso).

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path, used for each l1_ratio.

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically

    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    max_iter : int, default=1000
        The maximum number of iterations

    tol : float, default=1e-7
        The tolerance for the SGL solver

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - an sklearn `CV splitter <https://scikit-learn.org/stable/glossary.html#term-cv-splitter>`_,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`sklearn:sklearn.model_selection.KFold` is used.

        Refer to the :ref:`scikit-learn User Guide
        <sklearn:cross_validation>` for the various cross-validation
        strategies that can be used here.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=0
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation. ``None`` means 1
        unless in a :obj:`joblib:joblib.parallel_backend` context. ``-1`` means
        using all processors. See :term:`sklearn:n_jobs` for more details.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    named_steps : `sklearn.utils.Bunch <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn-utils-bunch>`_
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    def __init__(
        self,
        imputer="simple",
        scaler="standard",
        power_transformer=False,
        imputer_kwargs=None,
        scaler_kwargs=None,
        power_transformer_kwargs=None,
        memory=None,
        pipeline_verbosity=False,
        **sglcv_kwargs,
    ):
        super().__init__(
            imputer=imputer,
            scaler=scaler,
            power_transformer=power_transformer,
            estimator=gr.SGLCV,
            imputer_kwargs=imputer_kwargs,
            scaler_kwargs=scaler_kwargs,
            power_transformer_kwargs=power_transformer_kwargs,
            estimator_kwargs=sglcv_kwargs,
            memory=memory,
            verbose=pipeline_verbosity,
        )


# TODO: Use these classification scores: accuracy, roc_auc, avg_precision, f1
# TODO: Use these regression scores: rmse - RMSE score, R^2, medae - The median absolute error
# TODO: Default to cv=3 in the inner CV loop
# TODO: Default to n_splits=10, n_repeats=3 for outer CV loop
# TODO: Write outer loop CV using sklearn's cross_validate function
# TODO: Allow option to use FeatureTransformer to wrap the above pipelines
