"""sklearn-compatible pipelines for AFQ data."""
import inspect
import groupyr as gpr

from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin
from sklearn.base import is_classifier, is_regressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.preprocessing import PowerTransformer
from string import Template

from ._serial_bagging import SerialBaggingClassifier, SerialBaggingRegressor

__all__ = ["make_afq_classifier_pipeline", "make_afq_regressor_pipeline"]


def make_base_afq_pipeline(
    imputer="simple",
    scaler="standard",
    feature_transformer=False,
    estimator=None,
    ensemble_meta_estimator=None,
    imputer_kwargs=None,
    scaler_kwargs=None,
    feature_transformer_kwargs=None,
    estimator_kwargs=None,
    ensemble_meta_estimator_kwargs=None,
    memory=None,
    verbose=False,
    target_transformer=None,
    target_transform_func=None,
    target_transform_inverse_func=None,
    target_transform_check_inverse=True,
):
    """Return a base AFQ-specific modeling pipeline.

    This function returns a :ref:`Pipeline <sklearn:pipeline>` instance with the
    following steps::

        [imputer, scaler, feature_transformer, estimator]

    where ``imputer`` imputes missing data due to individual subjects missing
    metrics along an entire bundle; ``scaler`` is optional and scales the
    features of the feature matrix; ``feature_transformer`` is optional and
    applies a transform featurewise to make data more Gaussian-like;
    and ``estimator`` is a scikit-learn compatible estimator. The estimator
    may optionally be wrapped in an ensemble meta-estimator specified by
    ``ensemble_meta_estimator`` and given the keyword arguments in
    ``ensemble_meta_estimator_kwargs``. Additionally, the estimator may
    optionally be wrapped in
    ``sklearn:sklearn.compose.TransformedTargetRegressor``, such that the
    computation during ``fit`` is::

        estimator.fit(X, target_transform_func(y))

    or::

        estimator.fit(X, target_transformer.transform(y))

    The computation during ``predict`` is::

        target_transform_inverse_func(estimator.predict(X))

    or::

        target_transformer.inverse_transform(estimator.predict(X))

    This is a base function on which more specific classifier and regressor
    pipelines will be built.

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

    feature_transformer : bool or sklearn-compatible transformer, default=False
        An optional feature transformer for use on the feature matrix. If True,
        use :class:`sklearn:sklearn.preprocessing.PowerTransformer`. If
        False, skip this step. Custom transformers are allowed as long as
        they inherit from :class:`sklearn:sklearn.base.TransformerMixin`.

    estimator : sklearn-compatible estimator or None, default=None
        The estimator to use as the last step of the pipeline. If provided,
        it must inherit from :class:`sklearn:sklearn.base.BaseEstimator`

    ensemble_meta_estimator : "bagging", "adaboost", "serial-bagging", or None
        An optional ensemble meta-estimator to combine the predictions of
        several base estimators. "Adaboost" will result in the use of
        :class:`sklearn:sklearn.ensemble.AdaBoostClassifier` for classifier
        base estimators or
        :class:`sklearn.sklearn.ensemble.AdaBoostRegressor` for regressor
        base estimators. "Bagging" will result in the use of
        :class:`sklearn.sklearn.ensemble.BaggingClassifier` for classifier
        base estimators or :class:`sklearn.sklearn.ensemble.BaggingRegressor`
        for regressor base estimators.

    imputer_kwargs : dict, default=None,
        Key-word arguments for the imputer.

    scaler_kwargs : dict, default=None,
        Key-word arguments for the scaler.

    feature_transformer_kwargs : dict, default=None,
        Key-word arguments for the feature_transformer.

    estimator_kwargs : dict, default=None,
        Key-word arguments for the estimator.

    ensemble_meta_estimator_kwargs : dict, default=None,
        Key-word arguments for the ensemble meta-estimator.

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

    target_transformer : object, default=None
        Estimator object such as derived from
        :class:`sklearn.base.TransformerMixin`. Cannot be set at the same
        time as ``func`` and ``inverse_func``. If ``transformer`` is ``None``
        as well as ``func`` and ``inverse_func``, the transformer will be an
        identity transformer. Note that the transformer will be cloned during
        fitting. Also, the transformer is restricting ``y`` to be a numpy
        array.

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

    Returns
    -------
    pipeline : :ref:`Pipeline <sklearn:pipeline>` instance
    """
    base_msg = Template(
        "${kw} must be one of ${allowed} or a class that inherits "
        "from {base_class}; got ${input} instead."
    )
    transformer_msg = Template(
        base_msg.safe_substitute(base_class="sklearn.base.TransformerMixin")
    )
    ensembler_msg = Template(
        base_msg.safe_substitute(base_class="sklearn.base.MetaEstimatorMixin")
    )

    def call_with_kwargs(Transformer, kwargs):
        if kwargs is None:
            return Transformer()
        else:
            return Transformer(**kwargs)

    allowed = ["simple", "knn"]
    err_msg = Template(transformer_msg.safe_substitute(kw="imputer", allowed=allowed))
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
    err_msg = Template(transformer_msg.safe_substitute(kw="scaler", allowed=allowed))
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
        transformer_msg.safe_substitute(kw="feature_transformer", allowed=allowed)
    )
    if isinstance(feature_transformer, bool):
        if feature_transformer:
            pl_feature_transformer = call_with_kwargs(
                PowerTransformer, feature_transformer_kwargs
            )
        else:
            pl_feature_transformer = None
    elif inspect.isclass(feature_transformer):
        if issubclass(feature_transformer, TransformerMixin):
            pl_feature_transformer = call_with_kwargs(
                feature_transformer, feature_transformer_kwargs
            )
        else:
            raise ValueError(err_msg.substitute(input=feature_transformer))
    else:
        raise ValueError(err_msg.substitute(input=feature_transformer))

    if estimator is not None:
        if inspect.isclass(estimator) and issubclass(estimator, BaseEstimator):
            base_estimator = call_with_kwargs(estimator, estimator_kwargs)

            if ensemble_meta_estimator is not None:
                allowed = ["bagging", "adaboost", "serial-bagging"]
                err_msg = Template(
                    ensembler_msg.safe_substitute(
                        kw="ensemble_meta_estimator", allowed=allowed
                    )
                )
                if ensemble_meta_estimator_kwargs is not None:
                    ensembler_kwargs = ensemble_meta_estimator_kwargs.copy()
                else:
                    ensembler_kwargs = {}

                ensembler_kwargs["base_estimator"] = base_estimator

                if isinstance(ensemble_meta_estimator, str):
                    if ensemble_meta_estimator.lower() == "bagging":
                        if is_classifier(base_estimator):
                            ensembler = call_with_kwargs(
                                BaggingClassifier, ensembler_kwargs
                            )
                        elif is_regressor(base_estimator):
                            ensembler = call_with_kwargs(
                                BaggingRegressor, ensembler_kwargs
                            )
                    elif ensemble_meta_estimator.lower() == "serial-bagging":
                        if is_classifier(base_estimator):
                            ensembler = call_with_kwargs(
                                SerialBaggingClassifier, ensembler_kwargs
                            )
                        elif is_regressor(base_estimator):
                            ensembler = call_with_kwargs(
                                SerialBaggingRegressor, ensembler_kwargs
                            )
                    elif ensemble_meta_estimator.lower() == "adaboost":
                        if is_classifier(base_estimator):
                            ensembler = call_with_kwargs(
                                AdaBoostClassifier, ensembler_kwargs
                            )
                        elif is_regressor(base_estimator):
                            ensembler = call_with_kwargs(
                                AdaBoostRegressor, ensembler_kwargs
                            )
                    else:
                        raise ValueError(
                            err_msg.substitute(input=ensemble_meta_estimator)
                        )
                elif inspect.isclass(ensemble_meta_estimator):
                    if issubclass(ensemble_meta_estimator, MetaEstimatorMixin):
                        ensembler = call_with_kwargs(
                            ensemble_meta_estimator, ensembler_kwargs
                        )
                    else:
                        raise ValueError(
                            err_msg.substitute(input=ensemble_meta_estimator)
                        )
                else:
                    raise ValueError(err_msg.substitute(input=ensemble_meta_estimator))

                base_estimator = ensembler

            if any(
                [
                    target_transformer,
                    target_transform_func,
                    target_transform_inverse_func,
                ]
            ):
                pl_estimator = TransformedTargetRegressor(
                    base_estimator,
                    transformer=target_transformer,
                    func=target_transform_func,
                    inverse_func=target_transform_inverse_func,
                    check_inverse=target_transform_check_inverse,
                )
            else:
                pl_estimator = base_estimator
        else:
            raise ValueError(
                "If provided, estimator must inherit from sklearn.base.BaseEstimator; "
                "got {0} instead.".format(estimator)
            )
    else:
        pl_estimator = None

    # Build the pipeline steps. We will always start with the imputer and end
    # with the estimator. The scaler and feature_transform steps are optional.
    pl = [
        ("impute", pl_imputer),
        ("scale", pl_scaler),
        ("feature_transform", pl_feature_transformer),
        ("estimate", pl_estimator),
    ]

    return Pipeline(steps=pl, memory=memory, verbose=verbose)


def make_afq_classifier_pipeline(
    imputer="simple",
    scaler="standard",
    feature_transformer=False,
    ensemble_meta_estimator=None,
    imputer_kwargs=None,
    scaler_kwargs=None,
    feature_transformer_kwargs=None,
    ensemble_meta_estimator_kwargs=None,
    use_cv_estimator=True,
    memory=None,
    pipeline_verbosity=False,
    target_transformer=None,
    target_transform_func=None,
    target_transform_inverse_func=None,
    target_transform_check_inverse=True,
    **estimator_kwargs,
):
    """Return the recommended AFQ-specific classification pipeline.

    This function returns a :ref:`Pipeline <sklearn:pipeline>` instance with the
    following steps::

        [imputer, scaler, feature_transformer, estimator]

    where ``imputer`` imputes missing data due to individual subjects missing
    metrics along an entire bundle; ``scaler`` is optional and scales the
    features of the feature matrix; ``feature_transformer`` is optional and
    applies a transform featurewise to make data more Gaussian-like;
    and ``estimator`` is an instance of
    :class:`groupyr:groupyr.LogisticSGLCV` if ``use_cv_estimator=True`` or
    :class:`groupyr:groupyr.LogisticSGL` if ``use_cv_estimator=False``. The
    estimator may optionally be wrapped in an ensemble meta-estimator
    specified by ``ensemble_meta_estimator`` and given the keyword arguments
    in ``ensemble_meta_estimator_kwargs``. Additionally, the estimator may
    optionally be wrapped in
    ``sklearn:sklearn.compose.TransformedTargetRegressor``, such that the
    computation during ``fit`` is::

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

    feature_transformer : bool or sklearn-compatible transformer, default=False
        An optional transformer for use on the feature matrix. If True,
        use :class:`sklearn:sklearn.preprocessing.PowerTransformer`. If
        False, skip this step. Custom transformers are allowed as long as
        they inherit from :class:`sklearn:sklearn.base.TransformerMixin`.

    ensemble_meta_estimator : "bagging", "adaboost", or None
        An optional ensemble meta-estimator to combine the predictions of
        several base estimators. "Adaboost" will result in the use of
        :class:`sklearn:sklearn.ensemble.AdaBoostClassifier` and "bagging"
        will result in the use of
        :class:`sklearn.sklearn.ensemble.BaggingClassifier`.

    imputer_kwargs : dict, default=None,
        Key-word arguments for the imputer.

    scaler_kwargs : dict, default=None,
        Key-word arguments for the scaler.

    feature_transformer_kwargs : dict, default=None,
        Key-word arguments for the feature_transformer.

    ensemble_meta_estimator_kwargs : dict, default=None,
        Key-word arguments for the ensemble meta-estimator.

    use_cv_estimator : bool, default=True,
        If True, use :class:`groupyr:groupyr.LogisticSGLCV` as the final
        estimator. Otherwise, use :class:`groupyr:groupyr.LogisticSGL`.

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

    target_transformer : object, default=None
        Estimator object such as derived from
        :class:`sklearn.base.TransformerMixin`. Cannot be set at the same
        time as ``func`` and ``inverse_func``. If ``transformer`` is ``None``
        as well as ``func`` and ``inverse_func``, the transformer will be an
        identity transformer. Note that the transformer will be cloned during
        fitting. Also, the transformer is restricting ``y`` to be a numpy
        array.

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

    **estimator_kwargs : kwargs
        Keyword arguments passed to :class:`groupyr:groupyr.LogisticSGLCV` if
        ``use_cv_estimator=True`` or :class:`groupyr:groupyr.LogisticSGL` if
        ``use_cv_estimator=False``.

    Returns
    -------
    pipeline : :ref:`Pipeline <sklearn:pipeline>` instance
    """
    return make_base_afq_pipeline(
        imputer=imputer,
        scaler=scaler,
        feature_transformer=feature_transformer,
        estimator=gpr.LogisticSGLCV if use_cv_estimator else gpr.LogisticSGL,
        ensemble_meta_estimator=ensemble_meta_estimator,
        imputer_kwargs=imputer_kwargs,
        scaler_kwargs=scaler_kwargs,
        feature_transformer_kwargs=feature_transformer_kwargs,
        ensemble_meta_estimator_kwargs=ensemble_meta_estimator_kwargs,
        estimator_kwargs=estimator_kwargs,
        memory=memory,
        verbose=pipeline_verbosity,
        target_transformer=target_transformer,
        target_transform_func=target_transform_func,
        target_transform_inverse_func=target_transform_inverse_func,
        target_transform_check_inverse=target_transform_check_inverse,
    )


def make_afq_regressor_pipeline(
    imputer="simple",
    scaler="standard",
    feature_transformer=False,
    ensemble_meta_estimator=None,
    imputer_kwargs=None,
    scaler_kwargs=None,
    feature_transformer_kwargs=None,
    ensemble_meta_estimator_kwargs=None,
    use_cv_estimator=True,
    memory=None,
    pipeline_verbosity=False,
    target_transformer=None,
    target_transform_func=None,
    target_transform_inverse_func=None,
    target_transform_check_inverse=True,
    **estimator_kwargs,
):
    """Return the recommended AFQ-specific regression pipeline.

    This function returns a :ref:`Pipeline <sklearn:pipeline>` instance with the
    following steps::

        [imputer, scaler, feature_transformer, estimator]

    where ``imputer`` imputes missing data due to individual subjects missing
    metrics along an entire bundle; ``scaler`` is optional and scales the
    features of the feature matrix; ``feature_transformer`` is optional and
    applies a transform featurewise to make data more Gaussian-like;
    and ``estimator`` is an instance of :class:`groupyr:groupyr.SGLCV` if
    ``use_cv_estimator=True`` or :class:`groupyr:groupyr.SGL` if
    ``use_cv_estimator=False``. The estimator may optionally be wrapped in an
    ensemble meta-estimator specified by ``ensemble_meta_estimator`` and
    given the keyword arguments in ``ensemble_meta_estimator_kwargs``.
    Additionally, the estimator may optionally be wrapped in
    ``sklearn:sklearn.compose.TransformedTargetRegressor``, such that the
    computation during ``fit`` is::

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

    feature_transformer : bool or sklearn-compatible transformer, default=False
        An optional transformer for use on the feature matrix. If True,
        use :class:`sklearn:sklearn.preprocessing.PowerTransformer`. If
        False, skip this step. Custom transformers are allowed as long as
        they inherit from :class:`sklearn:sklearn.base.TransformerMixin`.

    ensemble_meta_estimator : "bagging", "adaboost", or None
        An optional ensemble meta-estimator to combine the predictions of
        several base estimators. "Adaboost" will result in the use of
        :class:`sklearn:sklearn.ensemble.AdaBoostClassifier` for classifier
        base estimators or
        :class:`sklearn.sklearn.ensemble.AdaBoostRegressor` for regressor
        base estimators. "Bagging" will result in the use of
        :class:`sklearn.sklearn.ensemble.BaggingClassifier` for classifier
        base estimators or :class:`sklearn.sklearn.ensemble.BaggingRegressor`
        for regressor base estimators.

    imputer_kwargs : dict, default=None,
        Key-word arguments for the imputer.

    scaler_kwargs : dict, default=None,
        Key-word arguments for the scaler.

    feature_transformer_kwargs : dict, default=None,
        Key-word arguments for the feature_transformer.

    use_cv_estimator : bool, default=True,
        If True, use :class:`groupyr:groupyr.SGLCV` as the final
        estimator. Otherwise, use :class:`groupyr:groupyr.SGL`.

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

    target_transformer : object, default=None
        Estimator object such as derived from
        :class:`sklearn.base.TransformerMixin`. Cannot be set at the same
        time as ``func`` and ``inverse_func``. If ``transformer`` is ``None``
        as well as ``func`` and ``inverse_func``, the transformer will be an
        identity transformer. Note that the transformer will be cloned during
        fitting. Also, the transformer is restricting ``y`` to be a numpy
        array.

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

    **estimator_kwargs : kwargs
        Keyword arguments passed to :class:`groupyr:groupyr.SGLCV` if
        ``use_cv_estimator=True`` or :class:`groupyr:groupyr.SGL` if
        ``use_cv_estimator=False``.

    Returns
    -------
    pipeline : :ref:`Pipeline <sklearn:pipeline>` instance
    """
    return make_base_afq_pipeline(
        imputer=imputer,
        scaler=scaler,
        feature_transformer=feature_transformer,
        ensemble_meta_estimator=ensemble_meta_estimator,
        estimator=gpr.SGLCV if use_cv_estimator else gpr.SGL,
        imputer_kwargs=imputer_kwargs,
        scaler_kwargs=scaler_kwargs,
        feature_transformer_kwargs=feature_transformer_kwargs,
        ensemble_meta_estimator_kwargs=ensemble_meta_estimator_kwargs,
        estimator_kwargs=estimator_kwargs,
        memory=memory,
        verbose=pipeline_verbosity,
        target_transformer=target_transformer,
        target_transform_func=target_transform_func,
        target_transform_inverse_func=target_transform_inverse_func,
        target_transform_check_inverse=target_transform_check_inverse,
    )
