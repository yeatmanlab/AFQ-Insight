import groupyr as gpr
import pytest

from afqinsight import make_afq_classifier_pipeline, make_afq_regressor_pipeline
from afqinsight.pipeline import make_base_afq_pipeline
from afqinsight._serial_bagging import SerialBaggingClassifier, SerialBaggingRegressor
from sklearn.base import is_classifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import Normalizer, PowerTransformer, QuantileTransformer

scaler_args = [
    ("standard", StandardScaler),
    ("maxabs", MaxAbsScaler),
    ("minmax", MinMaxScaler),
    ("robust", RobustScaler),
    (Normalizer, Normalizer),
    (None, None),
]
imputer_args = [
    ("simple", SimpleImputer),
    ("knn", KNNImputer),
    (IterativeImputer, IterativeImputer),
]
power_args = [
    (True, PowerTransformer),
    (False, None),
    (QuantileTransformer, QuantileTransformer),
]
type_args = [
    (make_afq_classifier_pipeline, True, gpr.LogisticSGLCV),
    (make_afq_classifier_pipeline, False, gpr.LogisticSGL),
    (make_afq_regressor_pipeline, True, gpr.SGLCV),
    (make_afq_regressor_pipeline, False, gpr.SGL),
]
ensembler_args = [
    ("bagging", {True: BaggingClassifier, False: BaggingRegressor}),
    ("adaboost", {True: AdaBoostClassifier, False: AdaBoostRegressor}),
    ("serial-bagging", {True: SerialBaggingClassifier, False: SerialBaggingRegressor}),
    (AdaBoostClassifier, {True: AdaBoostClassifier, False: AdaBoostClassifier}),
    (None, None),
]


@pytest.mark.parametrize("scaler, ScalerStep", scaler_args)
@pytest.mark.parametrize("imputer, ImputerStep", imputer_args)
@pytest.mark.parametrize("feature_transformer, PowerStep", power_args)
@pytest.mark.parametrize("make_pipe, use_cv, EstimatorStep", type_args)
@pytest.mark.parametrize("target_transformer", [None, PowerTransformer])
@pytest.mark.parametrize("ensembler, EnsembleEstimators", ensembler_args)
def test_classifier_pipeline_steps(
    make_pipe,
    use_cv,
    EstimatorStep,
    scaler,
    ScalerStep,
    imputer,
    ImputerStep,
    feature_transformer,
    PowerStep,
    target_transformer,
    ensembler,
    EnsembleEstimators,
):
    pipeline = make_pipe(
        imputer=imputer,
        scaler=scaler,
        use_cv_estimator=use_cv,
        feature_transformer=feature_transformer,
        target_transformer=target_transformer,
        ensemble_meta_estimator=ensembler,
    )

    if scaler is not None:
        assert isinstance(pipeline.named_steps["scale"], ScalerStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["scale"].get_params() == ScalerStep().get_params()
        )
    else:
        assert pipeline.named_steps["scale"] is None  # nosec

    if imputer is not None:
        assert isinstance(pipeline.named_steps["impute"], ImputerStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["impute"].get_params() == ImputerStep().get_params()
        )
    else:
        assert pipeline.named_steps["impute"] is None  # nosec

    if feature_transformer:
        assert isinstance(pipeline.named_steps["feature_transform"], PowerStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["feature_transform"].get_params()
            == PowerStep().get_params()
        )
    else:
        assert pipeline.named_steps["feature_transform"] is None  # nosec

    if ensembler is not None:
        EnsembleStep = EnsembleEstimators[is_classifier(EstimatorStep())]

    if target_transformer is None:
        if ensembler is None:
            assert isinstance(pipeline.named_steps["estimate"], EstimatorStep)  # nosec
            assert (  # nosec
                pipeline.named_steps["estimate"].get_params()
                == EstimatorStep().get_params()
            )
        else:
            assert isinstance(pipeline.named_steps["estimate"], EnsembleStep)  # nosec
            ensemble_params = pipeline.named_steps["estimate"].get_params()
            correct_params = EnsembleStep(base_estimator=EstimatorStep()).get_params()
            ensemble_base_est = ensemble_params.pop("base_estimator")
            correct_params.pop("base_estimator")
            assert ensemble_params == correct_params  # nosec
            assert isinstance(ensemble_base_est, EstimatorStep)  # nosec
    else:
        if ensembler is None:
            assert isinstance(  # nosec
                pipeline.named_steps["estimate"].regressor, EstimatorStep
            )
            assert (  # nosec
                pipeline.named_steps["estimate"].regressor.get_params()
                == EstimatorStep().get_params()
            )
        else:
            assert isinstance(  # nosec
                pipeline.named_steps["estimate"].regressor, EnsembleStep
            )
            ensemble_params = pipeline.named_steps["estimate"].regressor.get_params()
            correct_params = EnsembleStep(base_estimator=EstimatorStep()).get_params()
            ensemble_base_est = ensemble_params.pop("base_estimator")
            correct_params.pop("base_estimator")
            assert ensemble_params == correct_params  # nosec
            assert isinstance(ensemble_base_est, EstimatorStep)  # nosec


def test_pipeline_value_errors():
    with pytest.raises(ValueError):
        make_base_afq_pipeline(scaler="error")

    with pytest.raises(ValueError):
        make_base_afq_pipeline(scaler=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(scaler=1729)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(imputer="error")

    with pytest.raises(ValueError):
        make_base_afq_pipeline(imputer=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(imputer=1729)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(feature_transformer=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(feature_transformer=1729)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(estimator=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(estimator=1729)

    with pytest.raises(ValueError):
        make_afq_regressor_pipeline(ensemble_meta_estimator="error")

    with pytest.raises(ValueError):
        make_afq_regressor_pipeline(ensemble_meta_estimator=object)

    with pytest.raises(ValueError):
        make_afq_regressor_pipeline(ensemble_meta_estimator=1729)


def test_base_pipeline_with_none_estimator():
    pipeline = make_base_afq_pipeline()
    assert pipeline.named_steps["estimate"] is None  # nosec


def test_base_pipeline_pass_kwargs():
    pipeline = make_base_afq_pipeline(scaler_kwargs={"with_mean": False})
    assert (  # nosec
        pipeline.named_steps["scale"].get_params()
        == StandardScaler(with_mean=False).get_params()
    )


def test_base_pipeline_pass_ensemble_kwargs():
    pipeline = make_afq_classifier_pipeline(
        ensemble_meta_estimator="bagging",
        ensemble_meta_estimator_kwargs={"n_estimators": 100},
    )
    ensemble_params = pipeline.named_steps["estimate"].get_params()
    assert ensemble_params["n_estimators"] == 100  # nosec
