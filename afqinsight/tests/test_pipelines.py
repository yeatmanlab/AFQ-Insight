import groupyr as gpr
import pytest

from afqinsight import make_afq_classifier_pipeline, make_afq_regressor_pipeline
from afqinsight.pipeline import make_base_afq_pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._iterative import IterativeImputer

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


@pytest.mark.parametrize("scaler, ScalerStep", scaler_args)
@pytest.mark.parametrize("imputer, ImputerStep", imputer_args)
@pytest.mark.parametrize("power_transformer, PowerStep", power_args)
@pytest.mark.parametrize("make_pipe, use_cv, EstimatorStep", type_args)
@pytest.mark.parametrize("target_transformer", [None, PowerTransformer])
def test_classifier_pipeline_steps(
    make_pipe,
    use_cv,
    EstimatorStep,
    scaler,
    ScalerStep,
    imputer,
    ImputerStep,
    power_transformer,
    PowerStep,
    target_transformer,
):
    pipeline = make_pipe(
        imputer=imputer,
        scaler=scaler,
        use_cv_estimator=use_cv,
        power_transformer=power_transformer,
        target_transformer=target_transformer,
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

    if power_transformer:
        assert isinstance(pipeline.named_steps["power_transform"], PowerStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["power_transform"].get_params()
            == PowerStep().get_params()
        )
    else:
        assert pipeline.named_steps["power_transform"] is None  # nosec

    if target_transformer is None:
        assert isinstance(pipeline.named_steps["estimate"], EstimatorStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["estimate"].get_params()
            == EstimatorStep().get_params()
        )
    else:
        assert isinstance(  # nosec
            pipeline.named_steps["estimate"].regressor, EstimatorStep
        )
        assert (  # nosec
            pipeline.named_steps["estimate"].regressor.get_params()
            == EstimatorStep().get_params()
        )


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
        make_base_afq_pipeline(power_transformer=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(power_transformer=1729)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(estimator=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(estimator=1729)


def test_base_pipeline_with_none_estimator():
    pipeline = make_base_afq_pipeline()
    assert pipeline.named_steps["estimate"] is None  # nosec


def test_base_pipeline_pass_kwargs():
    pipeline = make_base_afq_pipeline(scaler_kwargs={"with_mean": False})
    assert (  # nosec
        pipeline.named_steps["scale"].get_params()
        == StandardScaler(with_mean=False).get_params()
    )
