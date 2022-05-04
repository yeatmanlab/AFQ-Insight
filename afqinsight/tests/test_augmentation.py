import afqinsight.augmentation as aug
import numpy as np
import os.path as op
import pytest

from afqinsight.datasets import bundles2channels, download_sarica, AFQDataset
from inspect import signature
from sklearn.impute import SimpleImputer


@pytest.fixture
def sarica_Xy():
    sarica_dir = download_sarica()
    afq_data = AFQDataset.from_files(
        fn_nodes=op.join(sarica_dir, "nodes.csv"),
        fn_subjects=op.join(sarica_dir, "subjects.csv"),
        dwi_metrics=["md", "fa"],
        target_cols=["class"],
        label_encode_cols=["class"],
    )
    X = SimpleImputer(strategy="median").fit_transform(afq_data.X)
    X = bundles2channels(X, n_nodes=100, n_channels=40, channels_last=True)
    return X, afq_data.y


RANDOM_AUGMENTATION_METHODS = [
    aug.jitter,
    aug.scaling,
    aug.permutation,
    aug.magnitude_warp,
    aug.time_warp,
    aug.window_slice,
    aug.window_warp,
]

PATTERN_AUGMENTATION_METHODS = [
    aug.spawner,
    aug.wdba,
    aug.random_guided_warp,
    aug.discriminative_guided_warp,
]


@pytest.mark.parametrize("aug_method", RANDOM_AUGMENTATION_METHODS)
def test_random_smoke(sarica_Xy, aug_method):
    X, _ = sarica_Xy
    X_aug = aug_method(X)
    assert X.shape == X_aug.shape  # nosec

    parameters = signature(aug_method).parameters
    if "warp_channels_independently" in parameters:
        X_aug = aug_method(X, warp_channels_independently=True)
        assert X.shape == X_aug.shape  # nosec

    if "seg_mode" in parameters:
        X_aug = aug_method(X, seg_mode="random")
        assert X.shape == X_aug.shape  # nosec


@pytest.mark.parametrize("aug_method", PATTERN_AUGMENTATION_METHODS)
def test_pattern_smoke(sarica_Xy, aug_method):
    X, y = sarica_Xy
    X_aug = aug_method(x=X, labels=y)
    assert X.shape == X_aug.shape  # nosec

    y[10] = 2.0
    X_aug = aug_method(x=X, labels=y)
    assert X.shape == X_aug.shape  # nosec

    parameters = signature(aug_method).parameters
    if "dtw_type" in parameters:
        X_aug = aug_method(x=X, labels=y, dtw_type="shape")
        assert X.shape == X_aug.shape  # nosec


MEAN_ZERO_METHODS = [aug.jitter, aug.scaling, aug.magnitude_warp, aug.time_warp]


@pytest.mark.parametrize("aug_method", MEAN_ZERO_METHODS)
def test_augmentation_average(sarica_Xy, aug_method):
    n_aug = 10
    X, _ = sarica_Xy
    sigma = 1e-2
    aug_mean = np.mean(
        np.stack([aug_method(X, sigma=sigma) for _ in range(n_aug)]), axis=0
    )
    assert np.linalg.norm(X - aug_mean) / np.linalg.norm(X) <= sigma
