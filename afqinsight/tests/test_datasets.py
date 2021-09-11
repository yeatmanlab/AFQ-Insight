import numpy as np
import os.path as op
import pandas as pd
import pytest
import tempfile
import torch

import afqinsight as afqi
from afqinsight.datasets import (
    bundles2channels,
    load_afq_data,
    download_sarica,
    download_weston_havens,
    AFQDataset,
)

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_bundles2channels():
    X0 = np.random.rand(50, 4000)
    X1 = bundles2channels(X0, n_nodes=100, n_channels=40, channels_last=True)
    assert X1.shape == (50, 100, 40)
    assert np.allclose(X0[:, :100], X1[:, :, 0])

    X1 = bundles2channels(X0, n_nodes=100, n_channels=40, channels_last=False)
    assert X1.shape == (50, 40, 100)
    assert np.allclose(X0[:, :100], X1[:, 0, :])

    with pytest.raises(ValueError):
        bundles2channels(X0, n_nodes=1000, n_channels=7)


@pytest.mark.parametrize("target_cols", [["class"], ["age", "class"]])
def test_AFQDataset(target_cols):
    sarica_dir = download_sarica()
    afq_data = AFQDataset(
        workdir=sarica_dir,
        dwi_metrics=["md", "fa"],
        target_cols=target_cols,
        label_encode_cols=["class"],
    )

    y_shape = (48, 2) if len(target_cols) == 2 else (48,)

    assert afq_data.X.shape == (48, 4000)  # nosec
    assert afq_data.y.shape == y_shape  # nosec
    assert len(afq_data.groups) == 40  # nosec
    assert len(afq_data.feature_names) == 4000  # nosec
    assert len(afq_data.group_names) == 40  # nosec
    assert len(afq_data.subjects) == 48  # nosec
    assert afq_data.bundle_means().shape == (48, 40)  # nosec

    # Test pytorch dataset method

    pt_dataset = afq_data.as_torch_dataset()
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 40, 100)  # nosec
    assert pt_dataset.y.shape == y_shape  # nosec
    assert np.allclose(
        pt_dataset[0][0][0], afq_data.X[0, :100], equal_nan=True
    )  # nosec

    pt_dataset = afq_data.as_torch_dataset(channels_last=True)
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 100, 40)  # nosec
    assert pt_dataset.y.shape == y_shape  # nosec
    assert np.allclose(
        pt_dataset[0][0][:, 0], afq_data.X[0, :100], equal_nan=True
    )  # nosec

    pt_dataset = afq_data.as_torch_dataset(bundles_as_channels=False)
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 4000)  # nosec
    assert pt_dataset.y.shape == y_shape  # nosec
    assert np.allclose(pt_dataset[0][0], afq_data.X[0], equal_nan=True)  # nosec

    # Test tensorflow dataset method

    tf_dataset = list(afq_data.as_tensorflow_dataset().as_numpy_iterator())
    assert len(tf_dataset) == 48
    assert np.allclose(
        tf_dataset[0][0][:, 0], afq_data.X[0, :100], equal_nan=True
    )  # nosec

    tf_dataset = list(
        afq_data.as_tensorflow_dataset(channels_last=False).as_numpy_iterator()
    )
    assert len(tf_dataset) == 48
    assert np.allclose(
        tf_dataset[0][0][0], afq_data.X[0, :100], equal_nan=True
    )  # nosec

    tf_dataset = list(
        afq_data.as_tensorflow_dataset(bundles_as_channels=False).as_numpy_iterator()
    )
    assert len(tf_dataset) == 48
    assert np.allclose(tf_dataset[0][0], afq_data.X[0], equal_nan=True)  # nosec

    # Test the drop_target_na method
    if len(target_cols) == 2:
        afq_data.y[0, 0] = np.nan
        y_shape = (47, 2)
    else:
        afq_data.y[0] = np.nan
        y_shape = (47,)

    afq_data.drop_target_na()
    assert afq_data.X.shape == (47, 4000)  # nosec
    assert afq_data.y.shape == y_shape  # nosec
    assert len(afq_data.subjects) == 47  # nosec

    # Do it all again for an unsupervised dataset

    afq_data = AFQDataset(
        workdir=sarica_dir, dwi_metrics=["md", "fa"], unsupervised=True
    )

    assert afq_data.X.shape == (48, 4000)  # nosec
    assert afq_data.y is None  # nosec
    assert len(afq_data.groups) == 40  # nosec
    assert len(afq_data.feature_names) == 4000  # nosec
    assert len(afq_data.group_names) == 40  # nosec
    assert len(afq_data.subjects) == 48  # nosec

    pt_dataset = afq_data.as_torch_dataset()
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 40, 100)  # nosec
    assert torch.all(torch.eq(pt_dataset.y, torch.tensor([])))  # nosec
    assert np.allclose(pt_dataset[0][0], afq_data.X[0, :100], equal_nan=True)  # nosec

    tf_dataset = list(afq_data.as_tensorflow_dataset().as_numpy_iterator())
    assert len(tf_dataset) == 48
    assert np.allclose(
        tf_dataset[0][:, 0], afq_data.X[0, :100], equal_nan=True
    )  # nosec

    # Test the drop_target_na method does nothing in the unsupervised case
    afq_data.drop_target_na()
    assert afq_data.X.shape == (48, 4000)  # nosec
    assert afq_data.y is None  # nosec
    assert len(afq_data.subjects) == 48  # nosec


def test_fetch():
    sarica_dir = download_sarica()

    with pytest.raises(ValueError):
        load_afq_data(
            workdir=sarica_dir,
            dwi_metrics=["md", "fa"],
            target_cols=["class"],
            label_encode_cols=["class"],
            concat_subject_session=True,
        )

    X, y, groups, feature_names, group_names, subjects, _, _ = load_afq_data(
        workdir=sarica_dir,
        dwi_metrics=["md", "fa"],
        target_cols=["class"],
        label_encode_cols=["class"],
    )

    assert X.shape == (48, 4000)  # nosec
    assert y.shape == (48,)  # nosec
    assert len(groups) == 40  # nosec
    assert len(feature_names) == 4000  # nosec
    assert len(group_names) == 40  # nosec
    assert len(subjects) == 48  # nosec
    assert op.isfile(
        op.join(afqi.datasets._DATA_DIR, "sarica_data", "nodes.csv")
    )  # nosec
    assert op.isfile(
        op.join(afqi.datasets._DATA_DIR, "sarica_data", "subjects.csv")
    )  # nosec

    wh_dir = download_weston_havens()
    X, y, groups, feature_names, group_names, subjects, _, _ = load_afq_data(
        workdir=wh_dir, dwi_metrics=["md", "fa"], target_cols=["Age"]
    )

    assert X.shape == (77, 4000)  # nosec
    assert y.shape == (77,)  # nosec
    assert len(groups) == 40  # nosec
    assert len(feature_names) == 4000  # nosec
    assert len(group_names) == 40  # nosec
    assert len(subjects) == 77  # nosec
    assert op.isfile(
        op.join(afqi.datasets._DATA_DIR, "weston_havens_data", "nodes.csv")
    )  # nosec
    assert op.isfile(
        op.join(afqi.datasets._DATA_DIR, "weston_havens_data", "subjects.csv")
    )  # nosec

    with tempfile.TemporaryDirectory() as td:
        _ = download_sarica(data_home=td)
        _ = download_weston_havens(data_home=td)
        assert op.isfile(op.join(td, "sarica_data", "nodes.csv"))  # nosec
        assert op.isfile(op.join(td, "sarica_data", "subjects.csv"))  # nosec
        assert op.isfile(op.join(td, "weston_havens_data", "nodes.csv"))  # nosec
        assert op.isfile(op.join(td, "weston_havens_data", "subjects.csv"))  # nosec


def test_load_afq_data_smoke():
    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
    )
    assert len(output) == 8  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
    )
    assert len(output) == 8  # nosec
    assert output.y is None  # nosec
    assert output.classes is None  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
    )
    assert len(output) == 8  # nosec
    assert output.y is None  # nosec
    assert output.classes is None  # nosec


def test_load_afq_data():
    (X, y, groups, feature_names, group_names, subjects, _, classes) = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        return_bundle_means=False,
    )

    nodes = pd.read_csv(op.join(test_data_path, "nodes.csv"))
    X_ref = np.load(op.join(test_data_path, "test_transform_x.npy"))
    y_ref = np.load(op.join(test_data_path, "test_data_y.npy"))
    groups_ref = np.load(op.join(test_data_path, "test_transform_groups.npy"))
    cols_ref = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]

    assert np.allclose(X, X_ref, equal_nan=True)  # nosec
    assert np.allclose(y, y_ref)  # nosec
    assert np.allclose(groups, groups_ref)  # nosec
    assert feature_names == cols_ref  # nosec
    assert group_names == [tup[0:2] for tup in cols_ref if tup[2] == 0]  # nosec
    assert set(subjects) == set(nodes.subjectID.unique())  # nosec
    assert all(classes["test_class"] == np.array(["c0", "c1"]))  # nosec

    (X, y, groups, feature_names, group_names, subjects, _, classes) = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        return_bundle_means=True,
    )

    means_ref = (
        nodes.groupby(["subjectID", "tractID"])
        .agg("mean")
        .drop("nodeID", axis="columns")
        .unstack("tractID")
    )
    assert np.allclose(X, means_ref.to_numpy(), equal_nan=True)  # nosec
    assert group_names == means_ref.columns.to_list()  # nosec
    assert feature_names == means_ref.columns.to_list()  # nosec
    assert set(subjects) == set(nodes.subjectID.unique())  # nosec

    with pytest.raises(ValueError):
        load_afq_data(
            workdir=test_data_path,
            target_cols=["test_class"],
            label_encode_cols=["test_class", "error"],
        )
    with pytest.raises(ValueError) as ee:
        load_afq_data(test_data_path)

    assert "please set `unsupervised=True`" in str(ee.value)  # nosec
