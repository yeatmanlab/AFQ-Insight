import numpy as np
import os.path as op
import pandas as pd
import pytest
import tempfile

import afqinsight as afqi
from afqinsight.datasets import load_afq_data, download_sarica, download_weston_havens

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_fetch():
    sarica_dir = download_sarica()
    X, y, groups, feature_names, group_names, subjects, _, _, _ = load_afq_data(
        workdir=sarica_dir,
        dwi_metrics=["md", "fa"],
        target_cols=["class"],
        label_encode_cols=["class"],
    )

    assert X.shape == (48, 4000)
    assert y.shape == (48,)
    assert len(groups) == 40
    assert len(feature_names) == 4000
    assert len(group_names) == 40
    assert len(subjects) == 48
    assert op.isfile(op.join(afqi.datasets.DATA_DIR, "sarica_data", "nodes.csv"))
    assert op.isfile(op.join(afqi.datasets.DATA_DIR, "sarica_data", "subjects.csv"))

    wh_dir = download_weston_havens()
    X, y, groups, feature_names, group_names, subjects, _, _, _ = load_afq_data(
        workdir=wh_dir, dwi_metrics=["md", "fa"], target_cols=["Age"]
    )

    assert X.shape == (77, 4000)
    assert y.shape == (77,)
    assert len(groups) == 40
    assert len(feature_names) == 4000
    assert len(group_names) == 40
    assert len(subjects) == 77
    assert op.isfile(op.join(afqi.datasets.DATA_DIR, "weston_havens_data", "nodes.csv"))
    assert op.isfile(
        op.join(afqi.datasets.DATA_DIR, "weston_havens_data", "subjects.csv")
    )

    with tempfile.TemporaryDirectory() as td:
        _ = download_sarica(data_home=td)
        _ = download_weston_havens(data_home=td)
        assert op.isfile(op.join(td, "sarica_data", "nodes.csv"))
        assert op.isfile(op.join(td, "sarica_data", "subjects.csv"))
        assert op.isfile(op.join(td, "weston_havens_data", "nodes.csv"))
        assert op.isfile(op.join(td, "weston_havens_data", "subjects.csv"))


def test_load_afq_data_smoke():
    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
    )
    assert len(output) == 9  # nosec
    assert output.sessions is None  # nosec
    assert output.bundle_means is None  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        return_sessions=True,
    )
    assert len(output) == 9  # nosec
    assert output.bundle_means is None  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
    )
    assert len(output) == 9  # nosec
    assert output.y is None  # nosec
    assert output.classes is None  # nosec
    assert output.sessions is None  # nosec
    assert output.bundle_means is None  # nosec

    output = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
        return_sessions=True,
    )
    assert len(output) == 9  # nosec
    assert output.y is None  # nosec
    assert output.classes is None  # nosec
    assert output.bundle_means is None  # nosec


def test_load_afq_data():
    (
        X,
        y,
        groups,
        feature_names,
        group_names,
        subjects,
        _,
        classes,
        bundle_means,
    ) = load_afq_data(
        workdir=test_data_path,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        return_bundle_means=True,
    )

    nodes = pd.read_csv(op.join(test_data_path, "nodes.csv"))
    X_ref = np.load(op.join(test_data_path, "test_transform_x.npy"))
    y_ref = np.load(op.join(test_data_path, "test_data_y.npy"))
    groups_ref = np.load(op.join(test_data_path, "test_transform_groups.npy"))
    cols_ref = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]
    means_ref = (
        nodes.groupby(["subjectID", "tractID"])
        .agg("mean")
        .drop("nodeID", axis="columns")
        .unstack("tractID")
        .to_numpy()
    )

    assert np.allclose(X, X_ref, equal_nan=True)  # nosec
    assert np.allclose(y, y_ref)  # nosec
    assert np.allclose(groups, groups_ref)  # nosec
    assert feature_names == cols_ref  # nosec
    assert group_names == [tup[0:2] for tup in cols_ref if tup[2] == 0]  # nosec
    assert set(subjects) == set(nodes.subjectID.unique())  # nosec
    assert all(classes["test_class"] == np.array(["c0", "c1"]))  # nosec
    assert np.allclose(bundle_means, means_ref, equal_nan=True)  # nosec

    with pytest.raises(ValueError):
        load_afq_data(
            workdir=test_data_path,
            target_cols=["test_class"],
            label_encode_cols=["test_class", "error"],
        )
    with pytest.raises(ValueError) as ee:
        load_afq_data(test_data_path)

    assert "please set `unsupervised=True`" in str(ee.value)  # nosec
