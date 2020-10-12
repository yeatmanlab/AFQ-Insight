"""Generate samples of synthetic data sets or extract AFQ data."""
import numpy as np
import os.path as op
import pandas as pd
from collections import namedtuple
from shutil import copyfile

from .transform import AFQDataFrameMapper

__all__ = ["load_afq_data", "output_beta_to_afq"]


def load_afq_data(
    workdir,
    target_cols,
    binary_positives=None,
    fn_nodes="nodes.csv",
    fn_subjects="subjects.csv",
):
    """Load AFQ data from CSV, transform it, return feature matrix and target.

    Parameters
    ----------
    workdir : str
        Directory in which to find the AFQ csv files

    target_cols : list of strings
        List of column names in subjects csv file to use as target variables

    binary_positives : list or dict of string values, or None, default=None
        If supplied, use these values as the positive value in a binary
        classification problem. If this is a list, it must have the same
        length as `target cols`. If this is a dict, it must have a key
        for each item in `target_cols`. If None, do not use a binary mapping
        (e.g. for a regression problem).

    fn_nodes : str, default='nodes.csv'
        Filename for the nodes csv file.

    fn_subjects : str, default='subjects.csv'
        Filename for the subjects csv file.

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        The feature samples.

    y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
        Target values.

    groups : list of numpy.ndarray
        feature indices for each feature group

    feature_names : list of tuples
        The multi-indexed columns of X

    subjects : list
        Subject IDs

    See Also
    --------
    transform.AFQDataFrameMapper
    """
    workdir = op.abspath(workdir)
    fn_nodes = op.join(workdir, fn_nodes)
    fn_subjects = op.join(workdir, fn_subjects)

    nodes = pd.read_csv(fn_nodes)
    targets = pd.read_csv(fn_subjects, index_col="subjectID").drop(
        ["Unnamed: 0"], axis="columns"
    )

    y = targets[target_cols]

    if binary_positives is not None:
        if not isinstance(binary_positives, dict):
            binary_positives = {
                key: val for key, val in zip(target_cols, binary_positives)
            }

        for col in y.columns:
            y.loc[:, col] = y[col].map(lambda c: int(c == binary_positives[col])).values

    y = y.values.flatten()

    mapper = AFQDataFrameMapper()
    X = mapper.fit_transform(nodes)
    groups = mapper.groups_
    feature_names = mapper.feature_names_
    subjects = mapper.subjects_

    return X, y, groups, feature_names, subjects


def output_beta_to_afq(
    beta_hat,
    columns,
    workdir_in,
    workdir_out,
    fn_nodes_in="nodes.csv",
    fn_subjects_in="subjects.csv",
    fn_nodes_out="nodes.csv",
    fn_subjects_out="subjects.csv",
    scale_beta=False,
):
    """Output coefficients to AFQ data format.

    Parameters
    ----------
    workdir_in : str
        Directory in which to find the input AFQ csv files

    workdir_out : str
        Directory in which to save the output AFQ csv files

    fn_nodes_in : str, default='nodes.csv'
        Filename for the input nodes csv file.

    fn_subjects_in : str, default='subjects.csv'
        Filename for the input subjects csv file.

    fn_nodes_out : str, default='nodes.csv'
        Filename for the output nodes csv file.

    fn_subjects_out : str, default='subjects.csv'
        Filename for the output subjects csv file.

    scale_beta : bool, default=False
        If True, scale the beta coefficients to have the same mean and
        variance as other values for the same metric and tract.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        nodes_file - output nodes csv file path
        subjects_file - output subjects csv file path
    """
    workdir_in = op.abspath(workdir_in)
    fn_nodes_in = op.join(workdir_in, fn_nodes_in)
    fn_subjects_in = op.join(workdir_in, fn_subjects_in)

    workdir_out = op.abspath(workdir_out)
    fn_nodes_out = op.join(workdir_out, fn_nodes_out)
    fn_subjects_out = op.join(workdir_out, fn_subjects_out)

    if op.samefile(workdir_in, workdir_out):
        raise ValueError(
            "output directory equals input directory, please "
            "output to a different directory to avoid "
            "overwriting your files."
        )

    df_nodes = pd.read_csv(fn_nodes_in)

    df_beta = pd.DataFrame(columns=columns)
    df_beta.loc["value"] = beta_hat
    df_beta = df_beta.transpose()
    df_beta = df_beta.unstack(level="metric")
    df_beta.columns = [t[-1] for t in df_beta.columns]
    df_beta.reset_index(inplace=True)
    df_beta["subjectID"] = "beta_hat"
    df_beta = df_beta[df_nodes.columns]

    if scale_beta:
        # For each tract-metric scale the beta_hat values to have the
        # same mean and standard deviation as the subjects' tract-metric
        # values. Suppose we have beta values given by x_i with mean `b_mean`
        # and standard deviation `b_std` and we want to arrive at a similar
        # set with mean `f_mean` and standard deviation `f_std`:
        #     y_i = f_mean + (x_i - b_mean) * f_std / b_std
        for tract in df_nodes["tractID"].unique():
            f_mean = (
                df_nodes.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_nodes["tractID"] == tract]
                .mean()
            )

            f_std = (
                df_nodes.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_nodes["tractID"] == tract]
                .std()
            )

            b_mean = (
                df_beta.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_beta["tractID"] == tract]
                .mean()
            )

            b_std = (
                df_beta.drop(["tractID", "subjectID", "nodeID"], axis="columns")
                .loc[df_beta["tractID"] == tract]
                .std()
            )

            metrics = b_mean.index
            df_beta.loc[df_beta["tractID"] == tract, metrics] = f_mean + (
                df_beta.loc[df_beta["tractID"] == tract, metrics] - b_mean
            ) * f_std.divide(b_std).replace([np.inf, -np.inf], 1)

    df_nodes = pd.concat([df_nodes, df_beta], axis="rows", ignore_index=True)
    df_nodes.to_csv(fn_nodes_out, index=False)

    df_subjects = pd.read_csv(fn_subjects_in, index_col=0)
    subject_row = {key: "" for key in df_subjects.columns}
    subject_row["subjectID"] = "beta_hat"
    df_subjects.loc[len(df_subjects)] = subject_row
    df_subjects.to_csv(fn_subjects_out, index=True)

    fn_streamlines_in = op.join(workdir_in, "streamlines.json")
    fn_streamlines_out = op.join(workdir_out, "streamlines.json")
    copyfile(fn_streamlines_in, fn_streamlines_out)

    fn_params_in = op.join(workdir_in, "params.json")
    fn_params_out = op.join(workdir_out, "params.json")
    copyfile(fn_params_in, fn_params_out)

    OutputFiles = namedtuple("OutputFiles", "nodes_file subjects_file")

    return OutputFiles(nodes_file=fn_nodes_out, subjects_file=fn_subjects_out)
