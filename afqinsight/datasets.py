"""Generate samples of synthetic data sets or extract AFQ data."""
import hashlib
import numpy as np
import os
import os.path as op
import pandas as pd
import requests

from collections import namedtuple
from groupyr.transform import GroupAggregator
from shutil import copyfile
from sklearn.preprocessing import LabelEncoder

try:
    import torch

    HAS_PYTORCH = True

    class AFQTorchDataset(torch.utils.data.Dataset):
        def __init__(self, X, y=None):
            """AFQ features and targets packages as a pytorch dataset.

            Parameters
            ----------
            X : np.ndarray
                The feature samples.

            y : np.ndarray, optional
                Target values.

            Attributes
            ----------
            X : np.ndarray
                The feature samples converted to torch.tensor

            y : np.ndarray
                Target values converted to torch tensor

            unsupervised : bool
                True if ``y`` was provided on init. False otherwise
            """
            _check_pytorch()

            self.X = torch.tensor(X)
            if y is None:
                self.unsupervised = True
                self.y = torch.tensor([])
            else:
                self.unsupervised = False
                self.y = torch.tensor(y.astype(float))

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            if self.unsupervised:
                return self.X[idx]
            else:
                return self.X[idx], self.y[idx]


except ImportError:  # pragma: no cover
    HAS_PYTORCH = False

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:  # pragma: no cover
    HAS_TF = False

from .transform import AFQDataFrameMapper

__all__ = ["load_afq_data", "output_beta_to_afq"]
_DATA_DIR = op.join(op.expanduser("~"), ".cache", "afq-insight")
_FIELDS = [
    "X",
    "y",
    "groups",
    "feature_names",
    "group_names",
    "subjects",
    "sessions",
    "classes",
]
try:
    AFQData = namedtuple("AFQData", _FIELDS, defaults=(None,) * 9)
except TypeError:
    AFQData = namedtuple("AFQData", _FIELDS)
    AFQData.__new__.__defaults__ = (None,) * len(AFQData._fields)


def _check_pytorch():
    if not HAS_PYTORCH:  # pragma: no cover
        raise ImportError(
            "To use AFQ-Insight's pytorch classes, you will need to have pytorch "
            "installed. You can do this by installing afqinsight with `pip install "
            "afqinsight[torch]`, or by separately installing these packages with "
            "`pip install torch`."
        )


def _check_tf():
    if not HAS_TF:  # pragma: no cover
        raise ImportError(
            "To use AFQ-Insight's tensorflow classes, you will need to have tensorflow "
            "installed. You can do this by installing afqinsight with `pip install "
            "afqinsight[tensorflow]`, or by separately installing these packages with "
            "`pip install tensorflow`."
        )


def bundles2channels(X, n_nodes, n_channels, channels_last=True):
    """Reshape AFQ feature matrix with bundles as channels.

    This function takes an input feature matrix of shape (n_samples,
    n_features), where n_features is n_nodes * n_bundles * n_metrics.  If
    ``channels_last=True``, it returns a reshaped feature matrix with shape
    (n_samples, n_nodes, n_channels), where n_channels = n_bundles * n_metrics.
    If ``channels_last=False``, the returned shape is (n_samples, n_channels,
    n_nodes).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The input feature matrix.

    n_nodes : int
        The number of nodes per bundle.

    n_channels : int
        The number of desired output channels.

    channels_last : bool, default=True
        If True, the output will have shape (n_samples, n_nodes, n_channels).
        Otherwise, the output will have shape (n_samples, n_channels, n_nodes).

    Returns
    -------
    np.ndarray
        The reshaped feature matrix
    """
    if n_nodes * n_channels != X.shape[1]:
        raise ValueError(
            "The product n_nodes and n_channels does not match the number of "
            f"features in X. Got n_nodes={n_nodes}, n_channels={n_channels}, "
            f"n_features_total={X.shape[1]}."
        )

    output = X.reshape((X.shape[0], n_channels, n_nodes))
    if channels_last:
        output = np.swapaxes(output, 1, 2)

    return output


def load_afq_data(
    workdir,
    dwi_metrics=None,
    target_cols=None,
    label_encode_cols=None,
    index_col="subjectID",
    fn_nodes="nodes.csv",
    fn_subjects="subjects.csv",
    unsupervised=False,
    concat_subject_session=False,
    return_bundle_means=False,
):
    """Load AFQ data from CSV, transform it, return feature matrix and target.

    This function expects a directory with a diffusion metric csv file
    (specified by ``fn_nodes``) and, optionally, a phenotypic data file
    (specified by ``fn_subjects``). The nodes csv file must be a long format
    dataframe with the following columns: "subjectID," "nodeID," "tractID,"
    an optional "sessionID". All other columns are assumed to be diffusion
    metric columns, which can be optionally subset using the ``dwi_metrics``
    parameter.

    For supervised learning problems (with parameter ``unsupervised=False``)
    this function will also load phenotypic targets from a subjects csv/tsv
    file. This function will load the subject data, drop subjects that are
    not found in the dwi feature matrix, and optionally label encode
    categorical values.

    Parameters
    ----------
    workdir : str
        Directory in which to find the AFQ csv files

    dwi_metrics : list of strings, optional
        List of diffusion metrics to extract from nodes csv.
        e.g. ["dki_md", "dki_fa"]

    target_cols : list of strings, optional
        List of column names in subjects csv file to use as target variables

    label_encode_cols : list of strings, subset of target_cols
        Must be a subset of target_cols. These columns will be encoded using
        :class:`sklearn:sklearn.preprocessing.LabelEncoder`.

    index_col : str, default='subjectID'
        The name of column in the subject csv file to use as the index. This
        should contain subject IDs.

    fn_nodes : str, default='nodes.csv'
        Filename for the nodes csv file.

    fn_subjects : str, default='subjects.csv'
        Filename for the subjects csv file.

    unsupervised : bool, default=False
        If True, do not load target data from the ``fn_subjects`` file.

    concat_subject_session : bool, default=False
        If True, create new subject IDs by concatenating the existing subject
        IDs with the session IDs. This is useful when subjects have multiple
        sessions and you with to disambiguate between them.

    return_bundle_means : bool, default=False
        If True, return diffusion metrics averaged along the length of each
        bundle.

    Returns
    -------
    AFQData : namedtuple
        A namedtuple with the fields:

        X : array-like of shape (n_samples, n_features)
            The feature samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target values. This will be None if ``unsupervised`` is True

        groups : list of numpy.ndarray
            feature indices for each feature group

        feature_names : list of tuples
            The multi-indexed columns of X

        group_names : list of tuples
            The multi-indexed groups of X

        subjects : list
            Subject IDs

        sessions : list
            Session IDs.

        classes : dict
            Class labels for each column specified in ``label_encode_cols``.
            This will be None if ``unsupervised`` is True

    See Also
    --------
    transform.AFQDataFrameMapper
    """
    workdir = op.abspath(workdir)
    fn_nodes = op.join(workdir, fn_nodes)
    fn_subjects = op.join(workdir, fn_subjects)

    nodes = pd.read_csv(fn_nodes)
    unnamed_cols = [col for col in nodes.columns if "Unnamed:" in col]
    nodes.drop(unnamed_cols, axis="columns", inplace=True)

    sessions = nodes["sessionID"] if "sessionID" in nodes.columns else None
    if concat_subject_session and "sessionID" not in nodes.columns:
        raise ValueError(
            "Cannot concatenate subjectID and sessionID because 'sessionID' "
            f"is not one of the columns in {fn_nodes}. Either choose a csv "
            "file with 'sessionID' as one of the columns or set "
            "concat_subject_session=False."
        )

    if dwi_metrics is not None:
        non_metric_cols = ["tractID", "nodeID", "subjectID"]
        if "sessionID" in nodes.columns:
            non_metric_cols += ["sessionID"]

        nodes = nodes[non_metric_cols + dwi_metrics]

    if return_bundle_means:
        mapper = AFQDataFrameMapper(
            bundle_agg_func="mean", concat_subject_session=concat_subject_session
        )
    else:
        mapper = AFQDataFrameMapper(concat_subject_session=concat_subject_session)

    X = mapper.fit_transform(nodes)
    subjects = mapper.subjects_
    groups = mapper.groups_
    feature_names = mapper.feature_names_

    if return_bundle_means:
        group_names = feature_names
    else:
        group_names = [tup[0:2] for tup in feature_names if tup[2] == 0]

    if unsupervised:
        y = None
        classes = None
    else:
        if target_cols is None:
            raise ValueError(
                "If you are loading data for a supervised "
                "learning problem you must specify the "
                "`target_cols` input. If you intended to "
                "load data for an unsupervised learning "
                "problem, please set `unsupervised=True`."
            )

        # Read using sep=None, engine="python" to allow for both csv and tsv
        targets = pd.read_csv(
            fn_subjects, sep=None, engine="python", index_col=index_col
        )

        # Drop unnamed columns
        unnamed_cols = [col for col in targets.columns if "Unnamed:" in col]
        targets.drop(unnamed_cols, axis="columns", inplace=True)

        # Drop subjects that are not in the dwi feature matrix
        targets = pd.DataFrame(index=subjects).merge(
            targets, how="left", left_index=True, right_index=True
        )

        # Select user defined target columns
        y = targets.loc[:, target_cols]

        # Label encode the user-supplied categorical columns
        if label_encode_cols is not None:
            classes = {}
            if not set(label_encode_cols) <= set(target_cols):
                raise ValueError(
                    "label_encode_cols must be a subset of target_cols; "
                    "got {0} instead.".format(label_encode_cols)
                )

            le = LabelEncoder()
            for col in label_encode_cols:
                y.loc[:, col] = le.fit_transform(y[col])
                classes[col] = le.classes_
        else:
            classes = None

        y = np.squeeze(y.to_numpy())

    return AFQData(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        group_names=group_names,
        subjects=subjects,
        sessions=sessions,
        classes=classes,
    )


class AFQDataset:
    def __init__(
        self,
        workdir,
        dwi_metrics=None,
        target_cols=None,
        label_encode_cols=None,
        index_col="subjectID",
        fn_nodes="nodes.csv",
        fn_subjects="subjects.csv",
        unsupervised=False,
        concat_subject_session=False,
    ):
        """Load AFQ data from CSV.

        This class expects a directory with a diffusion metric csv file
        (specified by ``fn_nodes``) and, optionally, a phenotypic data file
        (specified by ``fn_subjects``). The nodes csv file must be a long format
        dataframe with the following columns: "subjectID," "nodeID," "tractID,"
        an optional "sessionID". All other columns are assumed to be diffusion
        metric columns, which can be optionally subset using the ``dwi_metrics``
        parameter.

        For supervised learning problems (with parameter ``unsupervised=False``)
        this function will also load phenotypic targets from a subjects csv/tsv
        file. This function will load the subject data, drop subjects that are
        not found in the dwi feature matrix, and optionally label encode
        categorical values.

        Parameters
        ----------
        workdir : str
            Directory in which to find the AFQ csv files

        dwi_metrics : list of strings, optional
            List of diffusion metrics to extract from nodes csv.
            e.g. ["dki_md", "dki_fa"]

        target_cols : list of strings, optional
            List of column names in subjects csv file to use as target variables

        label_encode_cols : list of strings, subset of target_cols
            Must be a subset of target_cols. These columns will be encoded using
            :class:`sklearn:sklearn.preprocessing.LabelEncoder`.

        index_col : str, default='subjectID'
            The name of column in the subject csv file to use as the index. This
            should contain subject IDs.

        fn_nodes : str, default='nodes.csv'
            Filename for the nodes csv file.

        fn_subjects : str, default='subjects.csv'
            Filename for the subjects csv file.

        unsupervised : bool, default=False
            If True, do not load target data from the ``fn_subjects`` file.

        concat_subject_session : bool, default=False
            If True, create new subject IDs by concatenating the existing subject
            IDs with the session IDs. This is useful when subjects have multiple
            sessions and you with to disambiguate between them.

        Attributes
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target values. This will be None if ``unsupervised`` is True

        groups : list of numpy.ndarray
            feature indices for each feature group

        feature_names : list of tuples
            The multi-indexed columns of X

        group_names : list of tuples
            The multi-indexed groups of X

        subjects : list
            Subject IDs

        sessions : list
            Session IDs.

        classes : dict
            Class labels for each column specified in ``label_encode_cols``.
            This will be None if ``unsupervised`` is True

        See Also
        --------
        transform.AFQDataFrameMapper
        """
        afq_data = load_afq_data(
            workdir=workdir,
            dwi_metrics=dwi_metrics,
            target_cols=target_cols,
            label_encode_cols=label_encode_cols,
            index_col=index_col,
            fn_nodes=fn_nodes,
            fn_subjects=fn_subjects,
            unsupervised=unsupervised,
            concat_subject_session=concat_subject_session,
        )

        self.X = afq_data.X
        self.y = afq_data.y
        if self.y is not None:
            self.y = self.y.astype(float)

        self.groups = afq_data.groups
        self.feature_names = afq_data.feature_names
        self.group_names = afq_data.group_names
        self.subjects = afq_data.subjects
        self.sessions = afq_data.sessions
        self.classes = afq_data.classes

    def bundle_means(self):
        """Return diffusion metrics averaged along the length of each bundle.

        Returns
        -------
        means : np.ndarray
            The mean diffusion metric along the length of each bundle
        """
        ga = GroupAggregator(groups=self.groups)
        return ga.fit_transform(self.X)

    def drop_target_na(self):
        """Drop subjects who have nan values as targets.
        
        This method modifies the ``X``, ``y``, and ``subjects`` attributes in-place.
        """
        nan_mask = np.isnan(self.y).astype(int).sum(axis=1).astype(bool)
        self.X = self.X[~nan_mask]
        self.y = self.y[~nan_mask]
        self.subjects = [sub for mask, sub in zip(nan_mask, self.subjects) if mask]

    def as_torch_dataset(self, bundles_as_channels=True, channels_last=False):
        """Return features and labels packaged as a pytorch dataset.

        Parameters
        ----------
        bundles_as_channels : bool, default=True
            If True, reshape the feature matrix such that each bundle/metric
            combination gets it's own channel.

        channels_last : bool, default=False
            If True, the channels will be the last dimension of the feature tensor.
            Otherwise, the channels will be the penultimate dimension.

        Returns
        -------
        AFQTorchDataset
            The pytorch dataset
        """
        if bundles_as_channels:
            n_channels = len(self.group_names)
            _, n_features = self.X.shape
            n_nodes = n_features // n_channels
            X = bundles2channels(
                self.X,
                n_nodes=n_nodes,
                n_channels=n_channels,
                channels_last=channels_last,
            )
        else:
            X = self.X

        return AFQTorchDataset(X, self.y)

    def as_tensorflow_dataset(self, bundles_as_channels=True, channels_last=True):
        _check_tf()
        if bundles_as_channels:
            n_channels = len(self.group_names)
            _, n_features = self.X.shape
            n_nodes = n_features // n_channels
            X = bundles2channels(
                self.X,
                n_nodes=n_nodes,
                n_channels=n_channels,
                channels_last=channels_last,
            )
        else:
            X = self.X

        if self.y is None:
            return tf.data.Dataset.from_tensor_slices(X)
        else:
            return tf.data.Dataset.from_tensor_slices((X, self.y.astype(float)))


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


def _download_url_to_file(url, output_fn, encoding="utf-8"):
    fn_abs = op.abspath(output_fn)
    base = op.splitext(fn_abs)[0]
    os.makedirs(op.dirname(output_fn), exist_ok=True)

    # check if file with *.md5 exists
    if op.isfile(base + ".md5"):
        with open(base + ".md5", "r") as md5file:
            md5sum = md5file.read().replace("\n", "")
    else:
        md5sum = None
    # compare MD5 hash
    if (
        op.isfile(fn_abs)
        and hashlib.md5(open(fn_abs, "rb").read()).hexdigest() == md5sum
    ):
        print(f"File {op.relpath(fn_abs)} exists.")
    else:
        print(f"Downloading {url} to {op.relpath(fn_abs)}.")
        # Download from url and save to file
        with requests.Session() as s:
            download = s.get(url)
            with open(fn_abs, "w") as fp:
                fp.write(download.content.decode(encoding))

        # Write MD5 checksum to file
        with open(base + ".md5", "w") as md5file:
            md5file.write(hashlib.md5(open(fn_abs, "rb").read()).hexdigest())


def _download_afq_dataset(dataset, data_home):
    urls_files = {
        "sarica": [
            {
                "url": "https://github.com/yeatmanlab/Sarica_2017/raw/gh-pages/data/nodes.csv",
                "file": op.join(data_home, "sarica_data", "nodes.csv"),
            },
            {
                "url": "https://github.com/yeatmanlab/Sarica_2017/raw/gh-pages/data/subjects.csv",
                "file": op.join(data_home, "sarica_data", "subjects.csv"),
            },
        ],
        "weston_havens": [
            {
                "url": "https://yeatmanlab.github.io/AFQBrowser-demo/data/nodes.csv",
                "file": op.join(data_home, "weston_havens_data", "nodes.csv"),
            },
            {
                "url": "https://yeatmanlab.github.io/AFQBrowser-demo/data/subjects.csv",
                "file": op.join(data_home, "weston_havens_data", "subjects.csv"),
            },
        ],
    }

    for dict_ in urls_files[dataset]:
        _download_url_to_file(dict_["url"], dict_["file"])


def download_sarica(data_home=None):
    """Fetch the ALS classification dataset from Sarica et al [1]_.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default all
        afq-insight data is stored in ‘~/.afq-insight’ subfolders.

    Returns
    -------
    X : array-like of shape (48, 3600)
        The feature samples.

    y : array-like of shape (48,)
        Target values.

    groups : list of numpy.ndarray
        feature indices for each feature group

    feature_names : list of tuples
        The multi-indexed columns of X

    group_names : list of tuples
        The multi-indexed groups of X

    subjects : list
        Subject IDs

    classes : dict
        Class labels for ALS diagnosis.

    References
    ----------
    .. [1]  Alessia Sarica, et al.
        "The Corticospinal Tract Profile in AmyotrophicLateral Sclerosis"
        Human Brain Mapping, vol. 38, pp. 727-739, 2017
        DOI: 10.1002/hbm.23412
    """
    data_home = data_home if data_home is not None else _DATA_DIR
    _download_afq_dataset("sarica", data_home=data_home)
    return op.join(data_home, "sarica_data")


def download_weston_havens(data_home=None):
    """Load the age prediction dataset from Weston-Havens [1]_.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default all
        afq-insight data is stored in ‘~/.afq-insight’ subfolders.

    Returns
    -------
    X : array-like of shape (77, 3600)
        The feature samples.

    y : array-like of shape (77,) or (n_samples, n_targets), optional
        Target values.

    groups : list of numpy.ndarray
        feature indices for each feature group

    feature_names : list of tuples
        The multi-indexed columns of X

    group_names : list of tuples
        The multi-indexed groups of X

    subjects : list
        Subject IDs

    References
    ----------
    .. [1]  Jason D. Yeatman, Brian A. Wandell, & Aviv A. Mezer,
        "Lifespan maturation and degeneration of human brain white matter"
        Nature Communications, vol. 5:1, pp. 4932, 2014
        DOI: 10.1038/ncomms5932
    """
    data_home = data_home if data_home is not None else _DATA_DIR
    _download_afq_dataset("weston_havens", data_home=data_home)
    return op.join(data_home, "weston_havens_data")
