"""Generate samples of synthetic data sets or extract AFQ data."""
import hashlib
import numpy as np
import os
import os.path as op
import pandas as pd
import requests

from collections import namedtuple
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWire
from glob import glob
from groupyr.transform import GroupAggregator
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from .transform import AFQDataFrameMapper

torch_msg = (
    "To use AFQ-Insight's pytorch classes, you will need to have pytorch "
    "installed. You can do this by installing afqinsight with `pip install "
    "afqinsight[torch]`, or by separately installing these packages with "
    "`pip install torch`."
)
torch, HAS_TORCH, _ = optional_package("torch", torch_msg)

tf_msg = (
    "To use AFQ-Insight's tensorflow classes, you will need to have tensorflow "
    "installed. You can do this by installing afqinsight with `pip install "
    "afqinsight[tensorflow]`, or by separately installing these packages with "
    "`pip install tensorflow`."
)
tf, _, _ = optional_package("tensorflow", tf_msg)

__all__ = ["AFQDataset", "load_afq_data", "bundles2channels"]
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


def standardize_subject_id(sub_id):
    """Standardize subject ID to start with the prefix 'sub-'.

    Parameters
    ----------
    sub_id : str
        subject ID.

    Returns
    -------
    str
        Standardized subject IDs.
    """
    return sub_id if str(sub_id).startswith("sub-") else "sub-" + str(sub_id)


def load_afq_data(
    fn_nodes="nodes.csv",
    fn_subjects="subjects.csv",
    dwi_metrics=None,
    target_cols=None,
    label_encode_cols=None,
    index_col="subjectID",
    unsupervised=False,
    concat_subject_session=False,
    return_bundle_means=False,
    enforce_sub_prefix=True,
):
    """Load AFQ data from CSV, transform it, return feature matrix and target.

    This function expects a diffusion metric csv file (specified by
    ``fn_nodes``) and, optionally, a phenotypic data file (specified by
    ``fn_subjects``). The nodes csv file must be a long format dataframe with
    the following columns: "subjectID," "nodeID," "tractID," an optional
    "sessionID". All other columns are assumed to be diffusion metric columns,
    which can be optionally subset using the ``dwi_metrics`` parameter.

    For supervised learning problems (with parameter ``unsupervised=False``)
    this function will also load phenotypic targets from a subjects csv/tsv
    file. This function will load the subject data, drop subjects that are
    not found in the dwi feature matrix, and optionally label encode
    categorical values.

    Parameters
    ----------
    fn_nodes : str, default='nodes.csv'
        Filename for the nodes csv file.

    fn_subjects : str, default='subjects.csv'
        Filename for the subjects csv file.

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

    unsupervised : bool, default=False
        If True, do not load target data from the ``fn_subjects`` file.

    concat_subject_session : bool, default=False
        If True, create new subject IDs by concatenating the existing subject
        IDs with the session IDs. This is useful when subjects have multiple
        sessions and you with to disambiguate between them.

    return_bundle_means : bool, default=False
        If True, return diffusion metrics averaged along the length of each
        bundle.

    enforce_sub_prefix : bool, default=True
        If True, standardize subject IDs to start with the prefix "sub-".
        This is useful, for example, if the subject IDs in the nodes.csv file
        have the sub prefix but the subject IDs in the subjects.csv file do
        not. Default is True in order to comform to the BIDS standard.

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
    nodes = pd.read_csv(
        fn_nodes, converters={"subjectID": str, "nodeID": int, "tractID": str}
    )
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
    subjects = [
        standardize_subject_id(sub_id) if enforce_sub_prefix else sub_id
        for sub_id in mapper.subjects_
    ]
    groups = mapper.groups_
    feature_names = mapper.feature_names_

    if return_bundle_means:
        group_names = feature_names
    else:
        group_names = [tup[0:2] for tup in feature_names]
        # Now remove duplicates from group_names while preserving order
        group_names = list(dict.fromkeys(group_names))

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
            fn_subjects,
            sep=None,
            engine="python",
            index_col=index_col,
            converters={index_col: str},
        )

        # Drop unnamed columns
        unnamed_cols = [col for col in targets.columns if "Unnamed:" in col]
        targets.drop(unnamed_cols, axis="columns", inplace=True)

        if enforce_sub_prefix:
            targets.index = targets.index.map(standardize_subject_id)

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
                y.loc[:, col] = le.fit_transform(y[col].fillna("NaN"))
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


if HAS_TORCH:

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
                return self.X[idx].float()
            else:
                return self.X[idx].float(), self.y[idx].float()

else:  # pragma: no cover
    AFQTorchDataset = TripWire(torch_msg)


class AFQDataset:
    """Represent AFQ features and targets.

    The `AFQDataset` class represents tractometry features and, optionally,
    phenotypic targets.

    The simplest way to create a new AFQDataset is to pass in the tractometric
    features and phenotypic targets explicitly.

    >>> import numpy as np
    >>> AFQDataset(X=np.random.rand(50, 1000), y=np.random.rand(50))
    AFQDataset(n_samples=50, n_features=1000, n_targets=1)

    You can keep track of the names of the target variables with the `target_cols` parameter.
    >>> AFQDataset(X=np.random.rand(50, 1000), y=np.random.rand(50), target_cols=["age"])
    AFQDataset(n_samples=50, n_features=1000, n_targets=1, targets=['age'])

    Source Datasets:

    The most common way to create an AFQDataset is to load data from a set of
    csv files that conform to the AFQ data format. For example,

    >>> import os.path as op
    >>> sarica_dir = download_sarica(verbose=False)
    >>> dataset = AFQDataset.from_files(
    ...     fn_nodes=op.join(sarica_dir, "nodes.csv"),
    ...     fn_subjects=op.join(sarica_dir, "subjects.csv"),
    ...     dwi_metrics=["md", "fa"],
    ...     target_cols=["class"],
    ...     label_encode_cols=["class"],
    ... )
    >>> dataset
    AFQDataset(n_samples=48, n_features=4000, n_targets=1, targets=['class'])

    AFQDatasets are indexable and can be sliced.

    >>> dataset[0:10]
    AFQDataset(n_samples=10, n_features=4000, n_targets=1, targets=['class'])

    You can query the length of the dataset as well as the feature and target
    shapes.

    >>> len(dataset)
    48
    >>> dataset.shape
    ((48, 4000), (48,))

    Datasets can be used as expected in scikit-learn's model selection
    functions. For example

    >>> from sklearn.model_selection import train_test_split
    >>> train_data, test_data = train_test_split(dataset, test_size=0.3, stratify=dataset.y)
    >>> train_data
    AFQDataset(n_samples=33, n_features=4000, n_targets=1, targets=['class'])
    >>> test_data
    AFQDataset(n_samples=15, n_features=4000, n_targets=1, targets=['class'])

    You can drop samples from the dataset that have null target values using the
    `drop_target_na` method.
    >>> dataset.y = dataset.y.astype(float)
    >>> dataset.y[:5] = np.nan
    >>> dataset.drop_target_na()
    >>> dataset
    AFQDataset(n_samples=43, n_features=4000, n_targets=1, targets=['class'])

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The feature samples.

    y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
        Target values. This will be None if ``unsupervised`` is True

    groups : list of numpy.ndarray, optional
        The feature indices for each feature group. These are typically used
        to keep group collections of "nodes" into white matter bundles.

    feature_names : list of tuples, optional
        The multi-indexed columns of X. i.e. the names of the features.

    group_names : list of tuples, optional
        The multi-indexed groups of X. i.e. the names of the feature groups.

    target_cols : list of strings, optional
        List of column names for the target variables in `y`.

    subjects : list, optional
        Subject IDs

    sessions : list, optional
        Session IDs.

    classes : dict, optional
        Class labels for each label encoded column specified in ``y``.
    """

    def __init__(
        self,
        X,
        y=None,
        groups=None,
        feature_names=None,
        group_names=None,
        target_cols=None,
        subjects=None,
        sessions=None,
        classes=None,
    ):
        self.X = X
        self.y = y
        self.groups = groups
        self.feature_names = feature_names
        self.group_names = group_names
        self.target_cols = target_cols
        if subjects is None:
            subjects = [f"sub-{i}" for i in range(len(X))]
        self.subjects = subjects
        self.sessions = sessions
        self.classes = classes

    @staticmethod
    def from_files(
        fn_nodes="nodes.csv",
        fn_subjects="subjects.csv",
        dwi_metrics=None,
        target_cols=None,
        label_encode_cols=None,
        index_col="subjectID",
        unsupervised=False,
        concat_subject_session=False,
        enforce_sub_prefix=True,
    ):
        """Create an `AFQDataset` from csv files.

        This method expects a diffusion metric csv file (specified by ``fn_nodes``)
        and, optionally, a phenotypic data file (specified by ``fn_subjects``). The
        nodes csv file must be a long format dataframe with the following columns:
        "subjectID," "nodeID," "tractID," an optional "sessionID". All other columns
        are assumed to be diffusion metric columns, which can be optionally subset
        using the ``dwi_metrics`` parameter.

        For supervised learning problems (with parameter ``unsupervised=False``)
        this function will also load phenotypic targets from a subjects csv/tsv
        file. This function will load the subject data, drop subjects that are
        not found in the dwi feature matrix, and optionally label encode
        categorical values.

        Parameters
        ----------
        fn_nodes : str, default='nodes.csv'
            Filename for the nodes csv file.

        fn_subjects : str, default='subjects.csv'
            Filename for the subjects csv file.

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

        unsupervised : bool, default=False
            If True, do not load target data from the ``fn_subjects`` file.

        concat_subject_session : bool, default=False
            If True, create new subject IDs by concatenating the existing subject
            IDs with the session IDs. This is useful when subjects have multiple
            sessions and you with to disambiguate between them.

        enforce_sub_prefix : bool, default=True
            If True, standardize subject IDs to start with the prefix "sub-".
            This is useful, for example, if the subject IDs in the nodes.csv file
            have the sub prefix but the subject IDs in the subjects.csv file do
            not. Default is True in order to comform to the BIDS standard.

        Returns
        -------
        AFQDataset

        See Also
        --------
        transform.AFQDataFrameMapper
        """
        afq_data = load_afq_data(
            fn_nodes=fn_nodes,
            fn_subjects=fn_subjects,
            dwi_metrics=dwi_metrics,
            target_cols=target_cols,
            label_encode_cols=label_encode_cols,
            index_col=index_col,
            unsupervised=unsupervised,
            concat_subject_session=concat_subject_session,
            enforce_sub_prefix=enforce_sub_prefix,
        )

        return AFQDataset(
            X=afq_data.X,
            y=afq_data.y,
            groups=afq_data.groups,
            feature_names=afq_data.feature_names,
            target_cols=target_cols,
            group_names=afq_data.group_names,
            subjects=afq_data.subjects,
            sessions=afq_data.sessions,
            classes=afq_data.classes,
        )

    @staticmethod
    def from_study(study, verbose=None):
        """Fetch an AFQ dataset from a predefined study.

        This method expects downloads a pre-existing AFQ dataset.
        ``study="sarica"` will download the ALS classification dataset used in
        Sarica et al. [1]_ ``study="weston-havens"`` will download the lifespan
        maturation dataset described in Yeatman et al. [2]_. ``study="hbn"``
        will download the pediatric Healthy Brain Network dataset [3]_.

        Parameters
        ----------
        study : ["sarica", "weston-havens", "hbn"]
            Name of the study to download.

        verbose : bool, optional
            If True, print download status messages to stdout. The default of
            None will default to the datasets default verbosity.

        Returns
        -------
        AFQDataset

        See Also
        --------
        download_sarica : downloader for the Sarica dataset
        download_weston_havens : downloader for Weston-Havens dataset
        download_hbn : downloader for the HBN dataset

        References
        ----------
        .. [1]  Alessia Sarica, et al.
            "The Corticospinal Tract Profile in AmyotrophicLateral Sclerosis"
            Human Brain Mapping, vol. 38, pp. 727-739, 2017
            DOI: 10.1002/hbm.23412

        .. [2]  Jason D. Yeatman, Brian A. Wandell, & Aviv A. Mezer,
            "Lifespan maturation and degeneration of human brain white matter"
            Nature Communications, vol. 5:1, pp. 4932, 2014
            DOI: 10.1038/ncomms5932

        .. [3]  Adam Richie-Halford, Matthew Cieslak, Lei Ai, Sendy Caffarra, Sydney
           Covitz, Alexandre R. Franco, Iliana I. Karipidis, John Kruper, Michael
           Milham, Bárbara Avelar-Pereira, Ethan Roy, Valerie J. Sydnor, Jason
           Yeatman, The Fibr Community Science Consortium, Theodore D.
           Satterthwaite, and Ariel Rokem,
           "An open, analysis-ready, and quality controlled resource for pediatric
           brain white-matter research"
           bioRxiv 2022.02.24.481303;
           doi: https://doi.org/10.1101/2022.02.24.481303
        """
        downloaders = {
            "sarica": download_sarica,
            "weston-havens": download_weston_havens,
            "hbn": download_hbn,
        }

        if study.lower() not in downloaders.keys():
            raise ValueError(
                f"study must be one of {downloaders.keys()}. Got {study} instead."
            )

        download_kwargs = dict()
        if verbose is not None:
            download_kwargs["verbose"] = verbose

        data_dir = downloaders[study.lower()](**download_kwargs)
        fn_subs = glob(op.join(data_dir, "subjects.?sv"))[0]
        dataset_kwargs = {
            "sarica": dict(
                dwi_metrics=["md", "fa"],
                target_cols=["class"],
                label_encode_cols=["class"],
            ),
            "weston-havens": dict(dwi_metrics=["md", "fa"], target_cols=["Age"]),
            "hbn": dict(
                dwi_metrics=["dki_md", "dki_fa"],
                target_cols=["age", "sex", "scan_site_id"],
                label_encode_cols=["sex", "scan_site_id"],
                index_col="subject_id",
            ),
        }

        return AFQDataset.from_files(
            fn_nodes=op.join(data_dir, "nodes.csv"),
            fn_subjects=fn_subs,
            **dataset_kwargs[study],
        )

    def __repr__(self):
        """Return a string representation of the dataset."""
        n_samples, n_features = self.X.shape
        repr_params = [f"n_samples={n_samples}", f"n_features={n_features}"]

        if self.y is not None:
            try:
                n_targets = self.y.shape[1]
            except IndexError:
                n_targets = 1
            repr_params += [f"n_targets={n_targets}"]
        if self.target_cols is not None:
            repr_params += [f"targets={self.target_cols}"]

        return f"AFQDataset({', '.join(repr_params)})"

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, indices):
        """Return a subset of the dataset as a new AFQDataset."""
        return AFQDataset(
            X=self.X[indices],
            y=self.y[indices] if self.y is not None else None,
            groups=self.groups,
            feature_names=self.feature_names,
            target_cols=self.target_cols,
            group_names=self.group_names,
            subjects=np.array(self.subjects)[indices].tolist(),
            sessions=np.array(self.sessions)[indices].tolist()
            if self.sessions is not None
            else None,
            classes=self.classes,
        )

    @property
    def shape(self):
        """Return the shape of the features and targets.

        Returns
        -------
        tuple
            ((n_samples, n_features), (n_samples, n_targets))
            if y is not None, otherwise (n_samples, n_features)
        """
        if self.y is not None:
            return self.X.shape, self.y.shape
        else:
            return self.X.shape

    def copy(self):
        """Return a deep copy of this dataset.

        Returns
        -------
        AFQDataset
            A deep copy of this dataset
        """
        return AFQDataset(
            X=self.X,
            y=self.y,
            groups=self.groups,
            feature_names=self.feature_names,
            target_cols=self.target_cols,
            group_names=self.group_names,
            subjects=self.subjects,
            sessions=self.sessions,
            classes=self.classes,
        )

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
        if self.y is not None:
            nan_mask = np.isnan(self.y)
            if len(self.y.shape) > 1:
                nan_mask = nan_mask.astype(int).sum(axis=1).astype(bool)

            nan_mask = ~nan_mask

            # This nan_mask contains booleans for float NaN values
            # But we also potentially label encoded NaNs above so we need to
            # check for the string "NaN" in the encoded labels
            if self.classes is not None:
                nan_encoding = {
                    label: "NaN" in vals for label, vals in self.classes.items()
                }
                for label, nan_encoded in nan_encoding.items():
                    if nan_encoded:
                        encoded_value = np.where(self.classes[label] == "NaN")[0][0]
                        encoded_col = self.target_cols.index(label)
                        if len(self.y.shape) > 1:
                            nan_mask = np.logical_and(
                                nan_mask, self.y[:, encoded_col] != encoded_value
                            )
                        else:
                            nan_mask = np.logical_and(nan_mask, self.y != encoded_value)

            self.X = self.X[nan_mask]
            self.y = self.y[nan_mask]
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
        """Return features and labels packaged as a tensorflow dataset.

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
        tensorflow.data.Dataset.from_tensor_slices
            The tensorflow dataset
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

        if self.y is None:
            return tf.data.Dataset.from_tensor_slices(X)
        else:
            return tf.data.Dataset.from_tensor_slices((X, self.y.astype(float)))

    def model_fit(self, model, **fit_params):
        """Fit the dataset with a provided model object.

        Parameters
        ----------
        model : sklearn model
            The estimator or transformer to fit

        **fit_params : dict
            Additional parameters to pass to the fit method

        Returns
        -------
        model : object
            The fitted model
        """
        return model.fit(X=self.X, y=self.y, **fit_params)

    def model_fit_transform(self, model, **fit_params):
        """Fit and transform the dataset with a provided model object.

        Parameters
        ----------
        model : sklearn model
            The estimator or transformer to fit

        **fit_params : dict
            Additional parameters to pass to the fit_transform method

        Returns
        -------
        dataset_new : AFQDataset
            New AFQDataset with transformed features
        """
        return AFQDataset(
            X=model.fit_transform(X=self.X, y=self.y, **fit_params),
            y=self.y,
            groups=self.groups,
            feature_names=self.feature_names,
            target_cols=self.target_cols,
            group_names=self.group_names,
            subjects=self.subjects,
            sessions=self.sessions,
            classes=self.classes,
        )

    def model_transform(self, model, **transform_params):
        """Transform the dataset with a provided model object.

        Parameters
        ----------
        model : sklearn model
            The estimator or transformer to use to transform the features

        **transform_params : dict
            Additional parameters to pass to the transform method

        Returns
        -------
        dataset_new : AFQDataset
            New AFQDataset with transformed features
        """
        return AFQDataset(
            X=model.transform(X=self.X, **transform_params),
            y=self.y,
            groups=self.groups,
            feature_names=self.feature_names,
            target_cols=self.target_cols,
            group_names=self.group_names,
            subjects=self.subjects,
            sessions=self.sessions,
            classes=self.classes,
        )

    def model_predict(self, model, **predict_params):
        """Predict the targets with a provided model object.

        Parameters
        ----------
        model : sklearn model
            The estimator or transformer to use to predict the targets

        **predict_params : dict
            Additional parameters to pass to the predict method

        Returns
        -------
        y_pred : ndarray
            Predicted targets
        """
        return model.predict(X=self.X, **predict_params)

    def model_score(self, model, **score_params):
        """Score a model on this dataset.

        Parameters
        ----------
        model : sklearn model
            The estimator or transformer to use to score the model

        **score_params : dict
            Additional parameters to pass to the `score` method, e.g., `sample_weight`

        Returns
        -------
        score : float
            The score of the model (e.g. R2, accuracy, etc.)
        """
        return model.score(X=self.X, y=self.y, **score_params)


def _download_url_to_file(url, output_fn, verbose=True):
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
        if verbose:
            print(f"File {fn_abs} exists.")
    else:
        # Download from url and save to file
        with requests.Session() as s:
            response = s.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                desc=op.basename(output_fn),
                disable=not verbose,
            )
            with open(fn_abs, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.total != total_size_in_bytes:
                print("ERROR, something went wrong")

        # Write MD5 checksum to file
        with open(base + ".md5", "w") as md5file:
            md5file.write(hashlib.md5(open(fn_abs, "rb").read()).hexdigest())


def _download_afq_dataset(dataset, data_home, verbose=True):
    urls_files = {
        "sarica": [
            {
                "url": "https://github.com/yeatmanlab/Sarica_2017/raw/gh-pages/data/subjects.csv",
                "file": op.join(data_home, "sarica", "subjects.csv"),
            },
            {
                "url": "https://github.com/yeatmanlab/Sarica_2017/raw/gh-pages/data/nodes.csv",
                "file": op.join(data_home, "sarica", "nodes.csv"),
            },
        ],
        "weston_havens": [
            {
                "url": "https://yeatmanlab.github.io/AFQBrowser-demo/data/subjects.csv",
                "file": op.join(data_home, "weston_havens", "subjects.csv"),
            },
            {
                "url": "https://yeatmanlab.github.io/AFQBrowser-demo/data/nodes.csv",
                "file": op.join(data_home, "weston_havens", "nodes.csv"),
            },
        ],
        "hbn": [
            {
                "url": "http://s3.amazonaws.com/fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
                "file": op.join(data_home, "hbn", "subjects.tsv"),
            },
            {
                "url": "http://s3.amazonaws.com/fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/afq/combined_tract_profiles.csv",
                "file": op.join(data_home, "hbn", "nodes.csv"),
            },
        ],
    }

    for dict_ in urls_files[dataset]:
        _download_url_to_file(dict_["url"], dict_["file"], verbose=verbose)

    return op.commonpath(
        [download_dict["file"] for download_dict in urls_files[dataset]]
    )


def download_sarica(data_home=None, verbose=True):
    """Fetch the ALS classification dataset from Sarica et al [1]_.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default all
        afq-insight data is stored in '~/.afq-insight' subfolders.

    verbose : bool, default=True
        If True, print status messages to stdout.

    Returns
    -------
    dirname : str
        Path to the downloaded dataset directory.

    References
    ----------
    .. [1]  Alessia Sarica, et al.
        "The Corticospinal Tract Profile in AmyotrophicLateral Sclerosis"
        Human Brain Mapping, vol. 38, pp. 727-739, 2017
        DOI: 10.1002/hbm.23412
    """
    data_home = data_home if data_home is not None else _DATA_DIR
    return _download_afq_dataset("sarica", data_home=data_home, verbose=verbose)


def download_weston_havens(data_home=None, verbose=True):
    """Load the age prediction dataset from Weston-Havens [1]_.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default all
        afq-insight data is stored in '~/.afq-insight' subfolders.

    Returns
    -------
    dirname : str
        Path to the downloaded dataset directory.

    verbose : bool, default=True
        If True, print status messages to stdout.

    References
    ----------
    .. [1]  Jason D. Yeatman, Brian A. Wandell, & Aviv A. Mezer,
        "Lifespan maturation and degeneration of human brain white matter"
        Nature Communications, vol. 5:1, pp. 4932, 2014
        DOI: 10.1038/ncomms5932
    """
    data_home = data_home if data_home is not None else _DATA_DIR
    return _download_afq_dataset("weston_havens", data_home=data_home, verbose=verbose)


def download_hbn(data_home=None, verbose=True):
    """Load the Healthy Brain Network Preprocessed Diffusion Derivatives [1]_.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default all
        afq-insight data is stored in '~/.afq-insight' subfolders.

    Returns
    -------
    dirname : str
        Path to the downloaded dataset directory.

    verbose : bool, default=True
        If True, print status messages to stdout.

    References
    ----------
    .. [1]  Adam Richie-Halford, Matthew Cieslak, Lei Ai, Sendy Caffarra, Sydney
       Covitz, Alexandre R. Franco, Iliana I. Karipidis, John Kruper, Michael
       Milham, Bárbara Avelar-Pereira, Ethan Roy, Valerie J. Sydnor, Jason
       Yeatman, The Fibr Community Science Consortium, Theodore D.
       Satterthwaite, and Ariel Rokem,
       "An open, analysis-ready, and quality controlled resource for pediatric
       brain white-matter research"
       bioRxiv 2022.02.24.481303;
       doi: https://doi.org/10.1101/2022.02.24.481303
    """
    data_home = data_home if data_home is not None else _DATA_DIR
    return _download_afq_dataset("hbn", data_home=data_home, verbose=verbose)
