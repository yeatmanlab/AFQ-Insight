"""Plot bundle profiles."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import OrderedDict
from groupyr.transform import GroupExtractor
from tqdm.auto import tqdm

from .utils import BUNDLE_MAT_2_PYTHON
from .datasets import AFQDataset


POSITIONS = OrderedDict(
    {
        "IFO_L": (0, 0),
        "IFO_R": (0, 3),
        "IFOF_L": (0, 0),
        "IFOF_R": (0, 3),
        "UNC_L": (0, 1),
        "UNC_R": (0, 2),
        "ATR_L": (1, 0),
        "ATR_R": (1, 3),
        "CST_L": (1, 1),
        "CST_R": (1, 2),
        "ARC_L": (2, 0),
        "ARC_R": (2, 3),
        "SLF_L": (2, 1),
        "SLF_R": (2, 2),
        "ILF_L": (3, 0),
        "ILF_R": (3, 3),
        "CGC_L": (3, 1),
        "CGC_R": (3, 2),
        "HCC_L": (4, 0),
        "HCC_R": (4, 3),
        "FA": (4, 1),
        "FP": (4, 2),
        "CC_ForcepsMinor": (4, 1),
        "CC_ForcepsMajor": (4, 2),
        "Orbital": (4, 0),
        "AntFrontal": (4, 1),
        "SupFrontal": (4, 2),
        "Motor": (4, 3),
        "SupParietal": (5, 0),
        "Temporal": (5, 1),
        "PostParietal": (5, 2),
        "Occipital": (5, 3),
    }
)


def plot_tract_profiles(
    X,
    groups=None,
    group_names=None,
    group_by=None,
    group_by_name=None,
    bins=None,
    quantiles=None,
    palette="colorblind",
    ci=95.0,
    subplot_positions=None,
    nrows=None,
    ncols=None,
    legend_kwargs=None,
    figsize=None,
    fig_tight_layout_kws=None,
):
    """Plot profiles for each bundle and each metric.

    Parameters
    ----------
    X : numpy.ndarray or AFQDataset class instance
        If array, this is a matrix of tractometry features with shape (n_subjects, n_features).

    groups : list of numpy.ndarray, optional
        feature indices for each feature group of ``X``.
        Must be provided if ``X`` is an array. Should not be provided if
        ``X`` is an AFQDataset.

    group_names : list of tuples
        the multi-indexed name for each group in ``groups``. Must be of same
        length as ``groups``. Must be provided if ``X`` is an array.
        Should not be provided if ``X`` is an AFQDataset

    group_by : list-like
        grouping variable that will produce different bundle profiles with
        different colors. If ``group_by`` is categorical, this function will
        produce a mean bundle profile for each category and color them
        differently. If ``group_by`` is numerical, please also provide the
        ``bins`` or ``quantiles`` parameter to convert this variable into a
        categorical variable. Default: no grouping.

    group_by_name : str
        The name of the group_by variable to be used in the plot legend.

    bins : int, sequence of scalars, or pandas.IntervalIndex
        The criteria by which to bin the ``group_by`` data.

        * int : Defines the number of equal-width bins in the range of `X`. The
          range of `X` is extended by .1% on each side to include the minimum
          and maximum values of `X`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    quantiles : int or list-like of float
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.

    palette : string, list, dict, or matplotlib.colors.Colormap, default="colorblind"
        Method for choosing the colors to use when mapping the ``group_by``
        onto hue. String values are passed to seaborn.color_palette(). List
        or dict values imply categorical mapping, while a colormap object
        implies numeric mapping.

    ci : float, default=95.0
        Confidence interval with which to shade the bundle profiles

    subplot_positions : dict, optional
        Dictionary with keys corresponding to bundle names and values
        corresponding to the subplot positions. If not provided, this function
        will use afqinsight.plot.POSITIONS.

    nrows : int, optional
        Number of subplot rows.

    ncols : int, optional
        Number of subplot columns.

    legend_kwargs : dict, optional
        Keyword arguments to pass to the legend.

    figsize : tuple, optional
        Figure size for each figure.

    fig_tight_layout_kws : dict, optional
        Keyword arguments to pass to fig.tight_layout.

    Returns
    -------
    figs : dict
        dictionary of matplotlib figures, with keys corresponding to the
        different diffusion metrics
    """
    if isinstance(X, AFQDataset):
        if groups is not None or group_names is not None:
            raise ValueError(
                "You provided an AFQDataset class instance as `X` input and also a `groups` or `group_names` input, but these are mutually exclusive."
            )
        # Allocate the variables needed below based on the input dataset:
        group_names = X.group_names
        groups = X.groups
        X = X.X

    else:
        if groups is None or group_names is None:
            raise ValueError(
                "You provided an array input as `X` but did not provide both a `groups` and a `group_names` input. You must provide both of these for array input. "
            )

    plt_positions = subplot_positions if subplot_positions is not None else POSITIONS

    if bins is not None and quantiles is not None:
        raise ValueError(
            "You specified both bins and quantiles. These parameters are mutually exclusive."
        )

    if (bins is not None or quantiles is not None) and group_by_name is None:
        raise ValueError(
            "You must supply a group_by_name when binning using either the bins or quantiles parameter."
        )

    if group_by is None:
        if group_by_name is not None:
            raise ValueError(
                "You must supply a `group_by` value if a `group_by_name` is provided."
            )
        group_by = np.ones(X.shape[0])
        group_by_name = None

    figs = {}

    metrics = np.unique([grp[0] for grp in group_names])
    tract_names = np.unique([grp[1] for grp in group_names])

    if len(group_by.shape) == 1:
        group_by = np.copy(group_by)[:, np.newaxis]

    groups_metric = [grp for gn, grp in zip(group_names, groups) if metrics[0] in gn]

    for metric in tqdm(metrics):
        X_metric = GroupExtractor(
            select=metric, groups=groups, group_names=group_names
        ).fit_transform(X)
        group_names_metric = [gn for gn in group_names if metric in gn]

        # Create a dataframe for each bundle
        tract_stats = {}
        for tid in tract_names:
            X_select = GroupExtractor(
                select=tid, groups=groups_metric, group_names=group_names_metric
            ).fit_transform(X_metric)
            columns = [idx for idx in range(X_select.shape[1])]
            df = pd.concat(
                [
                    pd.DataFrame(X_select, columns=columns, dtype=np.float64),
                    pd.DataFrame(group_by, columns=["group_by"]),
                ],
                axis="columns",
            )
            id_vars = ["group_by"]
            hue = "group_by"  # used later in seaborn functions

            if quantiles is not None:
                df["quantile"], _bins = pd.qcut(
                    df["group_by"].astype(np.float64), quantiles, retbins=True
                )
                hue = "quantile"
                id_vars.append("quantile")

            if bins is not None:
                df["bin"], _bins = pd.cut(
                    df["group_by"].astype(np.float64), bins, retbins=True
                )
                hue = "bin"
                id_vars.append("bin")

            df = df.melt(id_vars=id_vars, var_name="nodeID", value_name=metric)
            df["nodeID"] = df["nodeID"].astype(int)
            tract_stats[tid] = df

        # Arrange the bundles into a grid
        bgcolor = "white"

        # Create the subplots
        if nrows is None:
            nrows = 5
            cc_bundles = ["PostParietal", "SupFrontal", "SupParietal", "Temporal"]
            if any([tid in tract_stats.keys() for tid in cc_bundles]):
                nrows = 6

        ncols = ncols if ncols is not None else 4

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=figsize)

        for tid, df_stat in tqdm(tract_stats.items()):
            bundle_id = BUNDLE_MAT_2_PYTHON.get(tid, tid)

            if "HCC_" in bundle_id or "Cingulum Hippocampus" in tid:
                continue

            ax = axes[plt_positions[bundle_id][0], plt_positions[bundle_id][1]]

            if metric == "dki_md":
                df_stat[metric] *= 1000.0

            _ = sns.lineplot(
                x="nodeID",
                y=metric,
                hue=hue,
                data=df_stat,
                ci=ci,
                palette=palette,
                ax=ax,
                linewidth=1.0,
                n_boot=500,
            )

            if plt_positions[bundle_id][0] == nrows - 1:
                _ = ax.set_xlabel("% distance along fiber bundle")

            if plt_positions[bundle_id][1] == 0 or (
                plt_positions[bundle_id][1] == 1
                and plt_positions[bundle_id][0] == 4
                and nrows == 5
                and subplot_positions is None
            ):
                _ = ax.set_ylabel(metric.lower().replace("_", " "))
            else:
                _ = ax.set(ylabel=None)

            _ = ax.tick_params(axis="both", which="major")
            _ = ax.set_facecolor(bgcolor)
            _ = ax.get_legend().remove()

            _ = ax.set_title(
                bundle_id.replace("_", "").replace("FA", "CFA").replace("FP", "CFP")
            )

        if subplot_positions is None and nrows == 5:
            _ = axes[4, 0].axis("off")
            _ = axes[4, 3].axis("off")

        handles, labels = ax.get_legend_handles_labels()

        if quantiles is not None or bins is not None:
            labels = [
                f"{group_by_name} {b[0]:.2f}-{b[1]:.2f}"
                for b in zip(_bins[:-1], _bins[1:])
            ]
        if group_by_name is not None:
            figlegend_kwargs = dict(
                facecolor="whitesmoke",
                bbox_to_anchor=(0.5, 0.02),
                loc="upper center",
                ncol=6,
            )

            if legend_kwargs is not None:
                figlegend_kwargs.update(legend_kwargs)

            leg = plt.figlegend(
                handles,
                labels,
                **figlegend_kwargs,
            )

            # set the linewidth of each legend object
            for legobj in leg.legendHandles:
                _ = legobj.set_linewidth(3.0)

        if fig_tight_layout_kws is None:
            fig_tight_layout_kws = dict(h_pad=0.5, w_pad=-0.5)

        fig.tight_layout(**fig_tight_layout_kws)

        figs[metric] = fig

    return figs
