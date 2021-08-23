"""Plot bundle profiles."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import OrderedDict
from groupyr.transform import GroupExtractor
from tqdm.auto import tqdm

from .utils import BUNDLE_MAT_2_PYTHON


POSITIONS = OrderedDict(
    {
        "ATR_L": (1, 0),
        "ATR_R": (1, 3),
        "CST_L": (1, 1),
        "CST_R": (1, 2),
        "CGC_L": (3, 1),
        "CGC_R": (3, 2),
        "FP": (0, 3),
        "FA": (0, 0),
        "CC_ForcepsMinor": (0, 3),
        "CC_ForcepsMajor": (0, 0),
        "IFO_L": (4, 1),
        "IFO_R": (4, 2),
        "IFOF_L": (4, 1),
        "IFOF_R": (4, 2),
        "HCC_L": (4, 0),
        "HCC_R": (4, 3),
        "ILF_L": (3, 0),
        "ILF_R": (3, 3),
        "SLF_L": (2, 1),
        "SLF_R": (2, 2),
        "ARC_L": (2, 0),
        "ARC_R": (2, 3),
        "UNC_L": (0, 1),
        "UNC_R": (0, 2),
    }
)

POSITIONS_CALLOSAL = OrderedDict(
    {
        "Orbital": (0),
        "AntFrontal": (1),
        "SupFrontal": (2),
        "Motor": (3),
        "SupParietal": (4),
        "PostParietal": (5),
        "Temporal": (6),
        "Occipital": (7),
    }
)


def plot_tract_profiles(
    X,
    groups,
    group_names,
    group_by,
    group_by_name=None,
    bins=None,
    quantiles=None,
    palette="colorblind",
):
    """Plot profiles for each bundle and each metric.

    Parameters
    ----------
    X : numpy.ndarray
        matrix of tractometry features with shape (n_subjects, n_features).

    groups : list of numpy.ndarray
        feature indices for each feature group of ``X``

    group_names : list of tuples
        the multi-indexed name for each group in ``groups``. Must be of same
        length as ``groups``.

    group_by : list-like
        grouping variable that will produce different bundle profiles with
        different colors. If ``group_by`` is categorical, this function will
        produce a mean bundle profile for each category and color them
        differently. If ``group_by`` is numerical, please also provide the
        ``bins`` or ``quantiles`` parameter to convert this variable into a
        categorical variable.

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

    Returns
    -------
    figs : dict
        dictionary of matplotlib figures, with keys corresponding to the
        different diffusion metrics
    """
    if bins is not None and quantiles is not None:
        raise ValueError(
            "You specified both bins and quantiles. These parameters are mutually exclusive."
        )

    if (bins is not None or quantiles is not None) and group_by_name is None:
        raise ValueError(
            "You must supply a group_by_name when binning using either the bins or quantiles parameter."
        )

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

            if tid in POSITIONS_CALLOSAL.keys():
                plot_callosal = True
                fig_cc, axes_cc = plt.subplots(nrows=1, ncols=8)

        # Arrange the bundles into a grid
        bgcolor = "white"

        # Create the subplots
        fig, axes = plt.subplots(nrows=5, ncols=4, sharex=True)

        for tid, df_stat in tqdm(tract_stats.items()):
            bundle_id = BUNDLE_MAT_2_PYTHON.get(tid, tid)

            if "HCC_" in bundle_id or "Cingulum Hippocampus" in tid:
                continue

            if tid in POSITIONS_CALLOSAL.keys():
                is_callosal = True
                ax = axes_cc[POSITIONS_CALLOSAL[tid]]
            else:
                is_callosal = False
                ax = axes[POSITIONS[bundle_id][0], POSITIONS[bundle_id][1]]

            if metric == "dki_md":
                df_stat[metric] *= 1000.0

            _ = sns.lineplot(
                x="nodeID",
                y=metric,
                hue=hue,
                data=df_stat,
                ci=68.2,
                palette=palette,
                ax=ax,
                linewidth=1.0,
                n_boot=500,
            )

            if is_callosal or POSITIONS[bundle_id][0] == 4:
                _ = ax.set_xlabel("% distance along fiber bundle")

            if (is_callosal and tid == "Orbital") or (
                POSITIONS[bundle_id][1] == 0
                or (POSITIONS[bundle_id][1] == 1 and POSITIONS[bundle_id][0] == 4)
            ):
                if metric in ["md", "dki_md"]:
                    _ = ax.set_ylabel(
                        r"{} $\left[ \mu \mathrm{{m}}^2 / \mathrm{{ms}} \right]$".format(
                            metric.lower().replace("_", " ")
                        )
                    )
                else:
                    _ = ax.set_ylabel(metric.lower().replace("_", " "))
            else:
                _ = ax.set(ylabel=None)

            _ = ax.tick_params(axis="both", which="major")
            _ = ax.set_facecolor(bgcolor)
            _ = ax.get_legend().remove()

            _ = ax.set_title(
                bundle_id.replace("_", "").replace("FA", "CFA").replace("FP", "CFP")
            )
        if not is_callosal:
            _ = axes[4, 0].axis("off")
            _ = axes[4, 3].axis("off")

        if not is_callosal and plot_callosal:
            _ = axes[0, 0].axis("off")
            _ = axes[0, 3].axis("off")

        handles, labels = ax.get_legend_handles_labels()

        if quantiles is not None or bins is not None:
            labels = [
                f"{group_by_name} {b[0]:.2f}-{b[1]:.2f}"
                for b in zip(_bins[:-1], _bins[1:])
            ]

        leg = plt.figlegend(
            handles,
            labels,
            facecolor="whitesmoke",
            bbox_to_anchor=(0.5, 0.02),
            loc="upper center",
            ncol=6,
        )

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            _ = legobj.set_linewidth(3.0)

        fig.tight_layout(h_pad=0.5, w_pad=-0.5)

        if plot_callosal:
            fig_cc.tight_layout(h_pad=0.5, w_pad=-0.5)
            figs[metric] = [fig, fig_cc]
        else:
            figs[metric] = fig

    return figs
