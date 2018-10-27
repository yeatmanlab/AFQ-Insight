"""
Create diagnostic plots of AFQ-Insight output
"""
from __future__ import absolute_import, division, print_function

import json
import numpy as np
import os.path as op
import palettable

from bokeh.embed import json_item
from bokeh.layouts import column
from bokeh.models import HoverTool, Title, Range1d
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, show, ColumnDataSource

from . import utils
from .insight import _sigmoid

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
def plot_betas(beta_hat, columns, ecdf=False, output_json=None):
    """Plot the classification probabilities for each cross-validation split

    Parameters
    ----------
    beta_hat : OrderedDict of beta coeffs
        Two-level ordered dict with beta_hat coefficients, ordered first
        by tract and then by metric

    columns : pd.MultiIndex
        MultiIndex columns of the feature matrix

    ecdf : bool
        If True, plot the estimated cumulative probability distribution (ECDF)
        of the beta coefficients. If False, plot the raw coefficients.

    output_json : string or None, default=None
        Filename for bokeh json output. If None, figure will not be saved

    See Also
    --------
    transform.AFQFeatureTransformer
        Transforms AFQ csv files into feature matrix. Use this to create
        the `columns` input.

    transform.beta_hat_by_groups
        Returns a two-level ordered dict instead of "unfolding" the tracts
    """
    all_metrics = columns.levels[columns.names.index('metric')]
    ps = []

    colors = {}
    for idx, metric in enumerate(all_metrics):
        colors[metric] = Spectral10[idx]

    for idx, tract in enumerate(beta_hat.keys()):
        ps.append(figure(plot_width=750, plot_height=250,
                         toolbar_location='right'))
        ps[idx].title.text = tract

        for metric in beta_hat[tract].keys():
            b = beta_hat[tract][metric]
            if not all(b == 0):
                if ecdf:
                    cdf = utils.ecdf(b)
                    ps[idx].circle(cdf.x, cdf.y,
                                   size=5, color=colors[metric],
                                   alpha=0.8, legend=metric)
                else:
                    ps[idx].line(np.arange(len(b)), b,
                                 line_width=4, color=colors[metric],
                                 alpha=0.8, legend=metric)

        ps[idx].legend.location = 'bottom_right'
        ps[idx].legend.click_policy = 'hide'

    p = column(ps)

    if output_json is not None:
        item_text = json.dumps(json_item(p, "myplot"))
        with open(op.abspath(output_json), 'w') as fp:
            fp.write(item_text)

    show(p)


@registered
def plot_classification_probabilities(x, y, cv_results, output_json=None):
    """Plot the classification probabilities for each cross-validation split

    Parameters
    ----------
    x : numpy.ndarray
        The original feature matrix

    y : numpy.ndarray
        The target array (i.e. "ground truth")

    cv_results : list of SGLResult namedtuples
        Results of each cross-validation split

    output_json : string or None, default=None
        Filename for bokeh json output. If None, figure will not be saved
    """
    p = figure(plot_width=700, plot_height=700, toolbar_location='above')
    p.title.text = 'Classification probabilities for each CV split'
    p.add_layout(
        Title(text='Click on legend entries to hide/show corresponding lines',
              align="left"), 'right'
    )

    names = ['cv_idx = {i:d}'.format(i=i) for i in range(len(cv_results))]

    hover = HoverTool(
        tooltips=[("index", "$index"), ],
        mode='vline',
    )
    hover.point_policy = 'snap_to_data'
    hover.line_policy = 'nearest'

    for res, color, name in zip(cv_results, Spectral10, names):
        p.line(np.arange(len(y)), _sigmoid(x.dot(res.beta_hat)),
               line_width=2, color=color, alpha=0.8, legend=name)

    p.line(np.arange(len(y)), y,
           line_width=3, alpha=0.8, legend='ground truth')
    p.line(np.arange(len(y)), 0.5 * np.ones(len(y)),
           line_width=2, line_dash='dashed', alpha=0.8, legend='threshold')
    p.add_tools(hover)
    p.legend.location = 'top_right'
    p.legend.click_policy = 'hide'

    if output_json is not None:
        item_text = json.dumps(json_item(p, "myplot"))
        with open(op.abspath(output_json), 'w') as fp:
            fp.write(item_text)

    show(p)


@registered
def plot_unfolded_beta(unfolded_beta, output_json=None):
    """Plot the regression coefficients on the "unfolded" brain.

    All tracts are represented contiguously on the x-axis in this order:
    - left hemisphere (lateral to medial)
    - callosal tracts (anterior then posterior)
    - Right hemisphere (medial to lateral)
    Each metric is plotted separately

    Parameters
    ----------
    unfolded_beta : dict
        Dictionary of beta arrays, where the keys are the diffusion metrics
        (e.g. 'fa', 'md') and the values are the contiguous regression
        coefficient arrays

    output_json : string or None, default=None
        Filename for bokeh json output. If None, figure will not be saved
    """
    p = figure(plot_width=700, plot_height=700, toolbar_location='above')
    p.title.text = 'Feature weights in the "unfolded" brain'
    p.add_layout(
        Title(text='Click on legend entries to hide/show corresponding lines',
              align="left"), 'right'
    )

    len_alltracts = len(unfolded_beta[list(unfolded_beta.keys())[0]])
    n_tracts = len(utils.canonical_tract_names)
    n_nodes = len_alltracts / n_tracts
    ticks = np.arange(0, len_alltracts, n_nodes)

    p.x_range = Range1d(0, len_alltracts)
    p.xgrid.grid_line_color = None
    p.xaxis.ticker = FixedTicker(
        ticks=ticks + n_nodes / 2,
        minor_ticks=ticks
    )
    p.xaxis.major_label_orientation = np.pi / 2
    p.xaxis.major_label_overrides = {
        str(n): utils.canonical_tract_names[i]
        for i, n in enumerate(ticks + n_nodes / 2)
    }

    attrs = [
        'line_alpha',
        'line_cap',
        'line_color',
        'line_dash',
        'line_dash_offset',
        'line_join',
        'line_width',
        'in',
        'out',
    ]

    for attr in attrs:
        major = 'major_tick_' + attr
        minor = 'minor_tick_' + attr
        setattr(p.xaxis[0], minor, getattr(p.xaxis[0], major))
        setattr(p.xaxis[0], major, None)

    top = 0.0
    bottom = 0.0
    for metric in unfolded_beta.keys():
        if np.max(unfolded_beta[metric]) > top:
            top = np.max(unfolded_beta[metric])
        if np.min(unfolded_beta[metric]) < bottom:
            bottom = np.min(unfolded_beta[metric])

    top *= 1.0 + p.y_range.range_padding
    bottom *= 1.0 + p.y_range.range_padding

    tract_colors = np.copy(palettable.tableau.Tableau_20.hex_colors).tolist()
    tract_colors = (tract_colors[1:-2:2]
                    + tract_colors[-2:]
                    + tract_colors[-3::-2])

    for i, color in enumerate(tract_colors):
        p.quad(top=[top], bottom=[bottom], left=[ticks[i]],
               right=[ticks[i] + n_nodes], color=color, alpha=0.3)

    unfolded_beta['x'] = np.arange(len_alltracts)
    source = ColumnDataSource(data=unfolded_beta)

    for name, color in zip(unfolded_beta.keys(), Spectral10):
        if name != 'x':
            p.line(x='x', y=name, source=source,
                   line_width=2, color=color, legend=dict(value=name))

    p.legend.location = 'bottom_left'
    p.legend.click_policy = 'hide'

    p.y_range = Range1d(bottom, top)

    if output_json is not None:
        item_text = json.dumps(json_item(p, "myplot"))
        with open(op.abspath(output_json), 'w') as fp:
            fp.write(item_text)

    show(p)


@registered
def plot_subspace_2d():
    pass
