"""
Create diagnostic plots of AFQ-Insight output
"""
from __future__ import absolute_import, division, print_function

import itertools
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import palettable
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import ColorBar, CustomJS, HoverTool, Range1d, Title
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Spectral10
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.resources import CDN
from matplotlib.colors import to_hex

from . import utils
from .insight import _sigmoid

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
def plot_betas(beta_hat, columns, ecdf=False, output_root_name=None):
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

    output_root_name : string or None, default=None
        Filename for bokeh html output. If None, figure will not be saved

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

    if output_root_name is not None:
        html = file_html(p, CDN, "my plot")
        with open(op.abspath(output_root_name), 'w') as fp:
            fp.write(html)
    else:
        show(p)


@registered
def plot_classification_probabilities(x, y, cv_results, output_root_name=None):
    """Plot the classification probabilities for each cross-validation split

    Parameters
    ----------
    x : numpy.ndarray
        The original feature matrix

    y : numpy.ndarray
        The target array (i.e. "ground truth")

    cv_results : list of SGLResult namedtuples
        Results of each cross-validation split

    output_root_name : string or None, default=None
        Filename for bokeh html output. If None, figure will not be saved
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

    if output_root_name is not None:
        html = file_html(p, CDN, "my plot")
        with open(op.abspath(output_root_name), 'w') as fp:
            fp.write(html)
    else:
        show(p)


@registered
def plot_unfolded_beta(unfolded_beta, output_root_name=None):
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

    output_root_name : string or None, default=None
        Filename for bokeh html output. If None, figure will not be saved
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

    if output_root_name is not None:
        html = file_html(p, CDN, "my plot")
        with open(op.abspath(output_root_name), 'w') as fp:
            fp.write(html)
    else:
        show(p)


@registered
def plot_pca_space_classification(x2_sgl, y, pca_sgl=None, beta=None,
                                  x2_orig=None, output_root_name=None):
    """Plot classification predictions in a 2-component PCA space.

    This function has two plot modes, specified by the presence or
    absence of certain input variables. If `x2_orig` is not None,
    this plots side-by-side scatter plots of the target class in
    2-D PCA space. The right plot is the post-SGL weighted feature
    matrix and the left plot is the pre-SGL original feature matrix.

    If `pca_sgl` and `beta` are not None, then this plots only the
    post-SGL weighted feature space and also plots a contour of the
    classification probabilities (from which one can infer the
    decision boundary).

    Parameters
    ----------
    x2_sgl : numpy.ndarray
        Projection of the feature matrix onto its first two principal
        components, after the feature matrix has been weighted by the
        regression coefficients (i.e. beta)

    y : pandas.Series
        Binary classification target array

    pca_sgl : sklearn.decomposition.pca.PCA
        PCA decomposition that has been fitted to the full post-SGL
        weighted feature matrix

    beta : numpy.ndarray
        Regression coefficients

    x2_orig :
        Projection of the original (pre-SGL) feature matrix onto its
        first two principal components.

    output_root_name : string or None, default=None
        Filename for bokeh html output. If None, figure will not be saved
    """
    if x2_orig is None and any([
        pca_sgl is None, beta is None
    ]):
        raise ValueError('You must supply either (`pca_sgl` and `beta`) or '
                         '`x2_orig`.')
    colors = [Spectral10[1 - 2 * b] for b in y]
    als = y.copy()
    als[y == 1] = 'ALS'
    als[y == 0] = 'Control'

    pc_info = {
        'pc0_sgl': x2_sgl[:, 0],
        'pc1_sgl': x2_sgl[:, 1],
        'als': als,
        'subject_id': y.index,
        'colors': colors
    }

    ps = [None]

    if x2_orig is not None:
        pc_info['pc0_orig'] = x2_orig[:, 0]
        pc_info['pc1_orig'] = x2_orig[:, 1]
        ps = [None] * 2

    tooltips = [
        ("subject", "@subject_id"),
        ("status", "@als"),
    ]

    source = ColumnDataSource(data=pc_info)
    code = "source.set('selected', cb_data.index);"
    callback = CustomJS(args={'source': source}, code=code)

    if x2_orig is None:
        ps[0] = figure(plot_width=550, plot_height=500,
                       toolbar_location='right')

        npoints = 100
        dx = np.max(x2_sgl[:, 0]) - np.min(x2_sgl[:, 0])
        xmid = 0.5 * (np.max(x2_sgl[:, 0]) + np.min(x2_sgl[:, 0]))
        xmin = xmid - (dx * 1.1 / 2.0)
        xmax = xmid + (dx * 1.1 / 2.0)

        dy = np.max(x2_sgl[:, 1]) - np.min(x2_sgl[:, 1])
        ymid = 0.5 * (np.max(x2_sgl[:, 1]) + np.min(x2_sgl[:, 1]))
        ymin = ymid - (dy * 1.1 / 2.0)
        ymax = ymid + (dy * 1.1 / 2.0)

        x_subspace = np.linspace(xmin, xmax, npoints)
        y_subspace = np.linspace(ymin, ymax, npoints)
        subspace_pairs = np.array([
            [p[0], p[1]] for p in itertools.product(x_subspace, y_subspace)
        ])
        bigspace_pairs = pca_sgl.inverse_transform(subspace_pairs)
        predict_pairs = _sigmoid(bigspace_pairs.dot(beta))
        x_grid, y_grid = np.meshgrid(x_subspace, y_subspace)
        p_grid = predict_pairs.reshape(x_grid.shape).transpose()

        cmap = plt.get_cmap('RdBu')
        colors = [to_hex(c) for c in cmap(np.linspace(1, 0, 256))]

        ps[0].image(image=[p_grid], x=xmin, y=ymin,
                    dw=dx * 1.1,
                    dh=dy * 1.1,
                    palette=colors)

        color_mapper = LinearColorMapper(palette=colors, low=0, high=1)
        color_bar = ColorBar(
            color_mapper=color_mapper,
            ticker=FixedTicker(ticks=np.arange(0, 1.1, 0.1)),
            label_standoff=8, border_line_color=None, location=(0, 0)
        )

        ps[0].add_layout(color_bar, 'right')

        ps[0].x_range = Range1d(xmin, xmax)
        ps[0].y_range = Range1d(ymin, ymax)
    else:
        ps[0] = figure(plot_width=500, plot_height=500,
                       toolbar_location='right')

    ps[0].title.text = 'Classification in Post-SGL PCA space'
    s0 = ps[0].scatter('pc0_sgl', 'pc1_sgl', source=source,
                       size=10, fill_color='colors', line_color='white',
                       line_width=1.5,
                       legend='als')
    hover0 = HoverTool(tooltips=tooltips, callback=callback, renderers=[s0])
    ps[0].add_tools(hover0)

    if x2_orig is not None:
        ps[1] = figure(plot_width=500, plot_height=500,
                       toolbar_location='right')
        ps[1].title.text = 'Classification in Original PCA space'
        s1 = ps[1].scatter('pc0_orig', 'pc1_orig', source=source,
                           size=10, fill_color='colors',
                           line_color='white', legend='als')
        hover1 = HoverTool(tooltips=tooltips,
                           callback=callback,
                           renderers=[s1])
        ps[1].add_tools(hover1)

    for idx in range(len(ps)):
        ps[idx].xaxis.axis_label = "1st Principal Component"
        ps[idx].yaxis.axis_label = "2nd Principal Component"

    layout = row(ps[::-1])

    if output_root_name is not None:
        html = file_html(layout, CDN, "my plot")
        with open(op.abspath(output_root_name), 'w') as fp:
            fp.write(html)
    else:
        show(layout)
