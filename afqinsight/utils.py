from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
def ecdf(data, reverse=False):
    """Compute ECDF for a one-dimensional array of measurements.

    Parameters
    ----------
    data : np.ndarray
        one-dimensional array of measurements

    reverse : bool, default=False
        If True, reverse the sorted data so that ecdf runs from top-left
        to bottom-right.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        x - sorted data
        y - cumulative probability
    """
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)
    if reverse:
        x = np.flip(x)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    ECDF = namedtuple('ECDF', 'x y')
    return ECDF(x=x, y=y)


@registered
def plot_ecdf(data, reverse=False):
    """Plot ECDF for a one-dimensional array of measurements.

    Parameters
    ----------
    data : np.ndarray
        one-dimensional array of measurements

    reverse : bool, default=False
        If True, reverse the sorted data so that ecdf runs from top-left
        to bottom-right.
    """
    cdf = ecdf(data, reverse=reverse)

    # Generate plot
    plt.plot(cdf.x, cdf.y, marker='.', linestyle='none')

    # Make the margins nice
    plt.margins(0.02)

    # Label the axes
    plt.xlabel('data')
    plt.ylabel('ECDF')

    # Display the plot
    plt.show()
