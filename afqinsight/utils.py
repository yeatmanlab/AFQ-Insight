"""Utility functions for AFQ-Insight."""
import numpy as np

__all__ = ["ecdf"]


CANONICAL_TRACT_NAMES = [
    "Left Arcuate",
    "Left SLF",
    "Left Uncinate",
    "Left ILF",
    "Left IFOF",
    "Left Cingulum Hippocampus",
    "Left Thalamic Radiation",
    "Left Corticospinal",
    "Left Cingulum Cingulate",
    "Callosum Forceps Minor",
    "Callosum Forceps Major",
    "Right Cingulum Cingulate",
    "Right Corticospinal",
    "Right Thalamic Radiation",
    "Right Cingulum Hippocampus",
    "Right IFOF",
    "Right ILF",
    "Right Uncinate",
    "Right SLF",
    "Right Arcuate",
]


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
    x : numpy.ndarray
        sorted data
    y : numpy.ndarray
        cumulative probability
    """
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)
    if reverse:
        x = np.flip(x)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y
