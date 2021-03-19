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

BUNDLE_MAT_2_PYTHON = {
    "Right Corticospinal": "CST_R",
    "Left Corticospinal": "CST_L",
    "RightCorticospinal": "CST_R",
    "LeftCorticospinal": "CST_L",
    "Right Uncinate": "UNC_R",
    "Left Uncinate": "UNC_L",
    "RightUncinate": "UNC_R",
    "LeftUncinate": "UNC_L",
    "Left IFOF": "IFO_L",
    "Right IFOF": "IFO_R",
    "LeftIFOF": "IFO_L",
    "RightIFOF": "IFO_R",
    "Right Arcuate": "ARC_R",
    "Left Arcuate": "ARC_L",
    "RightArcuate": "ARC_R",
    "LeftArcuate": "ARC_L",
    "Right Thalamic Radiation": "ATR_R",
    "Left Thalamic Radiation": "ATR_L",
    "RightThalamicRadiation": "ATR_R",
    "LeftThalamicRadiation": "ATR_L",
    "Right Cingulum Cingulate": "CGC_R",
    "Left Cingulum Cingulate": "CGC_L",
    "RightCingulumCingulate": "CGC_R",
    "LeftCingulumCingulate": "CGC_L",
    "Right Cingulum Hippocampus": "HCC_R",
    "Left Cingulum Hippocampus": "HCC_L",
    "RightCingulumHippocampus": "HCC_R",
    "LeftCingulumHippocampus": "HCC_L",
    "Callosum Forceps Major": "FP",
    "Callosum Forceps Minor": "FA",
    "CallosumForcepsMajor": "FP",
    "CallosumForcepsMinor": "FA",
    "Right ILF": "ILF_R",
    "Left ILF": "ILF_L",
    "RightILF": "ILF_R",
    "LeftILF": "ILF_L",
    "Right SLF": "SLF_R",
    "Left SLF": "SLF_L",
    "RightSLF": "SLF_R",
    "LeftSLF": "SLF_L",
}


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
