"""Data augmentation methods for (potentially multi-channel) one-dimensional data.

This module provides data augmentation methods for one-dimensional sequences.
The code itself borrows heavily from the code accompanying ref [1]_, which is
available at https://github.com/uchidalab/time_series_augmentation and is
licensed under the Apache-2.0 license. The code here has been modified to allow
independent multi-channel input. That is, is allows independent data
augmentation for each channel of multi-channel input data.

References
----------
.. [1]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
augmentation for time series classification with neural networks," PLOS ONE
16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841
"""
from .augmentation import *  # noqa: F401,F403
