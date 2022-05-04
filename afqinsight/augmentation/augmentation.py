"""Data augmentation methods for (potentially multi-channel) one-dimensional data.

This module provides data augmentation methods for one-dimensional sequences.
The code itself borrows heavily from the code accompanying ref [1]_, which is
available at https://github.com/uchidalab/time_series_augmentation and is
licensed under the Apache-2.0 license. The code here has been modified to allow
independent multi-channel input. That is, it allows independent data
augmentation for each channel of multi-channel input data.

References
----------
.. [1]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
augmentation for time series classification with neural networks," PLOS ONE
16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841
"""
import numpy as np
from tqdm import tqdm

from . import dtw

__all__ = [
    "jitter",
    "scaling",
    "permutation",
    "magnitude_warp",
    "time_warp",
    "window_slice",
    "window_warp",
    "spawner",
    "wdba",
    "random_guided_warp",
    "discriminative_guided_warp",
]


#    +--------------------------------------------------+
# ---| Random transformation-based augmentation methods |---
#    +--------------------------------------------------+


def jitter(x, sigma=0.03):
    """Add jitter, or noise, to 1D sequences [1]_.

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    sigma : float
        Standard deviation of the distribution to be added.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  Terry T. Um, et al., "Data augmentation of wearable sensor data for
    parkinson’s disease monitoring using convolutional neural networks,"
    Proceedings of the 19th ACM International Conference on Multimodal
    Interaction, 2017, DOI: 10.1145/3136755.3136817
    """
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    """Change the global magnitude of an input 1D sequence [1]_.

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    sigma : float
        Standard deviation of the scaling constant.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  Terry T. Um, et al., "Data augmentation of wearable sensor data for
    parkinson’s disease monitoring using convolutional neural networks,"
    Proceedings of the 19th ACM International Conference on Multimodal
    Interaction, 2017, DOI: 10.1145/3136755.3136817
    """
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def permutation(x, max_segments=5, seg_mode="equal"):
    """Return a random permutation of segments.

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    max_segments : int
        The maximum number of segments to use. The minimum number is 1.

    seg_mode : ["equal", "random"]
        If "equal", use equal sized permutation segments. If "random", use
        randomly sized segments.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  Terry T. Um, et al., "Data augmentation of wearable sensor data for
    parkinson’s disease monitoring using convolutional neural networks,"
    Proceedings of the 19th ACM International Conference on Multimodal
    Interaction, 2017, DOI: 10.1145/3136755.3136817
    """
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(
                    x.shape[1] - 2, num_segs[i] - 1, replace=False
                )
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):
    """Warp the magnitude of the input sequence.

    Scale the magnitude of each time series by a curve created by cubic spline
    with a set number of knots at random magnitudes [1]_.

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    sigma : float
        Standard devation of the random magnitudes.

    knot : int
        Number of knots, i.e. hills/valleys in the spline.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  Terry T. Um, et al., "Data augmentation of wearable sensor data for
    parkinson’s disease monitoring using convolutional neural networks,"
    Proceedings of the 19th ACM International Conference on Multimodal
    Interaction, 2017, DOI: 10.1145/3136755.3136817
    """
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [
                CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                for dim in range(x.shape[2])
            ]
        ).T
        ret[i] = pat * warper

    return ret


def time_warp(x, sigma=0.2, knot=4):
    """Return a random smooth time warping of the input sequences.

    This function warps the time steps based on a smooth curve defined by a cubic spline.
    See refs [1]_ and [2]_ for further details.

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    sigma : float
        Standard devation of the random magnitudes of the warping path.

    knot : int
        Number of knots, i.e. hills/valleys in the warping path.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  Terry T. Um, et al., "Data augmentation of wearable sensor data for
    parkinson’s disease monitoring using convolutional neural networks,"
    Proceedings of the 19th ACM International Conference on Multimodal
    Interaction, 2017, DOI: 10.1145/3136755.3136817

    .. [2]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
    augmentation for time series classification with neural networks," PLOS ONE
    16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841

    """
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(
                warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim]
            )(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(
                orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]
            ).T
    return ret


def window_slice(x, reduce_ratio=0.9, warp_channels_independently=False):
    """Crop the input sequence by the `reduce_ratio`.

    This function will randomly crop from both the left and right to fulfill the
    ``reduce_ratio`` requirement. The result is then "stretched" to fit on the
    original time domain. See refs [1]_ and [2]_ for further details.

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    reduce_ratio : float
        The amount of input sequence that should remain. A value of 1.0
        corresponds to no cropping.

    warp_channels_independently : bool, default=False
        If True, use independent warping windows and scales for each channel.
        If False, each channel will be warped in the same way.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  A. Le Guennec, S. Malinowski, R. Tavenard, "Data Augmentation for
    Time Series Classification using Convolutional Neural Networks," in
    ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data,
    2016. HAL Id: halshs-01357973,
    URL: https://halshs.archives-ouvertes.fr/halshs-01357973.

    .. [2]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
    augmentation for time series classification with neural networks," PLOS ONE
    16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841

    """
    if warp_channels_independently:
        shape = (x.shape[0], x.shape[2])
    else:
        shape = x.shape[0]

    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[1] - target_len, size=(shape)
    ).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            if warp_channels_independently:
                start = starts[i, dim]
                end = ends[i, dim]
            else:
                start = starts[i]
                end = ends[i]

            ret[i, :, dim] = np.interp(
                np.linspace(0, target_len, num=x.shape[1]),
                np.arange(target_len),
                pat[start:end, dim],
            ).T
    return ret


def window_warp(
    x, window_ratio=0.1, scales=[0.5, 2.0], warp_channels_independently=False
):
    """Randomly warp a time window by scales.

    Window warping takes a random window of the input sequence and stretches or
    contracts it. See refs [1]_ and [2]_. The contraction scale is chosen randomly from the ``scales``
    parameter.

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    window_ratio : float
        Ratio of the window to the full time series.

    scales : int
        A list ratios to warp the window by.

    warp_channels_independently : bool, default=False
        If True, use independent warping windows and scales for each channel.
        If False, each channel will be warped in the same way.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  A. Le Guennec, S. Malinowski, R. Tavenard, "Data Augmentation for
    Time Series Classification using Convolutional Neural Networks," in
    ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data,
    2016. HAL Id: halshs-01357973,
    URL: https://halshs.archives-ouvertes.fr/halshs-01357973.

    .. [2]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
    augmentation for time series classification with neural networks," PLOS ONE
    16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841

    """
    if warp_channels_independently:
        shape = (x.shape[0], x.shape[2])
    else:
        shape = x.shape[0]

    warp_scales = np.random.choice(scales, shape)
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(
        low=1, high=x.shape[1] - warp_size - 1, size=(shape)
    ).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            if warp_channels_independently:
                w_start = window_starts[i, dim]
                w_scale = warp_scales[i, dim]
                w_end = window_ends[i, dim]
            else:
                w_start = window_starts[i]
                w_scale = warp_scales[i]
                w_end = window_ends[i]

            start_seg = pat[:w_start, dim]
            window_seg = np.interp(
                np.linspace(0, warp_size - 1, num=int(warp_size * w_scale)),
                window_steps,
                pat[w_start:w_end, dim],
            )
            end_seg = pat[w_end:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(
                np.arange(x.shape[1]),
                np.linspace(0, x.shape[1] - 1.0, num=warped.size),
                warped,
            ).T

    return ret


#    +-------------------------------------------+
# ---| Pattern mixing-based augmentation methods |---
#    +-------------------------------------------+


def spawner(x, labels, sigma=0.05, verbose=0):
    """Augment using SuboPtimAl Warped time-series geNEratoR (SPAWNER).

    From ref [2]_:
    "SuboPtimAl Warped time series geNEratoR (SPAWNER) [1]_ was introduced...
    as a method of generating patterns through a novel method called suboptimal
    time warping. Suboptimal time warping uses the warping ability of [dynamic
    time warping] DTW but adding an additional constraint that forces the
    warping path through a random point. By using the suboptimal time warping,
    SPAWNER is able to create an almost unlimited number of new time series by
    averaging aligned patterns."

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    labels : numpy.ndarray, 2D or 3D
        Either list of integers or one hot encoding of the labels.

    sigma : float
        Standard deviation of the jittering.

    verbose : int
        Verbosity flag. `1` prints out a DTW matrix. `0` shows nothing.

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  K. Kamycki, T. Kapuscinski, M. Oszust, "Data Augmentation with
    Suboptimal Warping for Time-Series Classification," Sensors, vol. 20, no. 1,
    2020. URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/

    .. [2]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
    augmentation for time series classification with neural networks," PLOS ONE
    16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841
    """
    random_points = np.random.randint(low=1, high=x.shape[1] - 1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.0).astype(int)
    orig_steps = np.arange(x.shape[1])
    _labels = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(_labels[choices] == _labels[i])[0]
        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw.dtw(
                pat[: random_points[i]],
                random_sample[: random_points[i]],
                dtw.RETURN_PATH,
                slope_constraint="symmetric",
                window=window,
            )
            path2 = dtw.dtw(
                pat[random_points[i] :],
                random_sample[random_points[i] :],
                dtw.RETURN_PATH,
                slope_constraint="symmetric",
                window=window,
            )
            combined = np.concatenate(
                (np.vstack(path1), np.vstack(path2 + random_points[i])), axis=1
            )
            if verbose:
                print(random_points[i])
                _, cost, DTW_map, path = dtw.dtw(
                    pat,
                    random_sample,
                    return_flag=dtw.RETURN_ALL,
                    slope_constraint="symmetric",
                    window=window,
                )
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(
                    orig_steps,
                    np.linspace(0, x.shape[1] - 1.0, num=mean.shape[0]),
                    mean[:, dim],
                ).T
        else:
            print(
                "There is only one pattern of class %d, skipping pattern average"
                % _labels[i]
            )
            ret[i, :] = pat
    return jitter(ret, sigma=sigma)


def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    """Weighted Dynamic Time Warping Barycenter Averaging (wDBA).

    Uses the Average Selected with Distance (ASD) version of DBA from ref [1]_.

    A summary from ref [2]_:
    "DTW Barycentric Averaging (DBA) [3]_ is a method of averaging multiple
    discrete time series by finding the center of time aligned elements. It does
    this through an iterative process of adding sample patterns that are time
    aligned by DTW to a cumulative centroid pattern. The advantage of using DBA
    over linear averaging is that the underlying pattern is preserved, whereas
    linear averaging might smooth features (for example, linear averaging time
    series that are just offset in time would lose distinct features).

    For data augmentation, Forestier et al. [1]_ proposed a weighted version of
    DBA (wDBA). They propose three weighting schemes, Average All (AA), Average
    Selected (AS), and Average Selected with Distance (ASD). AA weights all of
    the time series in a class input into wDBA by a flat Dirichlet distribution.
    AS selects a reference time series and weights two of the five nearest
    neighbors by a large constant amount and all the rest by a small constant
    amount. ASD is similar to AS except that it weights based on the distance to
    the reference.
    "

    References
    ----------
    .. [1]  G. Forestier, F. Petitjean, H. A. Dau, G. I. Webb and E. Keogh,
    "Generating Synthetic Time Series to Augment Sparse Datasets," 2017 IEEE
    International Conference on Data Mining (ICDM), 2017, pp. 865-870, DOI:
    10.1109/ICDM.2017.106.

    .. [2]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
    augmentation for time series classification with neural networks," PLOS ONE
    16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841

    .. [3]  Petitjean F, Ketterlin A, Gançarski P. A global averaging method for
    dynamic time warping, with applications to clustering. Pattern Recogn.
    2011;44(3):678–693. DOI: https://doi.org/10.1016/j.patcog.2010.09.013
    """
    if use_window:
        window = np.ceil(x.shape[1] / 10.0).astype(int)
    else:
        window = None
    _labels = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(_labels == _labels[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]

            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.0
                    else:
                        dtw_matrix[p, s] = dtw.dtw(
                            prototype,
                            sample,
                            dtw.RETURN_VALUE,
                            slope_constraint=slope_constraint,
                            window=window,
                        )

            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]

            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.0:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    path = dtw.dtw(
                        medoid_pattern,
                        random_prototypes[nid],
                        dtw.RETURN_PATH,
                        slope_constraint=slope_constraint,
                        window=window,
                    )
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(
                        np.log(0.5)
                        * dtw_value
                        / dtw_matrix[medoid_id, nearest_order[1]]
                    )
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight

            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            print(
                "There is only one pattern of class %d, skipping pattern average"
                % _labels[i]
            )
            ret[i, :] = x[i]
    return ret


def random_guided_warp(
    x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal"
):
    """Random guided warping based on ref[4]_.

    From ref [3]_:
    "Guided warping [1]_ combines time series by time warping a reference
    pattern by a teacher pattern using Dynamic Time Warping (DTW) [2]_. DTW is a
    method of measuring the distance between time series using elastic element
    matching found by dynamic programming. Guided warping uses the dynamic
    alignment function of DTW to warp the elements of a reference pattern to the
    elements of a teacher pattern. In this way, the reference pattern is set to
    the time steps of the teacher pattern. This is different from averaging in
    that the mixing happens only in the time domain. There are two variants,
    Random Guided Warping (RGW) which uses a random intra-class teacher and
    Discriminative Guided Warping (RGW) which uses a directed discriminative
    teacher [4]_."

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    labels : numpy.ndarray, 2D or 3D
        Either list of integers or one hot encoding of the labels.

    slope_constraint : ["symmetric", "asymmetric"], default="symmetric"
        Slope constraint for DTW.

    use_window : bool, default=True
        If True, use a 10% boundary constraint window for DTW.

    dtw_type : ["normal", "shape"], default="normal"
        Use DTW (`normal`) or shapeDTW (`shape`).

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  K. Kamycki, T. Kapuscinski, M. Oszust, "Data Augmentation with
    Suboptimal Warping for Time-Series Classification," Sensors, vol. 20, no. 1,
    2020. URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/

    .. [2]  H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for
    spoken word recognition," in IEEE Transactions on Acoustics, Speech, and
    Signal Processing, vol. 26, no. 1, pp. 43-49, February 1978, DOI:
    10.1109/TASSP.1978.1163055.

    .. [3]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
    augmentation for time series classification with neural networks," PLOS ONE
    16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841

    .. [4]  B. K. Iwana, S. Uchida, "Time Series Data Augmentation for Neural
    Networks by Time Warping with a Discriminative Teacher," arXiv, 2020. URL:
    https://arxiv.org/abs/2004.08780v1
    """
    if use_window:
        window = np.ceil(x.shape[1] / 10.0).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    _labels = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(_labels[choices] == _labels[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]

            if dtw_type == "shape":
                path = dtw.shape_dtw(
                    random_prototype,
                    pat,
                    dtw.RETURN_PATH,
                    slope_constraint=slope_constraint,
                    window=window,
                )
            else:
                path = dtw.dtw(
                    random_prototype,
                    pat,
                    dtw.RETURN_PATH,
                    slope_constraint=slope_constraint,
                    window=window,
                )

            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(
                    orig_steps,
                    np.linspace(0, x.shape[1] - 1.0, num=warped.shape[0]),
                    warped[:, dim],
                ).T
        else:
            print(
                "There is only one pattern of class %d, skipping timewarping"
                % _labels[i]
            )
            ret[i, :] = pat
    return ret


def discriminative_guided_warp(
    x,
    labels,
    batch_size=6,
    slope_constraint="symmetric",
    use_window=True,
    dtw_type="normal",
    use_variable_slice=True,
):
    """Discriminative guided warping based on ref[4]_.

    From ref [3]_:
    "Guided warping [1]_ combines time series by time warping a reference
    pattern by a teacher pattern using Dynamic Time Warping (DTW) [2]_. DTW is a
    method of measuring the distance between time series using elastic element
    matching found by dynamic programming. Guided warping uses the dynamic
    alignment function of DTW to warp the elements of a reference pattern to the
    elements of a teacher pattern. In this way, the reference pattern is set to
    the time steps of the teacher pattern. This is different from averaging in
    that the mixing happens only in the time domain. There are two variants,
    Random Guided Warping (RGW) which uses a random intra-class teacher and
    Discriminative Guided Warping (RGW) which uses a directed discriminative
    teacher [4]_."

    Parameters
    ----------
    x : numpy.ndarray
        Sequences with shape `(batch, time_steps, n_channels)`.

    labels : numpy.ndarray, 2D or 3D
        Either list of integers or one hot encoding of the labels.

    slope_constraint : ["symmetric", "asymmetric"], default="symmetric"
        Slope constraint for DTW.

    use_window : bool, default=True
        If True, use a 10% boundary constraint window for DTW.

    dtw_type : ["normal", "shape"], default="normal"
        Use DTW (`normal`) or shapeDTW (`shape`).

    Returns
    -------
    numpy.ndarray
        Numpy array of generated data of equal size of the input `x`.

    References
    ----------
    .. [1]  K. Kamycki, T. Kapuscinski, M. Oszust, "Data Augmentation with
    Suboptimal Warping for Time-Series Classification," Sensors, vol. 20, no. 1,
    2020. URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/

    .. [2]  H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for
    spoken word recognition," in IEEE Transactions on Acoustics, Speech, and
    Signal Processing, vol. 26, no. 1, pp. 43-49, February 1978, DOI:
    10.1109/TASSP.1978.1163055.

    .. [3]  Brian Kenji Iwana and Seiichi Uchida, "An empirical survey of data
    augmentation for time series classification with neural networks," PLOS ONE
    16(7): e0254841. DOI: https://doi.org/10.1371/journal.pone.0254841

    .. [4]  B. K. Iwana, S. Uchida, "Time Series Data Augmentation for Neural
    Networks by Time Warping with a Discriminative Teacher," arXiv, 2020. URL:
    https://arxiv.org/abs/2004.08780v1
    """
    if use_window:
        window = np.ceil(x.shape[1] / 10.0).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    _labels = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)

    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)

        # remove ones of different classes
        positive = np.where(_labels[choices] == _labels[i])[0]
        negative = np.where(_labels[choices] != _labels[i])[0]

        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]

            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1.0 / (pos_k - 1.0)) * dtw.shape_dtw(
                                pos_prot,
                                pos_samp,
                                dtw.RETURN_VALUE,
                                slope_constraint=slope_constraint,
                                window=window,
                            )
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1.0 / neg_k) * dtw.shape_dtw(
                            pos_prot,
                            neg_samp,
                            dtw.RETURN_VALUE,
                            slope_constraint=slope_constraint,
                            window=window,
                        )
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(
                    positive_prototypes[selected_id],
                    pat,
                    dtw.RETURN_PATH,
                    slope_constraint=slope_constraint,
                    window=window,
                )
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1.0 / (pos_k - 1.0)) * dtw.dtw(
                                pos_prot,
                                pos_samp,
                                dtw.RETURN_VALUE,
                                slope_constraint=slope_constraint,
                                window=window,
                            )
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1.0 / neg_k) * dtw.dtw(
                            pos_prot,
                            neg_samp,
                            dtw.RETURN_VALUE,
                            slope_constraint=slope_constraint,
                            window=window,
                        )
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(
                    positive_prototypes[selected_id],
                    pat,
                    dtw.RETURN_PATH,
                    slope_constraint=slope_constraint,
                    window=window,
                )

            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(
                orig_steps,
                np.linspace(0, x.shape[1] - 1.0, num=warped.shape[0]),
                path[1],
            )
            warp_amount[i] = np.sum(np.abs(orig_steps - warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(
                    orig_steps,
                    np.linspace(0, x.shape[1] - 1.0, num=warped.shape[0]),
                    warped[:, dim],
                ).T
        else:
            print("There is only one pattern of class %d" % _labels[i])
            ret[i, :] = pat
            warp_amount[i] = 0.0
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(
                    pat[np.newaxis, :, :],
                    reduce_ratio=0.9 + 0.1 * warp_amount[i] / max_warp,
                )[0]
    return ret
