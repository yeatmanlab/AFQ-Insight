import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class AFQFeatureTransformer(object):
    """Transforms AFQ data from an input dataframe into a feature matrix

    Using an object interface for eventual inclusion into sklearn Pipelines
    """
    def __init__(self):
        pass

    def transform(self, df, extrapolate=False):
        """Transforms an AFQ dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            input AFQ dataframe

        extrapolate : boolean
            If True, use column-wise linear interpolation/extrapolation
            for missing metric values. If False, use pandas built-in
            `interpolate` method, which uses interpolation for interior points
            and forward(back)-fill for exterior points.

        Returns
        -------
        X : numpy.ndarray
            feature matrix

        groups : numpy.ndarray
            group membership for each feature (column) of X

        columns : pandas.MultiIndex
            multi-indexed columns of X
        """
        # We'd like to interpolate the missing values, but first we need to
        # structure the data frame so that it does not interpolate from other
        # subjects, tracts, or metrics. It should only interpolate from nearby
        # nodes. So we want the nodeID as the row index and all the other
        # stuff as columns . After that we can interpolate along each column.
        by_node_idx = pd.pivot_table(
            data=df.melt(
                id_vars=['subjectID', 'tractID', 'nodeID'],
                var_name='metric'
            ),
            index='nodeID',
            columns=['metric', 'tractID', 'subjectID'],
            values='value'
        )

        if not extrapolate:
            # We could use the built-in `.interpolate` method. This has some
            # unexpected behavior when the NaN values are at the beginning or
            # end of a series. For NaN values at the end of the series, it
            # forward fills the most recent valid value. And for NaN values
            # at the beginning of the series, it back fills the next valid
            # value. For now, we accept this behavior because the execution
            # time is much, much faster than doing column-wise linear
            # extrapolation
            interpolated = by_node_idx.interpolate(
                method='linear', limit_direction='both'
            )
        else:
            # Instead, we may want to interpolate NaN values with
            # extrapolation at the end of the node range. But, pandas does
            # not currently support extrapolation
            # See this issue:
            # https://github.com/pandas-dev/pandas/issues/16284
            # And this stalled PR:
            # https://github.com/pandas-dev/pandas/pull/16513
            # Until that's fixed, we can perform the interpolation column by
            # column using the apply method. This is SLOW, but it does the job
            def interp_linear_with_extrap(series):
                """Linearly interpolate a series with extrapolation...

                ...outside the series range
                """
                x = series[~series.isnull()].index.values
                y = series[~series.isnull()].values
                f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                return f(series.index)

            # Apply the interpolation across all columns
            interpolated = by_node_idx.apply(interp_linear_with_extrap)

        # Now we have the NaN values filled in, we want to structure the nodes
        # dataframe as a feature matrix with one row per subject and one
        # column for each combination of metric, tractID, and nodeID
        features = interpolated.stack(
            ['subjectID', 'tractID', 'metric']
        ).unstack(
            ['metric', 'tractID', 'nodeID']
        )

        # We're almost there. It'd be nice if the multi-indexed columns were
        # ordered well. So let's reorder the columns
        new_columns = pd.MultiIndex.from_product(
            features.columns.levels,
            names=['metric', 'tractID', 'nodeID']
        )

        features = features.loc[:, new_columns]

        # Construct bundle group membership
        metric_level = features.columns.names.index('metric')
        tract_level = features.columns.names.index('tractID')
        n_tracts = len(features.columns.levels[tract_level])
        bundle_group_membership = np.array(
            features.columns.labels[metric_level] * n_tracts
            + features.columns.labels[tract_level],
            dtype=np.int64
        )

        return features.values, bundle_group_membership, features.columns
