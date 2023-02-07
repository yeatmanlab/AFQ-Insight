"""Utility functions to manipulate tractometry type dataframe
"""

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

BUNDLE_DICT = {
    'CST_R':        'Right Corticospinal',
    'CST_L':        'Left Corticospinal',
    'UNC_R':        'Right Uncinate',
    'UNC_L':        'Left Uncinate',
    'IFO_R':        'Right IFOF',
    'IFO_L':        'Left IFOF',
    'ARC_R':        'Right Arcuate',
    'ARC_L':        'Left Arcuate',
    'ATR_R':        'Right Thalamic Radiation',
    'ATR_L':        'Left Thalamic Radiation',
    'CGC_R':        'Right Cingulum Cingulate',
    'CGC_L':        'Left Cingulum Cingulate',
    'HCC_R':        'Right Cingulum Hippocampus',
    'HCC_L':        'Left Cingulum Hippocampus',
    'FP':           'Callosum Forceps Major',
    'FA':           'Callosum Forceps Minor',
    'ILF_R':        'Right ILF',
    'ILF_L':        'Left ILF',
    'SLF_R':        'Right SLF',
    'SLF_L':        'Left SLF',
    'VOF_R':        'Right Vertical Occipital',
    'VOF_L':        'Left Vertical Occipital',
    'pARC_R':       'Right Posterior Arcuate',
    'pARC_L':       'Left Posterior Arcuate',
    'AntFrontal':   'Callosum: AntFrontal',
    'Motor':        'Callosum: Motor',
    'Occipital':    'Callosum: Occipital',
    'Orbital':      'Callosum: Orbital',
    'PostParietal': 'Callosum: PostParietal',
    'SupFrontal':   'Callosum: SupFrontal',
    'SupParietal':  'Callosum: SupParietal',
    'Temporal':     'Callosum: Temporal',
}


def merge_profiles_varinterest(
    tracto_df:pd.DataFrame,
    var_df:pd.DataFrame,
    show_info=True,
) -> pd.DataFrame:
    """Merge tractometry and variable of interest dataframes

    Parameters
    ----------
    tracto_df : pd.DataFrame
        tractometry dataframe
    var_df : pd.DataFrame
        dataframe containing variables of interest
    show_info : bool, optional
        Whether to show what's beeing dropped with merge, by default True

    Returns
    -------
    pd.DataFrame
        Merged dataframe
    """
    
    var_df = optimize(var_df)
    tracto_df = optimize(tracto_df)
    
    df = pd.merge(
        left=tracto_df,
        right=var_df,
        on=['subjectID','sessionID'],
        how="outer")
    #TODO: show how many subjects we lose here
    if show_info:
        logger.info("Not implemented yet")
    df = df.dropna().reset_index(drop=True)
    return df



def center_cut(df: pd.DataFrame, cut: tuple=(25,75)) -> pd.DataFrame:
    """ Returns dataframe where the nodeID is cut between two indicated values.

    Args:
        df (pd.DataFrame): dataframe with column `nodeID` to be filtered on.
        cut (tuple, optional): extreme values to filter nodeID on. Defaults to (25,75).

    Returns:
        pd.DataFrame: filtered dataframe
    """

    df = df[(df.nodeID >= cut[0]) & (df.nodeID <= cut[1])]
    return df.reset_index(drop=True)



def optimize(df:pd.DataFrame) -> pd.DataFrame:
    cols = list(set(['subjectID','sessionID','tractID']) & set(df.columns))
    for col in cols:
        df[col] = df[col].astype('category').cat.remove_unused_categories()
    if 'nodeID' in df.columns:
        df['nodeID'] = df['nodeID'].astype(np.int16)
    return df


def beautify(tracto: pd.DataFrame) -> pd.DataFrame:
    """Makes tracto_df ready for plotting

    Args:
        tracto (pd.DataFrame): tractometry dataframe.

    Returns:
        pd.DataFrame: tractometry dataframe, with embelishments for 
    """
    
    for col in tracto.columns:
        if tracto[col].dtype == 'category':
            tracto[col] = tracto[col].cat.remove_unused_categories()
    if "tractID_b" not in tracto.columns:
        tracto = tracto.assign(tractID_b=tracto.tractID.map(BUNDLE_DICT))
    return tracto


def tracto_subsample(
    tracto_df:pd.DataFrame,
    subjects=5,
    sessions=1,
    tracts=2,
    ) -> pd.DataFrame:

    if isinstance(subjects, int):
        subjects = np.random.choice(
            tracto_df.subjectID.unique(), subjects, replace=False)
    if isinstance(sessions, int):
        sessions = np.random.choice(
            tracto_df.sessionID.unique(), sessions, replace=False)
    if isinstance(tracts, int):
        tracts = np.random.choice(
            tracto_df.tractID.unique(), tracts, replace=False)
    
    df = tracto_df[
        (tracto_df.subjectID.isin(subjects)) &
        (tracto_df.sessionID.isin(sessions)) &
        (tracto_df.tractID.isin(tracts))
        ]
    for col in df.columns:
        if df[col].dtype == 'category':
            df[col] = df[col].cat.remove_unused_categories()
    return df
    
