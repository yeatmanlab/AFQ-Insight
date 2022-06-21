"""
============================================================
Load HBN data and plot average bundle profiles for each site
============================================================

This example loads AFQ data from the Healthy Brain Network (HBN) preprocessed
diffusion derivatives [1]_. The HBN is a landmark...

We first load the data by using the :func:`AFQDataset.from_files` static method
and supplying AWS S3 URIs instead of local file names. We then impute missing
values and plot the mean bundle profiles by scanning site, noting that there are
substantial site differences. Lastly, we harmonize the site differences using
NeuroComBat [2]_ and plot the harmonized bundle profiles to verify that the site
differences have been removed.

.. [1]  Adam Richie-Halford, Matthew Cieslak, Lei Ai, Sendy Caffarra, Sydney
   Covitz, Alexandre R. Franco, Iliana I. Karipidis, John Kruper, Michael
   Milham, Bárbara Avelar-Pereira, Ethan Roy, Valerie J. Sydnor, Jason Yeatman,
   The Fibr Community Science Consortium, Theodore D. Satterthwaite, and Ariel
   Rokem,
   "An open, analysis-ready, and quality controlled resource for pediatric brain
   white-matter research"
   bioRxiv 2022.02.24.481303;
   doi: https://doi.org/10.1101/2022.02.24.481303

.. [2] Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A
   Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur,
   Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara.
   "Harmonization Of Multi-Site Diffusion Tensor Imaging Data"
   NeuroImage, 161, 149-170, 2017;
   doi: https://doi.org/10.1016/j.neuroimage.2017.08.047

"""
import numpy as np

from afqinsight import AFQDataset
from afqinsight.plot import plot_tract_profiles
from neurocombat_sklearn import CombatModel
from sklearn.impute import SimpleImputer

##########################################################################
# Fetch the HBN data
# ------------------
#
# The :method:`AFQDataset.from_files` static method expects a path to
# nodes.csv and subjects.csv files, but these file paths can be remote
# URLs or AWS S3 URIs. We'll use S3 URIs to grab the HBN data. After dropping
# participants with null phenotypic values, it has 1,867 participants.

dataset = AFQDataset.from_files(
    fn_nodes="s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/afq/combined_tract_profiles.csv",
    fn_subjects="s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
    dwi_metrics=["dki_fa", "dki_md"],
    target_cols=["age", "sex", "scan_site_id"],
    label_encode_cols=["sex", "scan_site_id"],
    index_col="subject_id",
)
dataset.drop_target_na()
print(dataset)

##########################################################################
# Impute missing values
# ---------------------
#
# Next we impute missing values using median imputation. There's no model
# evaluation going on here. We're just making pretty pictures so we're
# going to skip the usual train/test split.

dataset = dataset.model_fit_transform(SimpleImputer(strategy="median"))

##########################################################################
# Plot average bundle profiles by scan site
# -----------------------------------------
#
# Next we plot the mean bundle profiles by scanning site. The :func:`plot_tract_profiles`
# function takes as input an :class:`AFQDataset` and returns matplotlib figures
# displaying the mean bundle profile for each bundle and metric, optionally
# grouped by a categorical or continuous variable.

site_figs = plot_tract_profiles(
    X=dataset,
    group_by=dataset.classes["scan_site_id"][dataset.y[:, 2].astype(int)],
    group_by_name="Site",
    figsize=(14, 14),
)

##########################################################################
# Harmonize the sites and replot
# ------------------------------
#
# We can see that there are substantial scan site differences in both the
# FA and MD profiles. Let's use neuroComBat to harmonize the site differences
# and then replot the mean bundle profiles.
#
# N.B. We use the excellent `neurocombat_sklearn
# <https://github.com/Warvito/neurocombat_sklearn>`_ package to apply ComBat to
# our data. We love this library, however it is not fully compliant with the
# scikit-learn transformer API, so we cannot use the
# :func:`AFQDataset.model_fit_transform` method to apply this transformer to our
# dataset. No problem! We can simply copy the unharmonized dataset into a new
# variable and then overwrite the features of the new dataset with the ComBat
# output.
#
# Lastly, we replot the mean bundle profiles and confirm that ComBat did its
# job.

dataset_harmonized = dataset.copy()
dataset_harmonized.X = CombatModel().fit_transform(
    dataset.X,
    dataset.y[:, 2][:, np.newaxis],
    dataset.y[:, 1][:, np.newaxis],
    dataset.y[:, 0][:, np.newaxis],
)

site_figs = plot_tract_profiles(
    X=dataset_harmonized,
    group_by=dataset_harmonized.classes["scan_site_id"][
        dataset_harmonized.y[:, 2].astype(int)
    ],
    group_by_name="Site",
    figsize=(14, 14),
)
