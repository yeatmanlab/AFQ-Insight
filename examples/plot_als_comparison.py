"""
=================================================
Use parametric statistics for group comparison
=================================================

As a contrast to the approach presented in other examples, we also present a
more "standard" statistical approach to analyze tract profile data. Here, we
will conduct a point-by-point group comparison using a linear model fit using
the standard ordinary least squares (OLS) method.

This example first fetches the ALS classification dataset from Sarica et al
[1]_. This dataset contains tractometry features from 24 patients with ALS and
24 demographically matched control subjects. It then uses the statsmodels
library to compare between the tract profiles of the two groups in one
tract (the left corticospinal tract) and in one feature (FA).

.. [1]  Alessia Sarica, et al. "The Corticospinal Tract Profile in
   AmyotrophicLateral Sclerosis" Human Brain Mapping, vol. 38, pp. 727-739, 2017
   DOI: 10.1002/hbm.23412

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afqinsight import AFQDataset
from statsmodels.api import OLS
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

#############################################################################
# Fetch data from Sarica et al.
# -----------------------------
# As a shortcut, we have incorporated a few studies into the software. In these
# cases, a :class:`AFQDataset` class instance can be initialized using the
# :func:`AFQDataset.from_study` static method. This expects the name of one of
# the studies that are supported (see the method documentation for the list of
# these studies). By passing `"sarica"`, we request that the software download
# the data from this study and initialize an object for us from this data.

afqdata = AFQDataset.from_study("sarica")

# Examine the data
# ----------------
# ``afqdata`` is an ``AFQDataset`` object, with properties corresponding to the
# tractometry features and phenotypic targets. In this case, y includes only the
# group to which the subjects belong, encoded as 0 (patients) or 1 (controls).

X = afqdata.X
y = afqdata.y
groups = afqdata.groups
feature_names = afqdata.feature_names
group_names = afqdata.group_names
subjects = afqdata.subjects


# Filtering the data
# ----------------
# We start by filtering the data down to only the FA estimates in the left
# corticospinal tract. We can do that by creating a Pandas dataframe and then
# using the column labels that `afqdataset` has generated from the data.

data = pd.DataFrame(columns=afqdata.feature_names, data=X)
lcst_fa = data.filter(like="Left Corticospinal").filter(like="fa")

pvals = np.zeros(lcst_fa.shape[-1])
coefs = np.zeros(lcst_fa.shape[-1])
cis = np.zeros((lcst_fa.shape[-1], 2))

# Fit models
# ----------------
# Next, we fit a model for each node of the CST for the FA as a function of
# group. The initializer `OLS.from_formula` takes
# `R-style formulas <https://www.statsmodels.org/dev/example_formulas.html>`_
# as its model specification. Here, we are using the `"group"` variable as a
# categorical variable in the model. Much more complex linear models (including
# linear mixed effects model) are possible for datasets with more complex
# phenotypical data.

for ii, column in enumerate(lcst_fa.columns):
    feature, node, tract = column
    this = pd.DataFrame({"group": y, feature: lcst_fa[column]})
    model = OLS.from_formula(f"{feature} ~ C(group)", this)
    fit = model.fit()
    coefs[ii] = fit.params.loc["C(group)[T.1]"]
    cis[ii] = fit.conf_int(alpha=0.05).loc["C(group)[T.1]"].values
    pvals[ii] = anova_lm(fit, typ=2)["PR(>F)"][0]

# Correct for multiple comparison
# --------------------------------
# Because we conducted 100 comparisons, we need to correct the p-values that
# we obtained for the potential for a false discovery. There are multiple
# ways to conduct multuple comparison correction, and we will not get into
# the considerations in selecting a method here. For the example, we chose the
# Benjamini/Hochberg FDR controlling method. This returns a boolean array for
# the p-values that are rejected at a specified alpha level (after correction),
# as well as an array of the corrected p-values.

reject, pval_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
reject_idx = np.where(reject)

# Visualize
# ----------
# Using Matplotlib, we can visualize the results. The top figure shows the tract
# profiles of the two groups, with stars indicating the nodes at which the null
# hypothesis is rejected. The bottom figure shows the model coefficients
# together with their estimated 95% confidence interval.

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.mean(lcst_fa.iloc[afqdata.y == 0], 0).values)
ax[0].plot(np.mean(lcst_fa.iloc[afqdata.y == 1], 0).values)
ax[0].plot(reject_idx, np.zeros(len(reject_idx)), "k*")

ax[1].plot(coefs)
ax[1].fill_between(range(100), cis[:, 0], cis[:, 1], alpha=0.5)
ax[1].plot(range(100), np.zeros(100), "k--")
