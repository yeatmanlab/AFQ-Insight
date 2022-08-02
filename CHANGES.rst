v0.3.5 (August 02, 2022)
========================
  * Use the new from_dataset method to initialize objects in examples.  (#123)
  * Add HBN (#121)

v0.3.4 (August 02, 2022)
========================
  * Add Python 3.10 to CI workflows (#122)
  * Add fit, fit_transform, transform, predict, and score methods (#119)
  * NF: Allow passing an AFQDataset object as input. (#113)
  * Add 1D cnn models and data augmentation (#112)
  *  (#107)
  * Remove groupyr ref from user guide (#110)
  * Remove groupyr references from contribution guide (#109)

v0.3.3 (February 16, 2022)
==========================
  * Fix attribute errors in bagging and cross-validation. Loosen dependency reqs (#108)
  * Make AFQDataset work with different input formats and make indexable (#105)

v0.3.2 (December 22, 2021)
==========================
  * ENH: Expose figsize and tight_layout kwargs to plot_tract_profiles (#104)
  * Expose legend kwargs to user in plot_tract_profiles (#102)

v0.3.1 (December 07, 2021)
==========================
  * BF: Fix label encoding in the presence of NaN labels (#101)
  * Add enforce_sub_prefix option to AFQDataset and load_afq_data (#96)
  * RF: For plotting, sometimes you don't want to group by any other variable (#99)
  * BF: Cast integer session identifiers to strings. (#91)
  * Add an AFQDataset class (#89)
  * Add callosal bundles to plot_tract_profiles (#82)
  * ENH: Add Convolutional Neural Network Models (#83)
  * TST: Improve coverage. (#86)
  * Fixes tests broken by previous commits. (#85)
  * DOC + RF: Adds docstrings to class definitions. (#84)
  * ENH: Allow bundle aggregation and put load_afq_data output into a named_tuple (#80)
  * ENH: Allow bundle aggregation and put load_afq_data output into a named_tuple (#80)
  * DOC: Make examples a bit more explicit (#79)

v0.3.0 (June 28, 2021)
======================
  * Add research examples to documentation (#77)
  * ENH: Add dataset fetchers (#76)
  * Adds error handling for cases where users forget to set target_cols. (#73)
  * ENH: Add unsupervised boolean parameter to load_afq_data (#71)
  * ENH: Add bundle profile plotting function (#69)
  * Remove transformers that were migrated to groupyr (#68)
  * DEP: Loosen groupyr dependency (#67)
  * DEP: Loosen dependency requirements (#66)
  * Fixes broken links (#65)
  * ENH: Stratify bootstrap replicants in SerialBaggingClassifier (#61)
  * ENH: Add SerialBagging classes (#60)

v0.2.8 (December 10, 2020)
==========================
  * DEP: Bump groupyr version

v0.2.7 (December 09, 2020)
==========================
  * ENH: Add ensemble estimators to pipelines (#59)

v0.2.6 (October 28, 2020)
=========================
  * FIX: Add option to serialize cv execution (#57)


v0.2.5 (October 26, 2020)
=========================
  * ENH: Use joblib.Parallel instead of custom groupyr._ProgressParallel (#56)


v0.2.4 (October 26, 2020)
=========================
  * ENH: Add functions and test for outer cross-validation loop (#55)


v0.2.3 (October 22, 2020)
=========================
  * FIX: Use target transformer only if requested by user (#54)


v0.2.2 (October 15, 2020)
=========================
  * ENH: Bump groupyr version (#53)
  * ENH: Use sklearn LabelEncoder in load_afq_data and return classes (#52)
  * ENH: Increase flexibility of load_afq_data (#51)


v0.2.1 (October 14, 2020)
=========================

* Major refactoring after moving the sparse group lasso estimator functions into a new library (groupyr) and retaining only AFQ specific dataset transformers and pipelines in AFQ-Insight


v0.2.0 (October 14, 2020)
=========================

* Major refactoring after moving the sparse group lasso estimator functions into a new library (groupyr) and retaining only AFQ specific dataset transformers and pipelines in AFQ-Insight
