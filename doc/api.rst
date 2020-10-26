#############
API Reference
#############

.. currentmodule:: afqinsight

Pipelines
=========

These are AFQ-Insights recommended estimator pipelines. 

.. autofunction:: make_afq_regressor_pipeline

.. autofunction:: make_afq_classifier_pipeline

Cross Validation
================

This function validates model performance using cross-validation, while
checkpointing the estimators and scores.

.. autofunction:: cross_validate_checkpoint

Dataset Loader
==============

This function reads data from csv files conforming to the AFQ data standard
and return feature and target matrices, grouping arrays, and subject IDs.

.. autofunction:: load_afq_data

Transformers
============

These transformers transform tractometry information from the AFQ standard
data format to feature matrices that are ready for ingestion into
sklearn-compatible pipelines.

.. autoclass:: AFQDataFrameMapper
