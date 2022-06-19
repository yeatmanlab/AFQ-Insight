#############
API Reference
#############

.. currentmodule:: afqinsight

Datasets
========

This class encapsulates an AFQ dataset and has static methods to read data from csv files
conforming to the AFQ data standard.

.. autoclass:: AFQDataset

Pipelines
=========

These are AFQ-Insights recommended estimator pipelines. 

.. autofunction:: make_afq_regressor_pipeline

.. autofunction:: make_afq_classifier_pipeline

Transformers
============

These transformers transform tractometry information from the AFQ standard
data format to feature matrices that are ready for ingestion into
sklearn-compatible pipelines.

.. autoclass:: AFQDataFrameMapper

Cross Validation
================

This function validates model performance using cross-validation, while
checkpointing the estimators and scores.

.. autofunction:: cross_validate_checkpoint
