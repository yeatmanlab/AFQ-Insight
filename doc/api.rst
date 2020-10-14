#############
API Reference
#############

.. currentmodule:: afqinsight

Pipelines
=========

These are AFQ-Insights recommended estimator pipelines. 

.. autoclass:: AFQRegressorPipeline

.. autoclass:: AFQClassifierPipeline

Transformers
============

These transformers transform tractometry information from the AFQ standard
data format to feature matrices that are ready for ingestion into
sklearn-compatible pipelines.

.. autoclass:: AFQDataFrameMapper
