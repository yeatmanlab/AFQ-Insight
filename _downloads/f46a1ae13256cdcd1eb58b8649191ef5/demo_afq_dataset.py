"""
=====================================
Load and interact with an AFQ dataset
=====================================

This example loads AFQ data from CSV files and manipulates that data using
scikit-learn transformers and estimators. First we fetch the Weston-Havens
dataset described in Yeatman et al [1]_. This dataset contains tractometry
features from 77 subjects ages 6-50.

Next, we split the dataset into a train and test split, impute missing values,
and fit a LASSO model, all using :class:`AFQDataset` methods. Predictive
performance for the default LASSO model is abysmal. It is only used here to
demonstrate the use of scikit-learn estimators. In a research setting, one might
use more advanced estimators, such as the SGL [2]_, a gradient boosting machine,
or a neural network.

Finally, we convert the AFQDataset to a tensorflow dataset and fit a basic
one-dimensional CNN to predict age from the features. This last step requires that
AFQ-Insight has been installed with::

    pip install afqinsight[tf]

or that tensorflow has been separately installed with::

    pip install tensorflow

.. [1]  Jason D. Yeatman, Brian A. Wandell, & Aviv A. Mezer, "Lifespan
    maturation and degeneration of human brain white matter" Nature
    Communications, vol. 5:1, pp. 4932, 2014 DOI: 10.1038/ncomms5932

.. [2]  Adam Richie-Halford, Jason Yeatman, Noah Simon, and Ariel Rokem
   "Multidimensional analysis and detection of informative features in human
   brain white matter" PLOS Computational Biology, 2021 DOI:
   10.1371/journal.pcbi.1009136

"""
import afqinsight.nn.tf_models as nn
import os.path as op
import tensorflow as tf

from afqinsight.datasets import download_weston_havens
from afqinsight import AFQDataset

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

##########################################################################
# Fetch example data
# ------------------
#
# The :func:`download_weston_havens` function download the data used in this
# example and places it in the `~/.cache/afq-insight/weston_havens` directory.
# If the directory does not exist, it is created. The data follows the format
# expected by the :func:`load_afq_data` function: a file called `nodes.csv` that
# contains AFQ tract profiles and a file called `subjects.csv` that contains
# information about the subjects. The two files are linked through the
# `subjectID` column that should exist in both of them. For more information
# about this format, see also the `AFQ-Browser documentation
# <https://yeatmanlab.github.io/AFQ-Browser/dataformat.html>`_ (items 2 and 3).

workdir = download_weston_havens()

##########################################################################
# Read in the data
# ----------------
#
# Next, we read in the data. The :func:`AFQDataset.from_files` static method
# expects a the filenames of a nodes.csv and subjects.csv file, and returns a
# dataset object.

dataset = AFQDataset.from_files(
    fn_nodes=op.join(workdir, "nodes.csv"),
    fn_subjects=op.join(workdir, "subjects.csv"),
    dwi_metrics=["md", "fa"],
    target_cols=["Age"],
)

##########################################################################
# Train / test split
# ------------------
#
# We can use the dataset in the :func:`train_test_split` function just as we
# would with an array.

dataset_train, dataset_test = train_test_split(dataset, test_size=1 / 3)

##########################################################################
# Impute missing values
# ---------------------
#
# Next we train an imputer on the training set and use it to transform the
# features in both the training and the test set.

imputer = dataset_train.model_fit(SimpleImputer(strategy="median"))
dataset_train = dataset_train.model_transform(imputer)
dataset_test = dataset_test.model_transform(imputer)

##########################################################################
# Fit a LASSO model
# -----------------
#
# Next we fit a LASSO estimator to the training data and print the score of that
# model on the test dataset.

estimator = dataset_train.model_fit(Lasso())
y_pred = dataset_test.model_predict(estimator)
train_score = dataset_train.model_score(estimator)
test_score = dataset_test.model_score(estimator)
print("LASSO train score:", train_score)
print("LASSO test score: ", test_score)

##########################################################################
# Convert to tensorflow datasets
# ------------------------------
#
# Next we convert the train and test datasets to tensorflow datasets
# and use one of AFQ-Insight's built-in one-dimensional CNNs to predict
# age. This part of the example will only work if you have either installed
# AFQ-Insight with tensorflow using::
#
#     pip install afqinsight[tf]
#
# or separately install tensorflow using::
#
#     pip install tensorflow
#
# This model also performs poorly. It turns out predicting age in
# this dataset requires a bit more work.

tfset_train = dataset_train.as_tensorflow_dataset()
tfset_test = dataset_test.as_tensorflow_dataset()

batch_size = 2
tfset_train = tfset_train.batch(8)
tfset_test = tfset_test.batch(8)

print("CNN Architecture")
model = nn.cnn_lenet(
    input_shape=(100, 40), output_activation=None, n_classes=1, verbose=True
)

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mean_squared_error"],
)

model.fit(tfset_train, epochs=500, validation_data=tfset_test, verbose=0)

print()
print("CNN R^2 score: ", r2_score(dataset_test.y, model.predict(tfset_test)))
