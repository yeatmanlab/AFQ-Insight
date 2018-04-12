from __future__ import absolute_import, division, print_function

import ksgl
import numpy as np
import pandas as pd
from collections import namedtuple
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, \
    StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler

from .transform import AFQFeatureTransformer

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


# fn_nodes = '../AFQ-Insight/afqinsight/data/nodes.csv'
# fn_subjects = '../AFQ-Insight/afqinsight/data/subjects.csv'
# target_col = 'class'
# binary_positive = 'ALS'
# scale_x = True
#
#
# nodes = pd.read_csv(fn_nodes)
# targets = pd.read_csv(fn_subjects, index_col='subjectID').drop(['Unnamed: 0'], axis='columns')
#
# y = targets[target_col]
# y = y.map(lambda c: int(c == binary_positive)).values
#
# transformer = AFQFeatureTransformer()
# x, groups, cols = transformer.transform(nodes)
#
# if scale_x:
#     scaler = StandardScaler()
#     x = scaler.fit_transform(x)
#
# progress = TQDMNotebookCallback(leave_inner=False)


@registered
def outer_cv_classify(x, y, groups, hidden_layers=None, n_classes=2,
                      n_splits_outer=10, test_size=0.15,
                      n_splits_inner=3, n_repeats_inner=2,
                      save_models=False, save_results=False,
                      validation_split=0.2):
    outer_cv = StratifiedShuffleSplit(
        n_splits=n_splits_outer, test_size=test_size
    )

    cv_results = []

    for split_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        x_train = x[train_idx, :]
        y_train = y[train_idx]
        x_test = x[test_idx, :]
        y_test = y[test_idx]

        cv_results.append(inner_cv_classify(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            groups=groups, hidden_layers=hidden_layers, n_classes=n_classes,
            n_splits=n_splits_inner, n_repeats=n_repeats_inner,
            save_model=save_models, save_results=save_results,
            save_suffix=split_idx, validation_split=validation_split
        ))

    return cv_results


@registered
def inner_cv_classify(x_train, y_train, x_test, y_test,
                      groups, hidden_layers=None, n_classes=2,
                      n_splits=3, n_repeats=2,
                      save_model=False, save_results=False,
                      save_suffix=None, validation_split=0.2):
    if (save_model or save_results) and save_suffix is None:
        raise ValueError('if save_best_model is True, you '
                         'must supply a model_save_suffix.')

    if n_classes != 2:
        raise ValueError('n_classes > 2 is not yet supported.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('training x and y do not have same length.')

    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError('training x and y do not have same length.')

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError('training and test x arrays have inconsistent '
                         'number of features.')

    # Define the named tuple for the return type
    CVResults = namedtuple(
        'CVResults',
        ['grid_results_df', 'best_params', 'best_model',
         'best_beta_hat', 'best_scores']
    )

    d = x_train.shape[1]

    def create_classification_model(alpha=0.1, lambda_=0.1, n_epochs=1000):
        if hidden_layers is None or len(list(hidden_layers)) == 0:
            clf = ksgl.SGLClassifier(
                dim_input=d, n_classes=n_classes,
                groups=groups, alpha=alpha, lambda_=lambda_, n_epochs=n_epochs,
                optimizer='adam', lr=0.001, validation_split=validation_split,
                early_stopping_patience=0, verbose=False,
            )
        else:
            clf = ksgl.SGLMultiLayerPerceptronClassifier(
                dim_input=d, n_classes=n_classes, hidden_layers=hidden_layers,
                groups=groups, alpha=alpha, lambda_=lambda_, n_epochs=n_epochs,
                optimizer='adam', lr=0.001, validation_split=validation_split,
                early_stopping_patience=0, verbose=False,
            )

        return clf.model

    inner_cv_generator = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats
    )

    model = KerasClassifier(
        build_fn=create_classification_model,
        verbose=0
    )

    if validation_split == 0.0:
        monitor = EarlyStopping(
            monitor='loss', min_delta=1e-3, patience=5, verbose=0, mode='auto'
        )
    else:
        monitor = EarlyStopping(
            monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto'
        )

    # define the grid search parameters
    batch_size = [32]
    alphas = np.array([0.05, 0.5, 0.95])
    lambdas = np.logspace(-4, 4, 20)
    callbacks = [[monitor]]
    validation_splits = [validation_split]
    param_grid = dict(
        callbacks=callbacks,
        validation_split=validation_splits,
        batch_size=batch_size,
        alpha=alphas,
        lambda_=lambdas
    )

    scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy'}

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv_generator,
        scoring=scoring,
        refit='AUC',
        n_jobs=1,
        verbose=1
    )

    grid.fit(x_train, y_train)

    cv_df = pd.DataFrame(grid.cv_results_)

    cv_df.drop(
        ['param_callbacks', 'param_validation_split'],
        axis='columns', inplace=True
    )

    best_model = grid.best_estimator_.model

    if save_results:
        # Save grid results DataFrame to disk
        cv_df.to_pickle('grid_results_{s:s}.pkl'.format(s=save_suffix))

    if save_model:
        # serialize model to JSON
        model_json = best_model.to_json()
        with open('model_{s:s}.json'.format(s=save_suffix), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        best_model.save_weights('model_weights_{s:s}.hdf5'.format(s=save_suffix))

    beta_hat = best_model.get_weights()[0]

    if n_classes == 2:
        y_pred_train = best_model.predict(x_train) > 0.5
        y_pred_test = best_model.predict(x_test) > 0.5
    else:
        raise ValueError('n_classes > 2 is not yet supported.')

    scores = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_AUC': roc_auc_score(y_train, y_pred_train),
        'test_AUC': roc_auc_score(y_test, y_pred_test),
    }

    return CVResults(
        grid_results_df=cv_df,
        best_params=grid.best_params_,
        best_model=best_model,
        best_beta_hat=beta_hat,
        best_scores=scores,
    )


@registered
def outer_cv_regressor(x, y, groups, hidden_layers=None,
                      n_splits_outer=10, test_size=0.15,
                      n_splits_inner=3, n_repeats_inner=2,
                      save_models=False, save_results=False,
                      validation_split=0.2):
    outer_cv = ShuffleSplit(
        n_splits=n_splits_outer, test_size=test_size
    )

    cv_results = []

    for split_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        x_train = x[train_idx, :]
        y_train = y[train_idx]
        x_test = x[test_idx, :]
        y_test = y[test_idx]

        cv_results.append(inner_cv_classify(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            groups=groups, hidden_layers=hidden_layers, n_classes=n_classes,
            n_splits=n_splits_inner, n_repeats=n_repeats_inner,
            save_model=save_models, save_results=save_results,
            save_suffix=split_idx, validation_split=validation_split
        ))

    return cv_results


@registered
def inner_cv_classify(x_train, y_train, x_test, y_test,
                      groups, hidden_layers=None, n_classes=2,
                      n_splits=3, n_repeats=2,
                      save_model=False, save_results=False,
                      save_suffix=None, validation_split=0.2):
    if (save_model or save_results) and save_suffix is None:
        raise ValueError('if save_best_model is True, you '
                         'must supply a model_save_suffix.')

    if n_classes != 2:
        raise ValueError('n_classes > 2 is not yet supported.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('training x and y do not have same length.')

    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError('training x and y do not have same length.')

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError('training and test x arrays have inconsistent '
                         'number of features.')

    # Define the named tuple for the return type
    CVResults = namedtuple(
        'CVResults',
        ['grid_results_df', 'best_params', 'best_model',
         'best_beta_hat', 'best_scores']
    )

    d = x_train.shape[1]

    def create_classification_model(alpha=0.1, lambda_=0.1, n_epochs=1000):
        if hidden_layers is None or len(list(hidden_layers)) == 0:
            clf = ksgl.SGLClassifier(
                dim_input=d, n_classes=n_classes,
                groups=groups, alpha=alpha, lambda_=lambda_, n_epochs=n_epochs,
                optimizer='adam', lr=0.001, validation_split=validation_split,
                early_stopping_patience=0, verbose=False,
            )
        else:
            clf = ksgl.SGLMultiLayerPerceptronClassifier(
                dim_input=d, n_classes=n_classes, hidden_layers=hidden_layers,
                groups=groups, alpha=alpha, lambda_=lambda_, n_epochs=n_epochs,
                optimizer='adam', lr=0.001, validation_split=validation_split,
                early_stopping_patience=0, verbose=False,
            )

        return clf.model

    inner_cv_generator = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats
    )

    model = KerasClassifier(
        build_fn=create_classification_model,
        verbose=0
    )

    if validation_split == 0.0:
        monitor = EarlyStopping(
            monitor='loss', min_delta=1e-3, patience=5, verbose=0, mode='auto'
        )
    else:
        monitor = EarlyStopping(
            monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto'
        )

    # define the grid search parameters
    batch_size = [32]
    alphas = np.array([0.05, 0.5, 0.95])
    lambdas = np.logspace(-4, 4, 20)
    callbacks = [[monitor]]
    validation_splits = [validation_split]
    param_grid = dict(
        callbacks=callbacks,
        validation_split=validation_splits,
        batch_size=batch_size,
        alpha=alphas,
        lambda_=lambdas
    )

    scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy'}

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv_generator,
        scoring=scoring,
        refit='AUC',
        n_jobs=1,
        verbose=1
    )

    grid.fit(x_train, y_train)

    cv_df = pd.DataFrame(grid.cv_results_)

    cv_df.drop(
        ['param_callbacks', 'param_validation_split'],
        axis='columns', inplace=True
    )

    best_model = grid.best_estimator_.model

    if save_results:
        # Save grid results DataFrame to disk
        cv_df.to_pickle('grid_results_{s:s}.pkl'.format(s=save_suffix))

    if save_model:
        # serialize model to JSON
        model_json = best_model.to_json()
        with open('model_{s:s}.json'.format(s=save_suffix), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        best_model.save_weights('model_weights_{s:s}.hdf5'.format(s=save_suffix))

    beta_hat = best_model.get_weights()[0]

    if n_classes == 2:
        y_pred_train = best_model.predict(x_train) > 0.5
        y_pred_test = best_model.predict(x_test) > 0.5
    else:
        raise ValueError('n_classes > 2 is not yet supported.')

    scores = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_AUC': roc_auc_score(y_train, y_pred_train),
        'test_AUC': roc_auc_score(y_test, y_pred_test),
    }

    return CVResults(
        grid_results_df=cv_df,
        best_params=grid.best_params_,
        best_model=best_model,
        best_beta_hat=beta_hat,
        best_scores=scores,
    )
