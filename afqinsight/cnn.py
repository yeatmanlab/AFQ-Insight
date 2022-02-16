"""Build, fit, and predict with 1-D convolutional neural networks."""

import functools
import numpy as np
import os.path as op
import tempfile

from dipy.utils.optpkg import optional_package
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_is_fitted

keras_msg = (
    "To use afqinsight's convolutional neural nets for tractometry data, you will need "
    "to have tensorflow and kerastuner installed. You can do this by installing "
    "afqinsight with `pip install afqinsight[tf]`, or by separately installing these packages "
    "with `pip install tensorflow keras-tuner`."
)

kt, _, _ = optional_package("keras_tuner", keras_msg)
tf, has_tf, _ = optional_package("tensorflow", keras_msg)

if has_tf:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint


def build_model(hp, conv_layers, input_shape):
    """Build a keras model.

    Uses keras tuner to build model - can control # layers, # filters in each layer, kernel size,
    regularization etc

    Parameters
    ----------
    hp : tensorflow.keras.HyperParameters()
            Hyperparameters class from which to sample hyperparameters

    conv_layers : int
            number of layers (one layer is Conv and MaxPool) in the sequential model.

    input_shape : int
            input shape of X so the model gets built continuously as you are adding layers

    Returns
    -------
    model : tensorflow.keras.Model
            compiled model that uses hyperparameters defined inline to hypertune the model

    """
    model = Sequential()
    model.add(
        Conv1D(
            filters=hp.Int("init_conv_filters", min_value=32, max_value=512, step=32),
            kernel_size=hp.Int("init_conv_kernel", min_value=1, max_value=4, step=1),
            activation="relu",
            input_shape=input_shape,
        )
    )

    for i in range(conv_layers - 1):
        model.add(
            Conv1D(
                filters=hp.Int(
                    "conv_filters" + str(i), min_value=32, max_value=512, step=32
                ),
                kernel_size=hp.Int(
                    "conv_kernel" + str(i), min_value=1, max_value=4, step=1
                ),
                activation="relu",
            )
        )

        model.add(MaxPool1D(pool_size=2, padding="same"))

    model.add(Dropout(0.25))
    model.add(Flatten())

    dense_filters_2 = hp.Int("dense_filters_2", min_value=32, max_value=512, step=32)
    model.add(Dense(dense_filters_2, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(
        loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"]
    )

    return model


class ModelBuilder:
    """Build a complex model architecture with the specified number of layers.

    Parameters
    ----------
    tuner_type : str or class.
        Tuner to use. One of {"hyperband", "bayesian", "random"}.

    input_shape : tuple
        Expected shape of the input data.

    layers : int
        Number of layers in the model.

    max_epochs : int
        Number of epochs to train the model.

    X_test : numpy.ndarray
        Test data.

    y_test : numpy.ndarray
        Test labels or test values.

    batch_size : int
        Batch size to use when training.

    directory : str
        Directory to save the model to.

    project_name : str, optional
        A string, the name to use as prefix for files saved by the tuner object. Defaults to None

    tuner_kwargs : dict, optional
        Keyword arguments to pass to the tuner class on initialization.
        Defaults to tuner defaults.
    """

    def __init__(
        self,
        tuner_type,
        input_shape,
        layers,
        max_epochs,
        X_test,
        y_test,
        batch_size,
        directory=None,
        project_name=None,
        **tuner_kwargs,
    ):

        self.tuner_type = tuner_type
        self.layers = layers
        self.input_shape = input_shape
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.X_test = X_test
        self.y_test = y_test
        self.directory = directory
        self.project_name = project_name
        self.tuner_kwargs = tuner_kwargs

    def _get_tuner(self):
        """Call build_model and instantiate a Keras Tuner for the returned model depending on user choice of tuner.

        Returns
        -------
        tuner : kerastuner.tuners
                BayesianOptimization, Hyperband, or RandomSearch tuner

        """
        # setting parameters beforehand
        hypermodel = functools.partial(
            build_model, conv_layers=self.layers, input_shape=self.input_shape
        )
        if isinstance(self.tuner_type, str):
            # instantiating tuner based on user's choice
            if self.tuner_type == "hyperband":
                tuner = kt.Hyperband(
                    hypermodel=hypermodel,
                    objective="mean_squared_error",
                    max_epochs=10,
                    overwrite=True,
                    project_name=self.project_name,
                    directory=self.directory,
                    **self.tuner_kwargs,
                )

            elif self.tuner_type == "bayesian":
                tuner = kt.BayesianOptimization(
                    hypermodel=hypermodel,
                    objective="mean_squared_error",
                    max_trials=10,
                    overwrite=True,
                    project_name=self.project_name,
                    directory=self.directory,
                    **self.tuner_kwargs,
                )

            elif self.tuner_type == "random":
                tuner = kt.RandomSearch(
                    hypermodel=hypermodel,
                    objective="mean_squared_error",
                    max_trials=10,
                    overwrite=True,
                    project_name=self.project_name,
                    directory=self.directory,
                    **self.tuner_kwargs,
                )
            else:
                raise ValueError(
                    f"tuner parameter expects 'hyperband', 'bayesian', or 'random', but you provided {self.tuner_type}"
                )
            return tuner
        # We do not cover the following line, because CNN also handles this
        # error:
        else:  # pragma: no cover
            raise TypeError(
                f"`tuner` parameter should be a string, but you provided {self.tuner_type}"
            )

    def _get_best_weights(self, model, X, y):
        """Fit a CNN and save the best weights.

        Use keras ModelCheckpoint to fit CNN and save the weights from the epoch
        that produced the lowest validation loss to a temporary file. Uses
        temporary file to load the best weights into the CNN model and returns
        this best model.

        Parameters
        ----------
        model : tensorflow.keras.Sequential()
                Hyperparameters class from which to sample hyperparameters

        X : array-like of shape (n_samples, n_features)
                The feature samples

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns
        -------
        model : tensorflow.keras.Model
                fitted keras model with best weights loaded

        """
        weights_path = op.join(tempfile.mkdtemp(), "weights.hdf5")
        # making model checkpoint to save best model (# epochs) to file
        model_checkpoint_callback = ModelCheckpoint(
            filepath=weights_path,
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            save_weights_only=True,
            verbose=True,
        )

        # Fitting model using model checkpoint callback to find best model which is saved to 'weights'
        model.fit(
            X,
            y,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[model_checkpoint_callback],
            validation_data=(self.X_test, self.y_test),
        )
        # loading in weights
        model.load_weights(weights_path)

        # return the model
        return model

    def build_basic_model(self, X, y):
        """Build a sequential model without hyperparameter tuning.

        Builds a static baseline sequential model with no hyperparameter tuning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                The feature samples

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns
        -------
        model : tensorflow.keras.Model
                compiled model using basic Weston Havens architecture

        """
        model = Sequential()
        model.add(Dense(128, activation="relu", input_shape=X.shape[1:]))
        model.add(Conv1D(24, kernel_size=2, activation="relu"))
        model.add(MaxPool1D(pool_size=2, padding="same"))
        model.add(Conv1D(32, kernel_size=2, activation="relu"))
        model.add(MaxPool1D(pool_size=2, padding="same"))
        model.add(Conv1D(64, kernel_size=3, activation="relu"))
        model.add(MaxPool1D(pool_size=2, padding="same"))
        model.add(Conv1D(128, kernel_size=4, activation="relu"))
        model.add(MaxPool1D(pool_size=2, padding="same"))
        model.add(Conv1D(256, kernel_size=4, activation="relu"))
        model.add(MaxPool1D(pool_size=2, padding="same"))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="linear"))

        model.compile(
            loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"]
        )

        best_model = self._get_best_weights(model, X, y)
        return best_model

    def build_tuned_model(self, X, y):
        """Build a tuned model using Keras tuner.

        Initializes a Keras tuner on user's model, searches for best hyperparameters, and saves them.
        Then builds "best" model using saved best hyperparameters found during the search and returns model
        with best weights loaded from _get_best_weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                The feature samples

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns
        -------
        model : tensorflow.keras.Model
                compiled model that uses hyperparameters defined inline to hypertune the model

        """
        # initialize tuner
        tuner = self._get_tuner()

        # Find the optimal hyperparameters
        tuner.search(X, y, epochs=50, validation_split=0.2)

        # Save the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # make CNN model using best hyperparameters
        model = tuner.hypermodel.build(best_hps)

        best_model = self._get_best_weights(model, X, y)
        return best_model


class CNN:
    """A Convolutional Neural Network model with a fit/predict interface.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in each bundle profile.

    n_channels : int
        Number of metrics in each bundle profile.

    max_epochs : int
        Maximum number of epochs to train model.

    batch_size : int
        Number of samples per batch.

    tuner_type : str
        Type of hyperparameter tuner to use. One of 'hyperband', 'bayesian', or
        'random'.

    layers : int
        Number of convolutional layers to use.

    test_size : float
        Fraction of data to use as test set.

    impute_strategy : str, optional
        Imputation strategy to use. One of 'mean', 'median', or 'knn'.
        Default: "median".

    random_state : int or RandomState instance, optional
        Default: None.

    directory : str, optional
        Directory to save model and hyperparameters. Default: "."

    project_name : str, optional
        A string, the name to use as prefix for files saved by the tuner
        object. Defaults to None

    tuner_kwargs : dict, optional
        Keyword arguments to pass to tuner. Default: tuner defaults.
    """

    def __init__(
        self,
        n_nodes,
        n_channels,
        max_epochs=50,
        batch_size=32,
        tuner_type=None,
        layers=1,
        test_size=0.2,
        impute_strategy="median",
        random_state=None,
        directory=None,
        project_name=None,
        **tuner_kwargs,
    ):
        # checking n_nodes is passed as int
        if not isinstance(n_nodes, int):
            raise TypeError("Parameter n_nodes must be an integer.")
        else:
            self.n_nodes = n_nodes

        # checking n_channels is passed as int
        if not isinstance(n_channels, int):
            raise TypeError("Parameter n_channels must be an integer.")
        else:
            self.n_channels = n_channels

        # checking layers is passed as int
        if not isinstance(layers, int):
            raise TypeError("Parameter layers must be an integer.")
        else:
            self.layers = layers

        # checking max epochs is passed as int
        if not isinstance(max_epochs, int):
            raise TypeError("Parameter max_epochs must be an integer.")
        else:
            self.max_epochs = max_epochs

        if not isinstance(batch_size, int):
            raise TypeError("Parameter batch_size must be an integer.")
        else:
            self.batch_size = batch_size

        # checking tiner is passed as str or None
        if not isinstance(tuner_type, str) and tuner_type is not None:
            raise TypeError("Parameter tuner must be str.")
        else:
            # tuner can be None (no tuning) BayesianOptimization, Hyperband, or RandomSearch
            self.tuner_type = tuner_type

        # checking val split is passed as float
        if not isinstance(test_size, float):
            raise TypeError("Parameter test_size must be a float.")
        else:
            self.test_size = test_size

        # checking strategy is passed as str and has value of 'median', 'mean', or 'knn'
        if not isinstance(impute_strategy, str):
            raise TypeError("Parameter impute_strategy must be a string.")
        elif impute_strategy not in ["median", "mean", "knn"]:
            raise ValueError(
                f"Parameter impute_strategy must be 'median', 'mean', or 'knn' but you provided {impute_strategy}"
            )
        else:
            self.impute_strategy = impute_strategy

        if random_state is not None:
            if not (isinstance(random_state, int) or isinstance(np.random.RandomState)):
                raise TypeError(
                    f"Parameter random_state must be an int or RandomState, but you provided {random_state}"
                )
        self.random_state = random_state

        self.directory = directory
        self.project_name = project_name
        self.tuner_kwargs = tuner_kwargs
        self.model_ = None
        self.best_hps_ = None

    def _preprocess(self, X, y=None):
        """Convert feature matrix for input into a CNN.

        Masks NAN values for X and y (if y is given), imputes X, and reshapes X
        to be in proper form for CNN model. In more conventional machine
        learning, X has shape (n_samples, n_features), where n_features is
        n_nodes * n_bundles * n_metrics. However, in our CNN approach, we treat
        each bundle/metric combination as a separate channel, analogous to RGB
        channels in a 2D image. The remaining one dimension is the nodes
        dimension. Thus the output has shape (n_samples, n_channels, n_nodes),
        where n_channels = n_metrics * n_bundles.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_metrics * n_nodes)
                Diffusion MRI tractometry features (columns) for each subject in the sample (rows).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns
        -------
        X : array-like of shape (n_samples, n_channels, n_nodes)
                The imputed and reshaped feature samples

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        """
        # n_nodes * n_channels must = X.shape[1]
        if self.n_nodes * self.n_channels != X.shape[1]:
            raise ValueError(
                "The product n_nodes and n_channels is not the correct shape."
            )

        # We don't cover the following line, because this case is also handled
        # in the fall to fit:
        if len(X.shape) > 2:  # pragma: no cover
            raise ValueError("Expected X to be a 2D matrix.")
        if y is not None:
            nan_mask = np.logical_not(np.isnan(y))
            X = X[nan_mask, :]
            y = y[nan_mask]

        imp = SimpleImputer(strategy=self.impute_strategy)
        X = imp.fit_transform(X)

        if y is not None:
            X, y = check_X_y(X, y)

        n_subjects = X.shape[0]

        X = np.swapaxes(X.reshape((n_subjects, self.n_channels, self.n_nodes)), 1, 2)

        if y is not None:
            return X, y
        else:
            return X

    def fit(self, X, y):
        """Fit the model.

        Preprocesses X and y, builds CNN model, tunes model hyperparameters and
        fits the model to given X and y, using X_test and y_test to validate and
        find best weights and hyperparameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_metrics * n_nodes)
                Diffusion MRI tractometry features (columns) for each subject (rows).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns
        -------
        self : CNN
                updated CNN instantiation

        """
        X, y = self._preprocess(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # CNN gets n_nodes, n_channels, max_epochs, tuner=None, layers=None
        # Model Builder takes tuner_type, input_shape, layers, max_epochs, **kwargs
        builder = ModelBuilder(
            self.tuner_type,
            X_train.shape[1:],
            self.layers,
            self.max_epochs,
            X_test,
            y_test,
            self.batch_size,
            self.directory,
            self.project_name,
            **self.tuner_kwargs,
        )
        if self.tuner_type is None:
            self.model_ = builder.build_basic_model(X_train, y_train)
        else:
            self.model_ = builder.build_tuned_model(X_train, y_train)

        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Predict target values.

        Preprocesses X and returns predicted y values for X from fitted CNN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_metrics * n_nodes)
                Tractometry features (columns) for each subject in the sample (rows).

        Returns
        -------
        pred : array-like of shape (n_samples,) or (n_samples, n_targets)
                predicted values
        """
        X = self._preprocess(X)
        check_is_fitted(self, "is_fitted_")
        pred = self.model_.predict(X).squeeze()
        return pred

    def score(self, y_test, y_hat):
        """Score the performance of the model.

        Masks out NaN values from y_test and returns $R^2$ score for the CNN model comparing to y_hat

        Parameters
        ----------
        y_test : array-like of shape (n_samples,) or (n_samples, n_targets)
                Testing target values

        y_hat : array-like of shape (n_samples,) or (n_samples, n_targets)
                Predicted target values

        Returns
        -------
        r2_score : float
                r-squared score for y_test and y_hat for CNN model

        """
        nan_mask = np.logical_not(np.isnan(y_test))
        y_test = y_test[nan_mask]
        return r2_score(y_test, y_hat)
