import math
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWire

keras_msg = (
    "To use afqinsight's tensorflow models, you need to have tensorflow "
    "installed. You can do this by installing afqinsight with `pip install "
    "afqinsight[tf]`, or by separately installing tensorflow with `pip install "
    "tensorflow`."
)

tf, has_tf, _ = optional_package("tensorflow", keras_msg)

if has_tf:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
    from tensorflow.keras.layers import MaxPooling1D, Conv1D
    from tensorflow.keras.layers import LSTM, Bidirectional
    from tensorflow.keras.layers import (
        BatchNormalization,
        GlobalAveragePooling1D,
        Permute,
        concatenate,
        Activation,
        add,
    )
else:
    # Since all model building functions start with Input, we make Input the
    # tripwire instance for cases where tensorflow is not installed.
    Input = TripWire(keras_msg)


def mlp4(input_shape, n_classes, output_activation="softmax", verbose=False):
    # Z. Wang, W. Yan, T. Oates, "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline," Int. Joint Conf. Neural Networks, 2017, pp. 1578-1585
    ip = Input(shape=input_shape)
    fc = Flatten()(ip)
    fc = Dropout(0.1)(fc)

    fc = Dense(500, activation="relu")(fc)
    fc = Dropout(0.2)(fc)

    fc = Dense(500, activation="relu")(fc)
    fc = Dropout(0.2)(fc)

    fc = Dense(500, activation="relu")(fc)
    fc = Dropout(0.3)(fc)

    out = Dense(n_classes, activation=output_activation)(fc)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def cnn_lenet(input_shape, n_classes, output_activation="softmax", verbose=False):
    # Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

    ip = Input(shape=input_shape)
    conv = ip

    n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
    if verbose:
        print("pooling layers: %d" % n_conv_layers)

    for i in range(n_conv_layers):
        conv = Conv1D(
            6 + 10 * i,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_uniform",
        )(conv)
        conv = MaxPooling1D(pool_size=2)(conv)

    flat = Flatten()(conv)

    fc = Dense(120, activation="relu")(flat)
    fc = Dropout(0.5)(fc)

    fc = Dense(84, activation="relu")(fc)
    fc = Dropout(0.5)(fc)

    out = Dense(n_classes, activation=output_activation)(fc)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def cnn_vgg(input_shape, n_classes, output_activation="softmax", verbose=False):
    # K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.

    ip = Input(shape=input_shape)
    conv = ip

    n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
    if verbose:
        print("pooling layers: %d" % n_conv_layers)

    for i in range(n_conv_layers):
        num_filters = min(64 * 2**i, 512)
        conv = Conv1D(
            num_filters,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_uniform",
        )(conv)
        conv = Conv1D(
            num_filters,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_uniform",
        )(conv)
        if i > 1:
            conv = Conv1D(
                num_filters,
                3,
                padding="same",
                activation="relu",
                kernel_initializer="he_uniform",
            )(conv)
        conv = MaxPooling1D(pool_size=2)(conv)

    flat = Flatten()(conv)

    fc = Dense(4096, activation="relu")(flat)
    fc = Dropout(0.5)(fc)

    fc = Dense(4096, activation="relu")(fc)
    fc = Dropout(0.5)(fc)

    out = Dense(n_classes, activation=output_activation)(fc)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def lstm1v0(input_shape, n_classes, output_activation="softmax", verbose=False):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

    ip = Input(shape=input_shape)

    l2 = LSTM(512)(ip)
    out = Dense(n_classes, activation=output_activation)(l2)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def lstm1(input_shape, n_classes, output_activation="softmax", verbose=False):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

    # Hyperparameter choices:
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017

    ip = Input(shape=input_shape)

    l2 = LSTM(100)(ip)
    out = Dense(n_classes, activation=output_activation)(l2)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def lstm2(input_shape, n_classes, output_activation="softmax", verbose=False):
    ip = Input(shape=input_shape)

    l1 = LSTM(100, return_sequences=True)(ip)
    l2 = LSTM(100)(l1)
    out = Dense(n_classes, activation=output_activation)(l2)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def blstm1(input_shape, n_classes, output_activation="softmax", verbose=False):
    # Original proposal:
    # M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.

    # Hyperparameter choices:
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    ip = Input(shape=input_shape)

    l2 = Bidirectional(LSTM(100))(ip)
    out = Dense(n_classes, activation=output_activation)(l2)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def blstm2(input_shape, n_classes, output_activation="softmax", verbose=False):
    ip = Input(shape=input_shape)

    l1 = Bidirectional(LSTM(100, return_sequences=True))(ip)
    l2 = Bidirectional(LSTM(100))(l1)
    out = Dense(n_classes, activation=output_activation)(l2)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def lstm_fcn(input_shape, n_classes, output_activation="softmax", verbose=False):
    # F. Karim, S. Majumdar, H. Darabi, and S. Chen, “LSTM Fully Convolutional Networks for Time Series Classification,” IEEE Access, vol. 6, pp. 1662–1669, 2018.

    ip = Input(shape=input_shape)

    # lstm part is a 1 time step multivariate as described in Karim et al. Seems strange, but works I guess.
    lstm = Permute((2, 1))(ip)

    lstm = LSTM(128)(lstm)
    lstm = Dropout(0.8)(lstm)

    conv = Conv1D(128, 8, padding="same", kernel_initializer="he_uniform")(ip)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    conv = Conv1D(256, 5, padding="same", kernel_initializer="he_uniform")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    conv = Conv1D(128, 3, padding="same", kernel_initializer="he_uniform")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    flat = GlobalAveragePooling1D()(conv)

    flat = concatenate([lstm, flat])

    out = Dense(n_classes, activation=output_activation)(flat)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


def cnn_resnet(input_shape, n_classes, output_activation="softmax", verbose=False):
    # I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P-A Muller, "Data augmentation using synthetic data for time series classification with deep residual networks," International Workshop on Advanced Analytics and Learning on Temporal Data ECML/PKDD, 2018

    ip = Input(shape=input_shape)
    residual = ip
    conv = ip

    for i, nb_nodes in enumerate([64, 128, 128]):
        conv = Conv1D(nb_nodes, 8, padding="same", kernel_initializer="glorot_uniform")(
            conv
        )
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)

        conv = Conv1D(nb_nodes, 5, padding="same", kernel_initializer="glorot_uniform")(
            conv
        )
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)

        conv = Conv1D(nb_nodes, 3, padding="same", kernel_initializer="glorot_uniform")(
            conv
        )
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)

        if i < 2:
            # expands dimensions according to Fawaz et al.
            residual = Conv1D(
                nb_nodes, 1, padding="same", kernel_initializer="glorot_uniform"
            )(residual)
        residual = BatchNormalization()(residual)
        conv = add([residual, conv])
        conv = Activation("relu")(conv)

        residual = conv

    flat = GlobalAveragePooling1D()(conv)

    out = Dense(n_classes, activation=output_activation)(flat)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model
