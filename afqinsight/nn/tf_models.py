import math
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWire

keras_msg = (
    "To use afqinsight's tensorflow models, you need to have tensorflow "
    "installed. You can do this by installing afqinsight with `pip install "
    "afqinsight[tf]`, or by separately installing tensorflow with `pip install "
    "tensorflow`."
)

tf, has_tf, _ = optional_package("tensorflow", trip_msg=keras_msg)

if has_tf:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Reshape
    from tensorflow.keras.layers import MaxPooling1D, Conv1D, Conv1DTranspose
    from tensorflow.keras.layers import LSTM, Bidirectional
    from tensorflow.keras.layers import (
        BatchNormalization,
        GlobalAveragePooling1D,
        Permute,
        concatenate,
        Activation,
        add,
        Layer,
    )
    from tensorflow.keras.losses import binary_crossentropy

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


def fc_autoencoder(input_shape, encoding_dim=None, verbose=False):
    """
    Fully connected autoencoder
    """
    ip = Input(shape=input_shape)
    if encoding_dim is None:
        encoding_dim = (input_shape[0] * input_shape[1]) // 8

    fc = Flatten()(ip)
    fc = Dense(input_shape[0] * input_shape[1], activation="relu")(fc)
    fc = Dense((input_shape[0] * input_shape[1]) // 2, activation="relu")(fc)
    fc = Dense((input_shape[0] * input_shape[1]) // 4, activation="relu")(fc)
    fc = Dense(encoding_dim, activation="relu")(fc)
    fc = Dense((input_shape[0] * input_shape[1]) // 4, activation="relu")(fc)
    fc = Dense((input_shape[0] * input_shape[1]) // 2, activation="relu")(fc)
    pre_out = Dense((input_shape[0] * input_shape[1]))(fc)
    out = Reshape(input_shape)(pre_out)

    model = Model([ip], [out])
    if verbose:
        model.summary()
    return model


def cnn_autoencoder(input_shape, verbose=False):
    """
    Convolutional autoencoder
    """
    ip = Input(shape=input_shape)
    # Encoder
    x = Conv1D(32, (3), activation="relu", padding="same")(ip)
    x = MaxPooling1D((2), padding="same")(x)
    x = Conv1D(32, (3), activation="relu", padding="same")(x)
    x = MaxPooling1D((2), padding="same")(x)

    # Decoder
    x = Conv1DTranspose(32, (3), strides=2, activation="relu", padding="same")(x)
    x = Conv1DTranspose(32, (3), strides=2, activation="relu", padding="same")(x)
    x = Conv1D(1, (3), activation="sigmoid", padding="same")(x)

    model = Model([ip], [x])
    if verbose:
        model.summary()

    return model


class _Sampling(Layer):
    """
    Sample the latent layer of a VAE
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def _fc_vae_encoder(input_shape, encoding_dim=None, verbose=False):
    """
    Encoder section for a fully connected variational autoencoder
    """
    ip = Input(shape=input_shape)
    if encoding_dim is None:
        encoding_dim = (input_shape[0] * input_shape[1]) // 8

    fc = Flatten()(ip)
    fc = Dense(input_shape[0] * input_shape[1], activation="relu")(fc)
    fc = Dense((input_shape[0] * input_shape[1]) // 2, activation="relu")(fc)
    fc = Dense((input_shape[0] * input_shape[1]) // 4, activation="relu")(fc)

    z_mean = Dense(encoding_dim, activation="relu")(fc)
    z_log_var = Dense(encoding_dim, name="z_mean")(fc)
    z = _Sampling()([z_mean, z_log_var])
    return Model(ip, [z_mean, z_log_var, z], name="encoder")


def _fc_vae_decoder(input_shape, encoding_dim=None, verbose=False):
    """
    Decoder section for a fully connected variational autoencoder
    """

    fc = Dense((input_shape[0] * input_shape[1]) // 4, activation="relu")(fc)
    fc = Dense((input_shape[0] * input_shape[1]) // 2, activation="relu")(fc)
    pre_out = Dense((input_shape[0] * input_shape[1]))(fc)
    out = Reshape(input_shape)(pre_out)


def _VAE(Model):
    """
    A variational autoencoder class
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(binary_crossentropy(data, reconstruction), axis=1)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def fc_vae(input_shape, encoding_dim=None, verbose=False):
    """
    Fully connected variational autoencoder.
    """
    encoder = _fc_vae_encoder(input_shape, encoding_dim, verbose)
    decoder = _fc_vae_decoder(input_shape, encoding_dim, verbose)
    return _VAE(encoder, decoder)
