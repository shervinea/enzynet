"""Model definitions."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import numpy as np

from enzynet import constants
from keras import initializers
from keras import layers
from keras.layers import advanced_activations
from keras import models
from keras import regularizers


def enzynet(input_v_size: int, n_channels: int):
    """Returns EnzyNet as a Keras model."""
    # Parameters.
    stddev_conv3d = np.sqrt(2.0/n_channels)

    # Initialization.
    model = models.Sequential()

    # Add layers.
    model.add(
        layers.Conv3D(
            filters=32,
            kernel_size=9,
            strides=2,
            padding='valid',
            kernel_initializer=initializers.RandomNormal(
                mean=0.0,
                stddev=stddev_conv3d * 9 ** (-3 / 2)),
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None,
            input_shape=(input_v_size,)*constants.N_DIMENSIONS + (n_channels,)))

    model.add(advanced_activations.LeakyReLU(alpha=0.1))

    model.add(layers.Dropout(rate=0.2))

    model.add(
        layers.Conv3D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding='valid',
            kernel_initializer=initializers.RandomNormal(
                mean=0.0,
                stddev=stddev_conv3d * 5 ** (-3 / 2)),
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None))

    model.add(advanced_activations.LeakyReLU(alpha=0.1))

    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Flatten())

    model.add(
        layers.Dense(
            units=128,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None))

    model.add(layers.Dropout(rate=0.4))

    model.add(
        layers.Dense(
            units=constants.N_CLASSES,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None))

    model.add(layers.Activation('softmax'))

    return model
