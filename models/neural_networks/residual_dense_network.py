import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from utils.metrics import mse, psnr

from . import NeuralNetwork
from .layers.rdb import RDB

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(0)

tf.keras.backend.set_floatx('float32')


class ResidualDenseNetwork(NeuralNetwork):
    def __init__(self, init_shape, filters=64, original_dim=64, rdb_blocks=3, rdb_block_size=3,
                 learning_rate=0.0001, activ_hidden=tf.nn.relu, activ_out=tf.nn.tanh, loss='mae', **kwargs):
        super(ResidualDenseNetwork, self).__init__(init_shape=init_shape,
                                                   base_name='ResidualDenseNetwork',
                                                   learning_rate=learning_rate, loss=loss, **kwargs)

        self.resolution = original_dim

        self.activation_hidden_layer = activ_hidden
        self.activation_output_layer = activ_out

        self.filters = filters

        self.sfe1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=activ_hidden,
            padding='same',
            name='sfe1_layer'
        )

        self.sfe2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=activ_hidden,
            padding='same',
            name='sfe2_layer'
        )

        # RDB
        self.rdb_block_size = rdb_block_size
        self.rdb = [RDB(filters=filters, block_size=rdb_block_size, name=f'rdb_layer_{i}') for i in range(rdb_blocks)]

        #########

        self.conv1x1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=activ_hidden,
            padding='same',
            name='conv1x1_layer'
        )

        self.conv_gf = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation=activ_hidden,
            padding='same',
            name='gf_layer'
        )

        self.upsample = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=tf.nn.tanh,
            padding='same',
            name=f'upsample_layer'
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation=activ_out,
            padding='same',
            name=f'output_layer'
        )

        self.create_model()

    def get_architecture_str(self):
        return f'a_{self.filters}-rdb_{len(self.rdb)}-bs_{self.rdb_block_size}-' \
               f'hl_{self.activation_hidden_layer.__name__}-ol_{self.activation_output_layer.__name__}'

    def call(self, input_features):
        x = tf.expand_dims(input_features, -1)
        f_m1 = self.sfe1(x)

        f_list = [self.sfe2(f_m1)]

        # RDB
        for i in range(len(self.rdb)):
            f_list.append(self.rdb[i](f_list[i]))

        x = self.conv1x1(tf.concat(f_list[1:], axis=-1))

        f_gf = self.conv_gf(x)

        x = self.upsample(f_gf + f_m1)
        x = self.output_layer(x)

        return tf.reshape(x, (-1, self.resolution, self.resolution))
