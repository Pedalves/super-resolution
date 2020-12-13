import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from utils.metrics import mse, psnr

from . import NeuralNetwork

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(0)

tf.keras.backend.set_floatx('float32')


class MLP(NeuralNetwork):
    def __init__(self, init_shape, layers_dims, original_dim=64, learning_rate=0.001,
                 activ_hidden=tf.nn.relu, activ_out=tf.nn.tanh):
        super(MLP, self).__init__(init_shape=init_shape, base_name='MLP', learning_rate=learning_rate)

        self.resolution = original_dim

        self.activation_hidden_layer = activ_hidden
        self.activation_output_layer = activ_out

        self.hidden_layers = []

        for i in range(len(layers_dims)):
            self.hidden_layers.append(tf.keras.layers.Dense(
                units=layers_dims[i],
                activation=activ_hidden,
                kernel_initializer='he_uniform',
                name=f'dense_layer_{i}'
            ))

        self.output_layer = tf.keras.layers.Dense(
            units=self.resolution ** 2,
            activation=activ_out,
            kernel_initializer='he_uniform',
            name=f'output_layer'
        )

        self.create_model()

    def call(self, input_features):
        result = tf.reshape(input_features, (-1, (self.resolution // 2) ** 2))
        for i in range(len(self.hidden_layers)):
            result = self.hidden_layers[i](result)

        out = self.output_layer(result)

        return tf.reshape(out, (-1, self.resolution, self.resolution))
