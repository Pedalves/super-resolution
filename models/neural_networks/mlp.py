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
    """
    Multi Layer Perceptron
    """
    def __init__(self, init_shape, layers_dims=[512], original_dim=64, learning_rate=0.0001,
                 activ_hidden=tf.nn.tanh, activ_out=tf.nn.tanh, loss='mae', **kwargs):
        """
        Class init
        :param init_shape: initial input shape
        :param layers_dims: List with the number of units for each hidden layer
        :param original_dim: Image dimension after increasing resolution
        :param learning_rate: learning rate
        :param activ_hidden: hidden layers activation function
        :param activ_out: output layers activation function
        :param loss: 'mae' or 'mse' loss function
        :param kwargs: kwargs
        """
        super(MLP, self).__init__(init_shape=init_shape, base_name='MLP',
                                  learning_rate=learning_rate, loss=loss, **kwargs)


        self.resolution = original_dim

        self.activation_hidden_layer = activ_hidden
        self.activation_output_layer = activ_out

        self.layer_dims = layers_dims

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

    def get_architecture_str(self):
        return f'a_{"_".join(str(x) for x in self.layer_dims)}-' \
               f'hl_{self.activation_hidden_layer.__name__}-ol_{self.activation_output_layer.__name__}'

    def call(self, input_features):
        result = tf.reshape(input_features, (-1, (self.resolution // 2) ** 2))
        for i in range(len(self.hidden_layers)):
            result = self.hidden_layers[i](result)

        out = self.output_layer(result)

        return tf.reshape(out, (-1, self.resolution, self.resolution))
