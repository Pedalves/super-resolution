import tensorflow as tf
import numpy as np

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(0)

tf.keras.backend.set_floatx('float32')


class RDB(tf.keras.layers.Layer):
    def __init__(self, filters, c=3, kernel_size=(3, 3)):
        super(RDB, self).__init__()
        self.filters = filters
        self.c = c
        self.kernel_size = kernel_size
        self.convs = []

    def build(self, input_shape):
        trainable_weights = []
        self.convs = []

        for i in range(self.c):
            self.convs.append(tf.keras.layers.Conv2D(filters=self.filters,
                                                     kernel_size=self.kernel_size,
                                                     activation=tf.nn.relu,
                                                     padding='same')
                              )
            self.convs[i].build(input_shape=input_shape)
            trainable_weights = trainable_weights + self.convs[i].trainable_weights

        self.conv_concat = tf.keras.layers.Conv2D(filters=self.filters,
                                                  kernel_size=(1, 1),
                                                  activation=tf.nn.relu,
                                                  padding='same')
        self.conv_concat.build(input_shape=input_shape[:-1] + [(self.c + 1) * self.filters])
        trainable_weights = trainable_weights + self.conv_concat.trainable_weights

        self._trainable_weights = trainable_weights

    def call(self, x):
        x_list = [x]
        for i in range(self.c):
            x_list.append(self.convs[i](sum(x_list[:i + 1])))

        x = self.conv_concat(tf.concat(x_list, axis=-1)) + x

        return x
