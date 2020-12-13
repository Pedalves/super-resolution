import tensorflow as tf
import numpy as np

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(0)

tf.keras.backend.set_floatx('float32')


class RDB(tf.keras.layers.Layer):
    def __init__(self, filters, block_size=3, kernel_size=(3, 3), name='rdb_layer', **kwargs):
        super(RDB, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.convs = []

        for i in range(self.block_size):
            self.convs.append(tf.keras.layers.Conv2D(filters=self.filters,
                                                     kernel_size=self.kernel_size,
                                                     activation=tf.nn.relu,
                                                     padding='same',
                                                     name=f'conv_{i}_{self.name}'
                                                     )
                              )

        self.conv_concat = tf.keras.layers.Conv2D(filters=self.filters,
                                                  kernel_size=(1, 1),
                                                  activation=tf.nn.relu,
                                                  padding='same',
                                                  name=f'conv_concat_{self.name}')

    def call(self, x):
        x_list = [x]
        for i in range(self.block_size):
            x_list.append(self.convs[i](sum(x_list[:i + 1])))

        x = self.conv_concat(tf.concat(x_list, axis=-1)) + x

        return x
