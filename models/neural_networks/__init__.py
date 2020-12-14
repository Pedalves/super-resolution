import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from utils.metrics import mse, psnr

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(0)


class NeuralNetwork(tf.keras.Model):
    """
    Neural network base class
    """
    def __init__(self, base_name, init_shape, learning_rate=0.001, loss='mae', **kwargs):
        """
        Class init
        :param base_name: network name
        :param init_shape: initial input shape
        :param learning_rate: learning rate
        :param loss: 'mae' or 'mse' loss function
        :param kwargs: kwargs
        """
        super(NeuralNetwork, self).__init__(name=base_name, **kwargs)

        self.base_name = base_name

        self.loss_name = loss

        self.init_shape = init_shape

        self.learning_rate = learning_rate
        self.total_epochs = 0

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if loss.lower() == 'mae':
            self.loss_metric = tf.keras.metrics.MeanAbsoluteError()
            self.loss = tf.keras.losses.MeanAbsoluteError()
        elif loss.lower() == 'mse':
            self.loss_metric = tf.keras.metrics.MeanSquaredError()
            self.loss = tf.keras.losses.MeanSquaredError()
        else:
            raise AttributeError('Base class NeuralNetwork only supports mse and mae losses')

        self.log_metrics = defaultdict(list)

    def __str__(self):
        """
        :return: network str representation
        """
        return self.get_name()

    def get_architecture_str(self):
        """
        Get the architecture`s string representation
        :return: architecture`s str
        """
        raise NotImplementedError

    def get_name(self):
        """
        Get the network`s name
        :return: network`s str name
        """
        return f'{self.base_name}-{self.get_architecture_str()}-l_{self.loss_name}-lr_{self.learning_rate}-ep_{self.total_epochs}'

    def call(self, input_features):
        """
        Tensorflow call function
        :param input_features: input features
        :return: output value
        """
        raise NotImplementedError

    @tf.function
    def train(self, x, y):
        """
        Get gradient and update weights
        :param x: input values
        :param y: ground truth
        :return: loss value
        """
        with tf.GradientTape() as tape:
            predictions = self(x)

            loss_value = self.loss(predictions, y)

            self.loss_metric.update_state(predictions, y)

            gradients = tape.gradient(loss_value, self.trainable_variables)

            gradient_variables = zip(gradients, self.trainable_variables)

            self.optimizer.apply_gradients(gradient_variables)

            return loss_value

    def save_weights(self, path='../_data/weights/'):
        """
        Save network`s weights as h5
        :param path: weights folder path
        :return: file name
        """
        name = path + self.get_name() + '.h5'

        super().save_weights(name)

        print(f'weights saved as {name}')
        return name

    def load_weights(self, weights):
        """
        Load weights
        :param weights: weights path
        """
        self.total_epochs = int(weights.split('ep_')[-1].split('.h5')[0])

        super().load_weights(weights)

    def create_model(self):
        """
        Build model
        """
        _ = self(np.random.random_sample((1, *self.init_shape)))

    @staticmethod
    def get_tensor_dataset(x, y, x_val, y_val):
        """
        Convert numpy array into tensor slices
        :param x: train input
        :param y: train ground truth
        :param x_val: validation input
        :param y_val: validation ground truth
        :return: x_train, y_train, x_val, y_val tensor slices
        """
        x_train = tf.data.Dataset.from_tensor_slices(x)
        x_val = tf.data.Dataset.from_tensor_slices(x_val)

        y_train = tf.data.Dataset.from_tensor_slices(y)
        y_val = tf.data.Dataset.from_tensor_slices(y_val)

        return x_train, y_train, x_val, y_val

    def train_epochs(self, x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose=True, frequency=5):
        """
        Train model for a given number of epochs
        :param x_train: train input
        :param y_train: train ground truth
        :param x_val: validation input
        :param y_val: validation ground truth
        :param n_epochs: number of epochs
        :param batch_size: batch size
        :param verbose: verbose
        :param frequency: print frequency by number of epochs
        :return: train report DataFrame
        """
        x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor = self.get_tensor_dataset(
            x_train, y_train, x_val, y_val)
        loss = []
        for epoch in range(n_epochs):
            loss_val = []
            batches_x = x_train_tensor.batch(batch_size)
            batches_y = y_train_tensor.batch(batch_size)

            # Training loop
            for _x, _y in zip(batches_x, batches_y):
                self.train(_x, _y)

            batches_x_val = x_val_tensor.batch(batch_size)
            batches_y_val = y_val_tensor.batch(batch_size)

            # Validation loop
            for _x, _y in zip(batches_x_val, batches_y_val):
                loss_val.append(self.loss(self(_x), _y))

            loss_train = self.loss_metric.result().numpy()

            self.total_epochs += 1

            if verbose:
                if epoch % frequency == 0 or epoch == 0:
                    psnr_val = psnr(y_val, self(x_val), verbose=False)
                    print(f'Epoch: {epoch} Train Loss: {loss_train} Validation Loss: {np.mean(loss_val)} '
                          f'Validation PSNR: {psnr_val}')

            loss.append(np.mean(loss_val))

            self.log_metrics['loss_train'].append(loss_train)
            self.log_metrics['loss_val'].append(np.mean(loss_val))

            self.loss_metric.reset_states()

        if verbose:
            psnr_val = psnr(y_val, self(x_val), verbose=False)
            print(f'Epoch: {epoch} Train Loss: {loss_train} Validation Loss: {np.mean(loss_val)} '
                  f'Validation PSNR: {psnr_val}')

        return pd.DataFrame(self.log_metrics)


from .mlp import MLP
from .residual_dense_network import ResidualDenseNetwork

__all__ = [
    'MLP',
    'ResidualDenseNetwork',
    'NeuralNetwork'
]
