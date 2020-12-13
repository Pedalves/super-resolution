import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from utils.metrics import mse, psnr

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(0)


class NeuralNetwork(tf.keras.Model):
    def __init__(self, base_name, init_shape, learning_rate=0.001, loss='mae', **kwargs):
        super(NeuralNetwork, self).__init__(name=base_name, **kwargs)

        self.base_name = base_name

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
        return self.get_name()

    def get_name(self):
        return f'{self.base_name}-ep_{self.total_epochs}'

    def call(self, input_features):
        raise NotImplementedError

    @tf.function
    def train(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self(x)

            loss_value = self.loss(predictions, y)

            self.loss_metric.update_state(predictions, y)

            gradients = tape.gradient(loss_value, self.trainable_variables)

            gradient_variables = zip(gradients, self.trainable_variables)

            self.optimizer.apply_gradients(gradient_variables)

            return loss_value

    def save_weights(self, path='../_data/weights/'):
        name = path + self.get_name() + '.h5'

        super().save_weights(name)

        print(f'weights saved as {name}')
        return name

    def load_weights(self, weights):
        self.total_epochs = int(weights.split('ep_')[-1].split('.h5')[0])

        super().load_weights(weights)

    def create_model(self):
        _ = self(np.random.random_sample((1, *self.init_shape)))

    @staticmethod
    def get_tensor_dataset(x, y, x_val, y_val):
        x_train = tf.data.Dataset.from_tensor_slices(x)
        x_val = tf.data.Dataset.from_tensor_slices(x_val)

        y_train = tf.data.Dataset.from_tensor_slices(y)
        y_val = tf.data.Dataset.from_tensor_slices(y_val)

        return x_train, y_train, x_val, y_val

    def train_epochs(self, x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose=True, frequency=5):
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
                    psnr_val = psnr(y_val, self(x_val))
                    print(f'Epoch: {epoch} Train Loss: {loss_train} Validation Loss: {np.mean(loss_val)} '
                          f'Validation PSNR: {psnr_val}')

            loss.append(np.mean(loss_val))

            self.log_metrics['loss_train'].append(loss_train)
            self.log_metrics['loss_val'].append(np.mean(loss_val))

            self.loss_metric.reset_states()

        if verbose:
            psnr_val = psnr(y_val, self(x_val))
            print(f'Epoch: {epoch} Train Loss: {loss_train} Validation Loss: {np.mean(loss_val)} '
                  f'Validation PSNR: {psnr_val}')

        return pd.DataFrame(self.log_metrics)


from .mlp import MLP

__all__ = [
    'MLP'
]