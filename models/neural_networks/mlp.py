import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from utils.metrics import mse, psnr

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(0)

tf.keras.backend.set_floatx('float64')


class MLP(tf.keras.Model):
    def __init__(self, int_dims, original_dim=64, learning_rate=0.001, activ_hidden=tf.nn.relu, activ_out=None):
        super(MLP, self).__init__()

        self.resolution = original_dim

        self.activation_hidden_layer = activ_hidden
        self.activation_output_layer = activ_out

        self.hidden_layers = []

        for i in range(len(int_dims)):
            self.hidden_layers.append(tf.keras.layers.Dense(
                units=int_dims[i],
                activation=activ_hidden,
                kernel_initializer='he_uniform'))

        self.output_layer = tf.keras.layers.Dense(
            units=self.resolution ** 2,
            activation=activ_out,
            kernel_initializer='he_uniform')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.loss_metric = tf.keras.metrics.MeanSquaredError()

        self.loss = tf.keras.losses.MeanSquaredError()

        self.log_metrics = defaultdict(list)

        self.learning_rate = learning_rate

    def call(self, input_features):
        result = tf.reshape(input_features, (-1, (self.resolution // 2) ** 2))
        for i in range(len(self.hidden_layers)):
            result = self.hidden_layers[i](result)

        out = self.output_layer(result)

        return tf.reshape(out, (-1, self.resolution, self.resolution))

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

    def make_dataset(self, x, y, x_val, y_val):
        x_train = tf.data.Dataset.from_tensor_slices(x)
        x_val = tf.data.Dataset.from_tensor_slices(x_val)

        y_train = tf.data.Dataset.from_tensor_slices(y)
        y_val = tf.data.Dataset.from_tensor_slices(y_val)

        return x_train, y_train, x_val, y_val

    def train_epochs(self, x_train, y_train, x_val, y_val, n_epochs, batch_size, SEED=42):
        x_train, y_train, x_val, y_val = self.make_dataset(x_train, y_train, x_val, y_val)
        loss = []
        for epoch in range(n_epochs):
            loss_val = []
            batches_x = x_train.batch(batch_size)
            batches_y = y_train.batch(batch_size)

            # Training loop
            for _x, _y in zip(batches_x, batches_y):
                self.train(_x, _y)

            batches_x_val = x_val.batch(batch_size)
            batches_y_val = y_val.batch(batch_size)

            # Validation loop
            for _x, _y in zip(batches_x, batches_y):
                loss_val.append(self.loss(self(_x), _y))

            loss_train = self.loss_metric.result().numpy()

            if epoch % 20 == 0 or epoch == 0:
                print(f'Epoch: {epoch} Train Loss: {loss_train} Validation Loss: {np.mean(loss_val)}')
            loss.append(np.mean(loss_val))

            self.log_metrics['loss_train'].append(loss_train)
            self.log_metrics['loss_val'].append(np.mean(loss_val))

            self.loss_metric.reset_states()

        print(f'Epoch: {epoch} Train Loss: {loss_train} Validation Loss: {np.mean(loss_val)}')
        return pd.DataFrame(self.log_metrics)

    def evaluate(self, x, y):
        y_pred = self(x)

        metrics = {'mse': [mse(y, y_pred)],
                   'psnr': [psnr(y, y_pred)]}

        return pd.DataFrame(metrics)