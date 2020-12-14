import pytest
import tensorflow as tf

from models.neural_networks import MLP, NeuralNetwork, ResidualDenseNetwork
from dataset.artificial_dataset import ArtificialDatasetReader
from utils.metrics import psnr, mse


def test_neural_networks_super_class():
    model = MLP((32, 32), [512], learning_rate=0.0001, activ_hidden=tf.nn.tanh, activ_out=tf.nn.tanh)

    assert isinstance(model, NeuralNetwork)

    model_2 = ResidualDenseNetwork(init_shape=(32, 32), filters=64, rdb_block_size=6, rdb_blocks=12)

    assert isinstance(model_2, NeuralNetwork)


@pytest.mark.parametrize(
    "model, name", [
        (
                MLP((32, 32), [512], learning_rate=0.0001, activ_hidden=tf.nn.tanh, activ_out=tf.nn.tanh),
                'MLP-a_512-hl_tanh-ol_tanh-l_mae-lr_0.0001-ep_0'
         ),
        (
                MLP((32, 32), [512, 10], learning_rate=0.01, activ_hidden=tf.nn.tanh, activ_out=tf.nn.tanh, loss='mse'),
                'MLP-a_512_10-hl_tanh-ol_tanh-l_mse-lr_0.01-ep_0'
         ),
        (
                ResidualDenseNetwork(init_shape=(32, 32), filters=64, rdb_block_size=6, rdb_blocks=12),
                'RDN-a_64-rdb_12-bs_6-hl_relu-ol_tanh-l_mae-lr_0.0001-ep_0'
         )
    ]
)
def test_neural_network_name(model, name):
    assert model.get_name() == name


@pytest.mark.parametrize(
    "model_1, model_2", [
        (
                MLP((32, 32), [512], learning_rate=0.0001, activ_hidden=tf.nn.tanh, activ_out=tf.nn.tanh),
                MLP((32, 32), [512], learning_rate=0.0001, activ_hidden=tf.nn.tanh, activ_out=tf.nn.tanh),
        ),
        (
                ResidualDenseNetwork(init_shape=(32, 32), filters=64, rdb_block_size=1, rdb_blocks=3),
                ResidualDenseNetwork(init_shape=(32, 32), filters=64, rdb_block_size=1, rdb_blocks=3)
        )
    ]
)
def test_save_neural_network(model_1, model_2):
    dr = ArtificialDatasetReader('_data/dataset/artificial/')
    x_train, y_train, x_val, y_val, _, _ = dr.get_dataset(img_size=64)

    _ = model_1.train_epochs(x_train, y_train, x_val, y_val, 1, 64, verbose=False)
    y = model_1(x_val)
    mse_1 = mse(y_val, y)
    psnr_1 = psnr(y_val, y)

    # Save
    save_path = model_1.save_weights(path='_data/weights/')

    # Load
    model_2.load_weights(save_path)

    y_2 = model_2(x_val)

    mse_2 = mse(y_val, y_2)
    psnr_2 = psnr(y_val, y_2)

    assert mse_1 == mse_2
    assert psnr_1 == psnr_2
