from dataset.artificial_dataset import ArtificialDatasetReader
from models.baseline import Baseline
from models.neural_networks import MLP, ResidualDenseNetwork
from utils.metrics import mse, psnr


def run_baseline(dataset_path='../_data/dataset/artificial'):
    dr = ArtificialDatasetReader(dataset_path=dataset_path)
    x_train, y_train, x_val, y_val, _, _ = dr.get_dataset(img_size=64)

    baseline = Baseline()

    print('Running Baseline')
    print('Train: ')
    _ = baseline.evaluate(x_train, y_train)

    print('Validation: ')
    _ = baseline.evaluate(x_val, y_val)


def run_mlp(load_path=None, train=True, epochs=3000, dataset_path='../_data/dataset/artificial',
            verbose_train=True, save_path='../_data/weights/', **kwargs):
    dr = ArtificialDatasetReader(dataset_path=dataset_path)
    x_train, y_train, x_val, y_val, _, _ = dr.get_dataset(img_size=64)

    model = MLP((32, 32),  **kwargs)

    if load_path:
        model.load_weights(load_path)

    print('Running MLP')

    if train:
        print('Start training')
        result = model.train_epochs(x_train, y_train, x_val, y_val, epochs, 64, frequency=1,
                                    verbose=verbose_train)
        result.plot()
        model.save_weights(save_path)

    print('Train: ')
    y = model(x_train)
    _ = mse(y_train, y)
    _ = psnr(y_train, y)

    print('Validation: ')
    y = model(x_val)
    _ = mse(y_val, y)
    _ = psnr(y_val, y)


def run_residual_dense_network(load_path=None, train=True, epochs=500, verbose_train=True,
                               dataset_path='../_data/dataset/artificial', save_path='../_data/weights/', **kwargs):
    dr = ArtificialDatasetReader(dataset_path=dataset_path)
    x_train, y_train, x_val, y_val, _, _ = dr.get_dataset(img_size=64)

    model = ResidualDenseNetwork((32, 32), **kwargs)

    if load_path:
        model.load_weights(load_path)

    print('Running ResidualDenseNetwork')

    if train:
        print('Start training')
        result = model.train_epochs(x_train, y_train, x_val, y_val, epochs, 64, frequency=1,
                                    verbose=verbose_train)
        result.plot()
        model.save_weights(save_path)

    print('Train: ')
    y = model(x_train)
    _ = mse(y_train, y)
    _ = psnr(y_train, y)

    print('Validation: ')
    y = model(x_val)
    _ = mse(y_val, y)
    _ = psnr(y_val, y)
