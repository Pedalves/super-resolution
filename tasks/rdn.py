from invoke import task
import os
import json
import sys

from super_resolution.main import run_residual_dense_network


@task
def train(context, epochs, dataset_path='_data/dataset/artificial', save_path='_data/weights/', **kwargs):
    if isinstance(epochs, str):
        epochs = int(epochs)
    run_residual_dense_network(epochs=epochs, train=True, verbose_train=True,
                               dataset_path=dataset_path, save_path=save_path, **kwargs)


@task
def evaluate(context, load_path, dataset_path='_data/dataset/artificial', **kwargs):
    run_residual_dense_network(load_path=load_path, train=False, dataset_path=dataset_path, **kwargs)

