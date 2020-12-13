from invoke import task
import os
import json
import sys

from super_resolution.main import run_baseline


@task
def evaluate(context, dataset_path='_data/dataset/artificial'):
    run_baseline(dataset_path)