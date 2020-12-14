import pytest
import numpy as np

from utils.metrics import psnr, mse


def test_mse():
   assert mse(np.array([[1], [1], [1]]), np.array([[0], [0], [0]]), verbose=False) == 1.0


def test_mse_all_equal():
    assert mse(np.array([[1], [1], [1]]), np.array([[1], [1], [1]]), verbose=False) == 0.0


def test_psnr():
   assert psnr(np.array([[1], [1], [1]]), np.array([[0], [0], [0]]), verbose=False) == 0.0


def test_psnr_all_equal():
    assert psnr(np.array([[1], [1], [1]]), np.array([[1], [1], [1]]), verbose=False) == 100.0
