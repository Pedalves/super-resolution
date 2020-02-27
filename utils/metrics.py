import math

import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error


def _psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return (20 * math.log10(PIXEL_MAX)) - (10 * math.log10(mse))


def psnr(y, y_pred, log=True):
    psnr_sum = 0

    for i in range(len(y)):
        psnr_sum += _psnr(y[i], y_pred[i])

    if log:
        print(f"Mean PSNR {psnr_sum / len(y)}")

    return psnr_sum / len(y)


def mse(y, y_pred, log=True):
    mse_sum = 0

    for i in range(len(y)):
        mse_sum += mean_squared_error(y[i], y_pred[i])

    if log:
        print(f"Mean MSE {mse_sum / len(y)}")

    return mse_sum / len(y)


def ssim(y, y_pred, log=True):
    ssim_sum = 0

    for i in range(len(y)):
        ssim_sum += structural_similarity(y[i], y_pred[i])

    if log:
        print(f"Mean SSIM {ssim_sum / len(y)}")

    return ssim_sum / len(y)
