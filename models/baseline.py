import pandas as pd
import cv2

from utils.metrics import mse, psnr


class Baseline:
    """"
    Bi-cubic interpolation baseline
    """
    def __init__(self, scale=2):
        self.scale = scale

    def predict(self, x):
        return [cv2.resize(img, (img.shape[0]*self.scale, img.shape[1]*self.scale)) for img in x]

    def evaluate(self, x, y):
        y_pred = self.predict(x)

        metrics = {'mse': [mse(y, y_pred)],
                   'psnr': [psnr(y, y_pred)]}

        return pd.DataFrame(metrics)
