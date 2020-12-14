import pandas as pd
import cv2

from utils.metrics import mse, psnr


class Baseline:
    """"
    Bi-cubic interpolation baseline
    """
    def __init__(self, scale=2):
        """
        Class init
        :param scale: resolution increase rate
        """
        self.scale = scale

    def predict(self, x):
        """
        Increase resolution of a given array
        :param x: input values
        :return:
        """
        return [cv2.resize(img, (img.shape[0]*self.scale, img.shape[1]*self.scale)) for img in x]

    def evaluate(self, x, y):
        """
        Evaluate prediction's MSE and PSNR
        :param x: input values
        :param y: ground truth
        :return:
        """
        y_pred = self.predict(x)

        metrics = {'mse': [mse(y, y_pred)],
                   'psnr': [psnr(y, y_pred)]}

        return pd.DataFrame(metrics)
