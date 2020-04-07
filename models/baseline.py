import pandas as pd
from skimage.transform import rescale

from utils.metrics import mse, psnr


class Baseline:
    def __init__(self, scale=2):
        self.scale = scale

    def predict(self, x):
        return [rescale(img, self.scale, anti_aliasing=False, order=3) for img in x]

    def evaluate(self, x, y):
        y_pred = self.predict(x)

        metrics = {'mse': [mse(y, y_pred)],
                   'psnr': [psnr(y, y_pred)]}

        return pd.DataFrame(metrics)
