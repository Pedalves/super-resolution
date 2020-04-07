import numpy as np
from numpy.fft import fft2, fftshift


def get_spectogram(data):
        return np.abs(fftshift(fft2(data)))

