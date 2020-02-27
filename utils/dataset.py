import os

import numpy as np
from skimage.transform import downscale_local_mean

np.random.seed(0)


class DatasetReader:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.load_dataset1()
        self.load_dataset2()

    def get_dataset(self, scale=2, img_size=64):
        """
        breaks well image into 'img_size' x 'img_size' images and downscale them by 'scale'
        :param scale: down sampling scale factor
        :param img_size: size of the images in the dataset
        :return: rescaled cropped images(x) and original cropped images(y)
        """
        shape1 = (img_size * (self.well_1.shape[0] // img_size), img_size * (self.well_1.shape[1] // img_size))
        data_coord1 = [(img_size * i, img_size * j) for i in range(shape1[0] // img_size) for j in
                       range(shape1[1] // img_size)]

        shape2 = (img_size * (self.well_2.shape[0] // img_size), img_size * (self.well_2.shape[1] // img_size))
        data_coord2 = [(img_size * i, img_size * j) for i in range(shape2[0] // img_size) for j in
                       range(shape2[1] // img_size)]

        data = [self.well_1[coord[0]: coord[0] + img_size, coord[1]: coord[1] + img_size] for coord in data_coord1] + [
            self.well_2[coord[0]: coord[0] + img_size, coord[1]: coord[1] + img_size] for coord in data_coord2]

        np.random.shuffle(data)

        x = [downscale_local_mean(y, (scale, scale)) for y in data]

        x_train = x[:int(len(x) * 0.5)]
        x_val = x[int(len(x) * 0.5): int(len(x) * 0.7)]
        x_test = x[int(len(x) * 0.7):]

        y_train = data[:int(len(data) * 0.5)]
        y_val = data[int(len(data) * 0.5): int(len(data) * 0.7)]
        y_test = data[int(len(data) * 0.7):]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def load_dataset1(self, normalize=False):
        height = 1040
        length = 7760

        with open(os.path.join(self.dataset_path, 'IMG1_dip_FINAL_REF_model_1_true.bin'), 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
            self.well_1 = np.reshape(data, [length, height])
            self.well_1 = self.well_1.T

            # Cutting edges with repeated data
            self.well_1 = self.well_1[:, 1000:6900]

            # Normalizing
            if normalize:
                self.well_1 -= np.min(self.well_1)
                self.well_1 /= np.max(self.well_1)

    def load_dataset2(self, normalize=False):
        height = 1216
        length = 6912

        with open(os.path.join(self.dataset_path, 'IMG1_dip_FINAL_REF_model_2_true.bin'), 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
            self.well_2 = np.reshape(data, [length, height])
            self.well_2 = self.well_2.T

            # Cutting edges with repeated data
            self.well_2 = self.well_2[:, 1400:5550]

            # Normalizing
            if normalize:
                self.well_2 -= np.min(self.well_2)
                self.well_2 /= np.max(self.well_2)
