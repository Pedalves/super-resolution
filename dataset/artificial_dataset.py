import os

import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import downscale_local_mean
import cv2


class ArtificialDatasetReader:
    """
    This class is responsible for loading and pre-processing _data from the artificial dataset
    """

    def __init__(self, dataset_path='../_data/dataset/artificial'):
        self.dataset_path = dataset_path
        self.well_1 = self.load_dataset(
            height=1040,
            length=7760,
            vel_path='mod_vp_05_nx7760_nz1040.bin',
            well_path='IMG1_dip_FINAL_REF_model_1_true.bin',
            edges=(1000, 6900)
        )
        self.well_2 = self.load_dataset(
            height=1216,
            length=6912,
            vel_path='pluto_VP_SI_02.bin',
            well_path='IMG1_dip_FINAL_REF_model_2_true.bin',
            edges=(1400, 5550)
        )

    def get_dataset(self, scale=2, img_size=64, as_cmap=False):
        """
        breaks well image into 'img_size' x 'img_size' images and downscale them by 'scale'
        :param scale: down sampling scale factor
        :param img_size: size of the images in the dataset
        :param as_cmap: boolean to convert the dataset to a 3 channel RGB image
        :return: x_train, y_train, x_val, y_val, x_test, y_test, with
        rescaled cropped images as x and original cropped images as y
        """
        np.random.seed(0)

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

        # X train, val and test
        x_train = x[:int(len(x) * 0.5)]
        x_val = x[int(len(x) * 0.5): int(len(x) * 0.7)]
        x_test = x[int(len(x) * 0.7):]

        # Y train, val and test
        y_train = data[:int(len(data) * 0.5)]
        y_val = data[int(len(data) * 0.5): int(len(data) * 0.7)]
        y_test = data[int(len(data) * 0.7):]

        if as_cmap:
            return self.get_cmap(x_train), self.get_cmap(y_train), self.get_cmap(x_val), self.get_cmap(
                y_val), self.get_cmap(x_test), self.get_cmap(y_test)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def load_dataset(self, height, length, vel_path, well_path, edges, normalize=False):
        """
        load well's data from the artificial dataset
        :param height: well height
        :param length: well length
        :param vel_path: vel bin file name
        :param well_path: well bin file name
        :param edges: tuple containing lower and upper edges limits
        :param normalize: boolean to normalize the dataset
        :return: numpy array with well's data
        """
        with open(os.path.join(self.dataset_path, vel_path), 'rb') as f:
            vel_data = np.fromfile(f, dtype=np.float32)
            vel = np.reshape(vel_data, [length, height])
            vel = vel.T

            # Cutting edges with repeated _data
            vel = vel[:, edges[0]:edges[1]]

        with open(os.path.join(self.dataset_path, well_path), 'rb') as f:
            well_data = np.fromfile(f, dtype=np.float32)
            well = np.reshape(well_data, [length, height])
            well = well.T

            # Cutting edges with repeated _data
            well = well[:, edges[0]:edges[1]]

            # Clean Water
            i_max = 0
            min_vel = np.min(vel)
            for i in range(vel.shape[0]):
                for j in range(vel.shape[1]):
                    if (vel[i, j] <= min_vel) and (i > i_max):
                        i_max = i
            well = well[i_max:, :]

            # Normalizing
            if normalize:
                well = self._normalize(well)

        return well

    @staticmethod
    def _normalize(well):
        """
        min max normalization
        :param well: numpy array containing the well _data
        :return: normalized well
        """
        well -= np.min(well)
        well /= np.max(well) - np.min(well)

        return well

    @classmethod
    def get_cmap(cls, array):
        """
        convert an 'array' to a 3 channel RGB image
        :param array: 2 dimension numpy array
        :return: 3 channel RGB numpy array
        """
        array_rgb = []
        for i in range(len(array)):
            cmap = plt.get_cmap('seismic')
            norm = plt.Normalize()
            output = cmap(norm(array[i]))[:, :, :3]
            array_rgb.append(output)

        return np.array(array_rgb)
