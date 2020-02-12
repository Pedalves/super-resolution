import numpy as np
import os
from skimage.transform import downscale_local_mean


class DatasetReader:
    def __init__(self, dataset=1, dataset_path=''):
        self.dataset_path = dataset_path
        if dataset == 1:
            self.load_dataset1()
        elif dataset == 2:
            self.load_dataset2()

    def get_dataset(self, scale=2, img_size=64):
        """
        breaks well image into 'img_size' x 'img_size' images and downscale them by 'scale'
        :param scale: down sampling scale factor
        :param img_size: size of the images in the dataset
        :return: rescaled cropped images(x) and original cropped images(y)
        """
        shape = (img_size * (self.well.shape[0] // img_size), img_size * (self.well.shape[1] // img_size))

        data_coord = [(img_size * i, img_size * j) for i in range(shape[0] // img_size) for j in
                      range(shape[1] // img_size)]

        y = [self.well[coord[0]: coord[0] + img_size, coord[1]: coord[1] + img_size] for coord in data_coord]
        x = [downscale_local_mean(y, (scale, scale)) for y in y]

        return x, y

    def load_dataset1(self, gain=False, normalize=False):
        height = 1040
        length = 7760

        self.datachoice = '1'

        if gain:
            with open(os.path.join(self.dataset_path, 'IMG1_dip_FINAL_REF_model_1_true_gain.bin'), 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                self.well = np.reshape(data, [length, height])
                self.well = self.well.T
        else:
            with open(os.path.join(self.dataset_path, 'IMG1_dip_FINAL_REF_model_1_true.bin'), 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                self.well = np.reshape(data, [length, height])
                self.well = self.well.T

        with open(os.path.join(self.dataset_path, 'mod_vp_05_nx7760_nz1040.bin'), 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
            self.vel = np.reshape(data, [length, height])
            self.vel = self.vel.T

        # Cutting edges with repeated data
        self.well = self.well[:, 1000:6900]
        self.vel = self.vel[:, 1000:6900]

        # Masking the salt: it has specific velocity on the image
        self.mask = np.ma.masked_where(self.vel == 4450, self.vel)
        self.mask = self.mask.mask.astype(np.int)

        # Normalizing
        if normalize:
            self.well -= np.min(self.well)
            self.well /= np.max(self.well)

    def load_dataset2(self, gain=False, normalize=False):
        height = 1216
        length = 6912

        self.datachoice = '2'

        if gain:
            with open(os.path.join(self.dataset_path, 'IMG1_dip_FINAL_REF_model_2_true_gain.bin'), 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                self.well = np.reshape(data, [length, height])
                self.well = self.well.T
        else:
            with open(os.path.join(self.dataset_path, 'IMG1_dip_FINAL_REF_model_2_true.bin'), 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                self.well = np.reshape(data, [length, height])
                self.well = self.well.T

        with open(os.path.join(self.dataset_path, 'pluto_VP_SI_02.bin'), 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
            self.vel = np.reshape(data, [length, height])
            self.vel = self.vel.T

        # Cutting edges with repeated data
        self.well = self.well[:, 1400:5550]
        self.vel = self.vel[:, 1400:5550]

        # Masking the salt: it has specific velocity on the image
        self.mask = np.ma.masked_where(self.vel == 4511.04, self.vel)
        self.mask = self.mask.mask.astype(np.int)

        # Normalizing
        if normalize:
            self.well -= np.min(self.well)
            self.well /= np.max(self.well)
