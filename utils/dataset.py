import numpy as np
import os


class DatasetReader:
    def __init__(self, dataset=1, dataset_path=''):
        self.dataset_path = dataset_path
        if dataset == 1:
            self.load_dataset1()
        elif dataset == 2:
            self.load_dataset2()

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
