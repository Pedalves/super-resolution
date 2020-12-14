import pytest

from dataset.artificial_dataset import ArtificialDatasetReader

import logging

LOGGER = logging.getLogger(__name__)


def test_load_model_shape():
    dr = ArtificialDatasetReader('_data/dataset/artificial/')

    well_1 = dr.load_dataset(
            height=1040,
            length=7760,
            vel_path='mod_vp_05_nx7760_nz1040.bin',
            well_path='IMG1_dip_FINAL_REF_model_1_true.bin',
            edges=(1000, 6900)
    )

    assert well_1.shape == (789, 5900)

    well_2 = dr.load_dataset(
        height=1216,
        length=6912,
        vel_path='pluto_VP_SI_02.bin',
        well_path='IMG1_dip_FINAL_REF_model_2_true.bin',
        edges=(1400, 5550)
    )

    assert well_2.shape == (1076, 4150)


def test_remove_water():
    dr = ArtificialDatasetReader('_data/dataset/artificial/')
    _, y_train, _, y_val, _, y_test = dr.get_dataset(img_size=64)

    # water blocks have std == 0
    for y in y_train:
        assert y.std() > 0
    for y in y_val:
        assert y.std() > 0
    for y in y_train:
        assert y.std() > 0


def test_get_rgb_data():
    dr = ArtificialDatasetReader('_data/dataset/artificial/')
    x, y, _, _, _, _ = dr.get_dataset(img_size=64, as_cmap=True)

    assert x.shape[-1] == 3
    assert y.shape[-1] == 3


def test_img_size():
    dr = ArtificialDatasetReader('_data/dataset/artificial/')

    for size in [32, 64, 256]:
        x, y, _, _, _, _ = dr.get_dataset(img_size=size)

        for _x in x:
            # we need to divide x by 2, since we are down sampling it
            assert _x.shape[-1] == size//2
        for _y in y:
            assert _y.shape[-1] == size


def test_normalize():
    dr = ArtificialDatasetReader('_data/dataset/artificial/')

    well_1 = dr.load_dataset(
        height=1040,
        length=7760,
        vel_path='mod_vp_05_nx7760_nz1040.bin',
        well_path='IMG1_dip_FINAL_REF_model_1_true.bin',
        edges=(1000, 6900),
        normalize=False
    )

    assert (well_1.min() != 0.0 or well_1.max() != 1.0)

    well_1 = dr.load_dataset(
        height=1040,
        length=7760,
        vel_path='mod_vp_05_nx7760_nz1040.bin',
        well_path='IMG1_dip_FINAL_REF_model_1_true.bin',
        edges=(1000, 6900),
        normalize=True
    )

    assert (well_1.min() == 0.0 or well_1.max() == 1.0)
