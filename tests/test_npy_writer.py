from pyctbgui.utils.numpyWriter.npy_writer import NumpyFileManager
import numpy as np


def test_create_new_file():
    npw = NumpyFileManager('tmp.npy', (400, 400), np.int32)
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.close()

    arr = np.load('tmp.npy')

    assert arr.dtype == np.int32
    assert arr.shape == (4, 400, 400)
    assert np.array_equal(arr, np.ones([4, 400, 400]))


def read_old_file():
    pass


def append_to_old_file():
    pass


def test_init_parameters():
    pass


def test_reset_file():
    pass
