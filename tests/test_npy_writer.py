import filecmp
import os
from pathlib import Path

from pyctbgui.utils.numpyWriter.npy_writer import NumpyFileManager
import numpy as np

prefix = Path('tests/.tmp/')


def __clean_tmp_dir(path=prefix):
    for file in os.listdir(path):
        Path.unlink(path / file)


def test_create_new_file():
    __clean_tmp_dir()
    npw = NumpyFileManager(prefix / 'tmp.npy', (400, 400), np.int32)
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.addFrame(np.ones([400, 400], dtype=np.int32))
    npw.close()

    arr = np.load(prefix / 'tmp.npy')

    assert arr.dtype == np.int32
    assert arr.shape == (4, 400, 400)
    assert np.array_equal(arr, np.ones([4, 400, 400], dtype=np.int32))

    np.save(prefix / 'tmp2.npy', np.ones([4, 400, 400], dtype=np.int32))
    assert filecmp.cmp(prefix / 'tmp.npy', prefix / 'tmp2.npy')


def test_open_old_file():
    __clean_tmp_dir()
    npw = NumpyFileManager(prefix / 'tmp.npy', (4000, ), np.float32)
    npw.addFrame(np.ones(4000, dtype=np.float32))
    npw.addFrame(np.ones(4000, dtype=np.float32))
    npw.flushBuffer(strict=True)
    npw.close()
    npw2 = NumpyFileManager(prefix / 'tmp.npy')
    assert npw2.frameCount == 2
    assert npw2.frameShape == (4000, )
    assert npw2.dtype == np.float32
    assert len(npw2.buffer) == 0
    npw2.addFrame(np.ones(4000, dtype=np.float32))
    del npw2
    np.save(prefix / 'tmp2.npy', np.ones([3, 4000], dtype=np.float32))
    assert filecmp.cmp(prefix / 'tmp.npy', prefix / 'tmp2.npy')


def test_buffer():
    __clean_tmp_dir()


def test_init_parameters():
    pass


def test_reset_file():
    pass
