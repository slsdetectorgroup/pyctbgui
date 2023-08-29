import filecmp
import os
from pathlib import Path

import pytest

from pyctbgui.utils.numpyWriter.npy_writer import NumpyFileManager
import numpy as np

from pyctbgui.utils.numpyWriter.npz_writer import NpzFileWriter

prefix = Path('tests/.tmp/')


def __clean_tmp_dir(path=prefix):
    if Path.is_dir(path):
        for file in os.listdir(path):
            Path.unlink(path / file)
    else:
        Path.mkdir(path)


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
    npw = NumpyFileManager(prefix / 'tmp.npy', (2, 2, 2), np.clongdouble, bufferMax=3)
    assert npw.bufferCount == 0
    npw.addFrame(np.ones((2, 2, 2), dtype=np.clongdouble))
    assert npw.bufferCount == 1
    npw.addFrame(np.ones((2, 2, 2), dtype=np.clongdouble))
    assert npw.bufferCount == 2
    npw.addFrame(np.ones((2, 2, 2), dtype=np.clongdouble))
    assert npw.bufferCount == 3
    npw.addFrame(np.ones((2, 2, 2), dtype=np.clongdouble))
    assert npw.bufferCount == 0
    npw.flushBuffer(strict=True)
    assert np.array_equal(np.load(prefix / 'tmp.npy'), np.ones((4, 2, 2, 2), dtype=np.clongdouble))
    del npw

    np.save(prefix / 'tmp2.npy', np.ones((4, 2, 2, 2), dtype=np.clongdouble))
    assert filecmp.cmp(prefix / 'tmp.npy', prefix / 'tmp2.npy')


def test_init_parameters():
    __clean_tmp_dir()
    with pytest.raises(TypeError):
        NumpyFileManager()

    # test opening file that does not exist without the necessary parameters
    with pytest.raises(AssertionError):
        NumpyFileManager(prefix / 'abaababababa.npyx')
    with pytest.raises(AssertionError):
        NumpyFileManager(prefix / 'abaababababa.npyx2', frameShape=(12, 34))
    with pytest.raises(AssertionError):
        NumpyFileManager(prefix / 'abaababababa.npyx3', dtype=np.int64)

    # opening new file with required parameters (this should work)
    NumpyFileManager(prefix / 'abaababababa.npyx3', dtype=np.int64, frameShape=(6, 6))
    assert Path.is_file(prefix / 'abaababababa.npyx3')

    # re-opening the same file
    NumpyFileManager(prefix / 'abaababababa.npyx3', dtype=np.int64, frameShape=(6, 6))
    NumpyFileManager(prefix / 'abaababababa.npyx3')

    # re-opening the file with wrong parameters
    with pytest.raises(AssertionError):
        NumpyFileManager(prefix / 'abaababababa.npyx3', frameShape=(6, 2))
    with pytest.raises(AssertionError):
        NumpyFileManager(prefix / 'abaababababa.npyx3', dtype=np.int32)
    with pytest.raises(AssertionError):
        NumpyFileManager(prefix / 'abaababababa.npyx3', dtype=np.float32, frameShape=(5, 5))

    # test resetting an existing file
    npw = NumpyFileManager(prefix / 'tmp4.npy', dtype=np.float32, frameShape=(5, 5))
    npw.addFrame(np.ones((5, 5), dtype=np.float32))
    npw.close()
    assert np.load(prefix / 'tmp4.npy').shape == (1, 5, 5)
    npw = NumpyFileManager(prefix / 'tmp4.npy', dtype=np.int64, frameShape=(7, 7), resetFile=True)
    npw.flushBuffer(strict=True)
    assert np.load(prefix / 'tmp4.npy').shape == (0, 7, 7)

    # test adding frames with the wrong shape to an existing file
    with pytest.raises(AssertionError):
        npw.addFrame(np.ones((9, 4, 4)))


def test_read_frames():
    __clean_tmp_dir()
    rng = np.random.default_rng(seed=42)
    arr = rng.random((10000, 20, 20))
    npw = NumpyFileManager(prefix / 'tmp.npy', frameShape=(20, 20), dtype=arr.dtype)
    for frame in arr:
        npw.addFrame(frame)
    npw.flushBuffer(strict=True)
    assert np.array_equal(npw.readFrames(50, 100), arr[50:150])
    assert np.array_equal(npw.readFrames(0, 1), arr[0:1])
    assert np.array_equal(npw.readFrames(0, 10000), arr)
    assert np.array_equal(npw.readFrames(9999, 1), arr[9999:])
    assert np.array_equal(npw.readFrames(499, 3000), arr[499:3499])


@pytest.mark.parametrize('compressed', [True, False])
def test_incremental_npz(compressed):
    __clean_tmp_dir()
    arr1 = np.ones((10, 5, 5))
    arr2 = np.zeros((10, 5, 5), dtype=np.int32)
    arr3 = np.ones((10, 5, 5), dtype=np.float32)
    with NpzFileWriter(prefix / 'tmp.npz', 'w', compress_file=compressed) as npz:
        npz.addArray('adc', arr1)
        npz.addArray('tx', arr2)
        npz.addArray('signal', arr3)

    npzFile = np.load(prefix / 'tmp.npz')
    assert sorted(npzFile.files) == sorted(('adc', 'tx', 'signal'))
    assert np.array_equal(npzFile['adc'], np.ones((10, 5, 5)))
    assert np.array_equal(npzFile['tx'], np.zeros((10, 5, 5), dtype=np.int32))
    assert np.array_equal(npzFile['signal'], np.ones((10, 5, 5), dtype=np.float32))
    if compressed:
        np.savez_compressed(prefix / 'tmp2.npz', adc=arr1, tx=arr2, signal=arr3)
    else:
        np.savez(prefix / 'tmp2.npz', adc=arr1, tx=arr2, signal=arr3)

    assert filecmp.cmp(prefix / 'tmp2.npz', prefix / 'tmp.npz')


@pytest.mark.parametrize('compressed', [True, False])
def test_zipping_npy_files(compressed):
    __clean_tmp_dir()
    data = {
        'arr1': np.ones((10, 5, 5)),
        'arr2': np.zeros((10, 5, 5), dtype=np.int32),
        'arr3': np.ones((10, 5, 5), dtype=np.float32)
    }
    filePaths = [prefix / (file + '.npy') for file in data.keys()]
    for file in data:
        np.save(prefix / (file + '.npy'), data[file])
    NpzFileWriter.zipNpyFiles(prefix / 'file.npz', filePaths, list(data.keys()), compressed=compressed)
    npz = np.load(prefix / 'file.npz')
    assert npz.files == list(data.keys())
    for file in data:
        assert np.array_equal(npz[file], data[file])

    if compressed:
        np.savez_compressed(prefix / 'numpy.npz', **data)
    else:
        np.savez(prefix / 'numpy.npz', **data)

    numpyz = np.load(prefix / 'numpy.npz')
    for file in data:
        assert np.array_equal(numpyz[file], npz[file])
    assert npz.files == numpyz.files

    # different files :(
    # assert filecmp.cmp(prefix / 'numpy.npz', prefix / 'file.npz')


def test_compression():
    pass


@pytest.mark.parametrize('compressed', [True, False])
@pytest.mark.parametrize('isPath', [True, False])
@pytest.mark.parametrize('deleteOriginals', [True, False])
def test_delete_files(compressed, isPath, deleteOriginals):
    __clean_tmp_dir()
    data = {
        'arr1': np.ones((10, 5, 5)),
        'arr2': np.zeros((10, 5, 5), dtype=np.int32),
        'arr3': np.ones((10, 5, 5), dtype=np.float32)
    }
    filePaths = [prefix / (file + '.npy') for file in data.keys()]
    for file in data:
        np.save(prefix / (file + '.npy'), data[file])
    path = prefix / 'file.npz'
    path = str(path) if isPath else path
    NpzFileWriter.zipNpyFiles(path,
                              filePaths,
                              list(data.keys()),
                              deleteOriginals=deleteOriginals,
                              compressed=compressed)
    if deleteOriginals:
        for file in filePaths:
            assert not Path.exists(file)
    else:
        for file in filePaths:
            assert Path.exists(file)
