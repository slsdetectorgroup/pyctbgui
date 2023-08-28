import pathlib
import shutil
import zipfile
import io

import numpy as np


class NpzFileWriter:
    """
    Write data to npz file incrementally rather than compute all and write
    once, as in ``np.save``. This class can be used with ``contextlib.closing``
    to ensure closed after usage.
    """

    def __init__(self, tofile: str, mode: str = 'x', compress_file=False):
        """
        :param tofile: the ``npz`` file to write
        :param mode: must be one of {'x', 'w', 'a'}. See
               https://docs.python.org/3/library/zipfile.html for detail
        """
        assert mode in 'xwa', str(mode)
        self.compression = zipfile.ZIP_DEFLATED if compress_file else zipfile.ZIP_STORED
        self.tofile = tofile
        self.mode = mode
        self.file = None

    def openFile(self):
        self.file = zipfile.ZipFile(self.tofile, mode=self.mode, compression=self.compression)

    def __enter__(self):
        self.openFile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def addArray(self, key: str, data: np.ndarray | bytes) -> None:
        """
        Same as ``self.write`` but overwrite existing data of name ``key``.

        :param key: the name of data to write
        :param data: the data
        """
        key += '.npy'
        with io.BytesIO() as cbuf:
            np.save(cbuf, data)
            cbuf.seek(0)
            with self.file.open(key, mode=self.mode, force_zip64=True) as outfile:
                shutil.copyfileobj(cbuf, outfile)

    @staticmethod
    def zipNpyFiles(filename: str, files: list[str | pathlib.Path], compressed=False):
        compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED

        with zipfile.ZipFile(filename, mode='w', compression=compression) as zipf:
            for file in files:
                zipf.write(file)

    def close(self):
        if self.tofile is not None:
            self.file.close()
            self.tofile = None

    def __del__(self):
        self.close()
