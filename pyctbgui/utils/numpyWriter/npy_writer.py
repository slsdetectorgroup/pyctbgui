"""
Wrapper to be able to append frames to a numpy file

numpy header v1

- 6bytes                         \x93NUMPY
- 1 byte major version number    \x01
- 1 byte minor version number    \x00
- 2 bytes (unsigned short) HEADER_LEN length of header to follow
- Header as an ASCII dict terminated by \n padded with space \x20 to make sure
we get len(magic string) + 2 + len(length) + HEADER_LEN divisible with 64
Allocate enough space to allow for the data to grow
"""

import ast
import logging
import os
from pathlib import Path

import numpy as np


class NumpyFileManager:
    magic_str = np.lib.format.magic(1, 0)
    headerLength = np.uint16(128)
    FSEEK_FILE_END = 2
    BUFFER_MAX = 500

    def __init__(self,
                 file: str,
                 frameShape: tuple = None,
                 dtype=None,
                 resetFile: bool = False,
                 isFortranOrder: bool = False,
                 bufferMax=BUFFER_MAX):
        self.logger = logging.getLogger('npy_writer')

        self.bufferMax = bufferMax
        self.dtype = np.dtype(dtype)  # in case we pass a type like np.float32
        self.isFortranOrder = isFortranOrder
        self.frameShape = frameShape
        self.frameCount = 0
        fileExist = Path.is_file(Path(file))
        newFile = resetFile or not fileExist
        self.buffer = bytearray()
        self.bufferCount = 0
        if newFile:
            assert frameShape is not None
            assert dtype is not None
            self.file = open(file, 'wb+')
            self.updateHeader()

        else:
            self.file = open(file, 'rb+')
            self.file.seek(10)
            headerStr = self.file.read(np.uint16(self.headerLength - 10)).decode("UTF-8")
            header_dict = ast.literal_eval(headerStr)
            self.frameShape = header_dict['shape'][1:]
            if frameShape is not None:
                assert frameShape == self.frameShape, \
                    f"shape in arguments ({frameShape}) is not the same as the shape of the stored " \
                    f"file ({self.frameShape})"

            self.dtype = np.lib.format.descr_to_dtype(header_dict['descr'])
            if dtype is not None:
                assert dtype == self.dtype, \
                    f"dtype in argument ({dtype}) is not the same as the dtype of the stored file ({self.dtype})"

            self.frameCount = header_dict['shape'][0]

            self.isFortranOrder = header_dict['fortran_order']
            assert self.isFortranOrder == isFortranOrder, \
                f"isFortranOrder ({isFortranOrder}) in argument is not the same as fortran_order" \
                f" of the stored file ({self.isFortranOrder})"

        self.__frameSize = np.dtype(self.dtype).itemsize * np.prod(self.frameShape)

    def updateHeader(self):
        self.file.seek(0)
        header_dict = {
            'descr': np.lib.format.dtype_to_descr(self.dtype),
            'fortran_order': self.isFortranOrder,
            'shape': (self.frameCount, *self.frameShape)
        }
        np.lib.format.write_array_header_1_0(self.file, header_dict)

    def writeOneFrame(self, frame: np.ndarray):
        assert frame.shape == self.frameShape
        assert frame.dtype == self.dtype
        self.file.seek(0, self.FSEEK_FILE_END)
        self.frameCount += 1
        self.file.write(frame.tobytes())
        self.updateHeader()

    def addFrame(self, frame: np.ndarray):
        assert frame.shape == self.frameShape
        assert frame.dtype == self.dtype
        self.buffer.extend(frame.tobytes())
        self.frameCount += 1
        self.bufferCount += 1
        if self.bufferCount > self.bufferMax:
            self.flushBuffer()

    def flushBuffer(self, strict=False):
        if len(self.buffer) > 0:
            self.file.seek(0, self.FSEEK_FILE_END)
            self.file.write(bytes(self.buffer))
            self.buffer = bytearray()
            self.updateHeader()
            if strict:
                self.file.flush()
                os.fsync(self.file)
            self.bufferCount = 0

    def readFrames(self, frameStart: int, frameCount: int) -> np.ndarray:
        self.file.seek(self.headerLength + frameStart * self.__frameSize)
        data = self.file.read(frameCount * self.__frameSize)
        arr = np.frombuffer(data, self.dtype).reshape([-1, *self.frameShape])
        self.logger.warning(f'stored array has less frames ({arr.shape[0]}) than frameCount argument ({frameCount})')
        return arr

    def close(self):
        self.flushBuffer()
        self.file.close()

    def __del__(self):
        if hasattr(self, 'file') and not self.file.closed:
            self.close()
