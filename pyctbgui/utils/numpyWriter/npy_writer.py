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
import zipfile
from pathlib import Path

import numpy as np


class NumpyFileManager:
    """
    class used to read and write into .npy files that can't be loaded completely into memory
    """
    magic_str = np.lib.format.magic(1, 0)
    headerLength = np.uint16(128)
    FSEEK_FILE_END = 2
    BUFFER_MAX = 500

    def __init__(
        self,
        file: str | Path | zipfile.ZipExtFile,
        frameShape: tuple = None,
        dtype=None,
        resetFile: bool = False,
        bufferMax=BUFFER_MAX,
    ):
        """
        initiates a NumpyFileManager class for reading or writing bytes directly to/from a .npy file
        @param file: path to the file to open or create
        @param frameShape: shape of the frame ex: (5000,) for waveforms or (400,400) for image
        @param dtype: type of the numpy array's header
        @param resetFile: set to True if you wish to clear and write on an existing file
        @param bufferMax: maximum length of the buffer that accumulates frames before flushing it  and calling syscalls
        """
        self.logger = logging.getLogger('npy_writer')

        self.bufferMax = bufferMax
        self.dtype = np.dtype(dtype)  # in case we pass a type like np.float32
        self.frameShape = frameShape
        self.frameCount = 0
        if not isinstance(file, zipfile.ZipExtFile):
            fileExist = Path.is_file(Path(file))
        else:
            fileExist = True

        newFile = resetFile or not fileExist
        self.buffer = bytearray()
        self.bufferCount = 0
        # if newFile frameShape and dtype should be present
        if newFile:
            assert frameShape is not None
            assert dtype is not None
            # create/clear the file with mode wb+
            self.file = open(file, 'wb+')
            self.updateHeader()

        else:
            # opens file for read/write and check if the header of the file corresponds to the given function
            # arguments
            if not isinstance(file, zipfile.ZipExtFile):
                self.file = open(file, 'rb+')
            else:
                self.file = file

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

            assert not header_dict['fortran_order'], "fortran_order in the stored file is not False"

        self.__frameSize = np.dtype(self.dtype).itemsize * np.prod(self.frameShape)

    def updateHeader(self):
        """
        updates the header of the .npy file with the class attributes
        @note: fortran_order is always set to False
        """
        self.file.seek(0)
        header_dict = {
            'descr': np.lib.format.dtype_to_descr(self.dtype),
            'fortran_order': False,
            'shape': (self.frameCount, *self.frameShape)
        }
        np.lib.format.write_array_header_1_0(self.file, header_dict)

    def writeOneFrame(self, frame: np.ndarray):
        """
        write one frame without buffering
        @param frame: numpy array for a frame
        """
        assert frame.shape == self.frameShape
        assert frame.dtype == self.dtype
        self.file.seek(0, self.FSEEK_FILE_END)
        self.frameCount += 1
        self.file.write(frame.tobytes())
        self.updateHeader()

    def addFrame(self, frame: np.ndarray):
        """
        asserts if the frame is not in the correct format
        appends frame to the buffer
        flush the buffer once bufferMax is reached
        @param frame:
        @return:
        """
        assert frame.shape == self.frameShape
        assert frame.dtype == self.dtype
        self.buffer.extend(frame.tobytes())
        self.frameCount += 1
        self.bufferCount += 1
        if self.bufferCount > self.bufferMax:
            self.flushBuffer()

    def flushBuffer(self, strict=False):
        """
        write the buffer's bytes to the end of the file
        clears the buffer and update the .npy file header
        @param strict: set to True to use fsync() syscall to persist the file directly to disk without waiting to wait
        fot the kernel's buffer to reach io.DEFAULT_BUFFER_SIZE
        """
        if len(self.buffer) > 0:
            self.file.seek(0, self.FSEEK_FILE_END)
            self.file.write(bytes(self.buffer))
            self.buffer = bytearray()
            self.updateHeader()
            self.bufferCount = 0
        if strict:
            self.file.flush()
            os.fsync(self.file)

    def readFrames(self, frameStart: int, frameCount: int) -> np.ndarray:
        """
        read frames from .npy file without loading the whole file to memory with np.load
        @param frameStart: number of the frame to start reading from
        @param frameCount: number of frames to load
        @return: np.ndarray of frames of the shape [frameCount,*self.frameShape]
        """
        if frameCount == 0:
            return np.array([], dtype=self.dtype)
        self.file.seek(self.headerLength + frameStart * self.__frameSize)
        data = self.file.read(frameCount * self.__frameSize)
        arr = np.frombuffer(data, self.dtype).reshape([-1, *self.frameShape])
        if arr.shape[0] < frameCount:
            self.logger.warning(
                f'stored array has less frames ({arr.shape[0]}) than frameCount argument ({frameCount})')
        return arr

    def close(self):
        self.flushBuffer()
        self.file.close()

    def __del__(self):
        """
        in case the user forgot to close the file
        """
        if hasattr(self, 'file') and not self.file.closed:
            self.close()
