# import json
import numpy as np

from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from .base import ImageLoader
from typing import Iterator, Union
from mapmanagercore.utils import getAutoContrast
from mapmanagercore.logger import logger

class MultiImageLoader(ImageLoader):
    """
    Class for building an MultiImageLoader.
    """

    def __init__(self):
        super().__init__()
        self._images = {}
        # self.paths = [] # for logging only
        
    def __str__(self):
        # return f"Multi image Loader paths: {self.paths}"
        retStr = ''
        for _timePoint, _imgList in self._images.items():
            for _channel, imgData in _imgList:
                retStr += f'   tp:"{_timePoint}" ch:{_channel} {imgData.shape} {imgData.dtype}\n'
        return retStr
    
    def read(self, path: Union[str, np.ndarray], time: int = 0, channel: int = 0):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): Either the path to the image file or a np array.
          time (int): The time index.
          channel (int): The channel index.
        """
        # TODO do not use imageio, use bioio
        # note, imageio is silently installed when scikit-image is installed
        # to update, see mapmanagercore.image_importers
        from imageio import imread

        if time not in self._images:
            self._images[time] = []
            # abb
            self._metadata[time] = Metadata()

        if isinstance(path, str):
            imgData = imread(path)
        else:
            imgData = path

        self._images[time].append([channel, imgData])
        
        # abb
        # logger.info(f'setting metadata time:{time} channel:{channel} imgData:{imgData.shape}')
        _metaData = Metadata()
        # shape of imgData
        _metaData.voxel.x = imgData.shape[2]
        _metaData.voxel.y = imgData.shape[1]
        _metaData.voxel.z = imgData.shape[0]

        # physicaal units (um)
        _metaData.physicalSize.x = 0.15
        _metaData.physicalSize.y = 0.15
        _metaData.physicalSize.z = 1

        # contrast
        _metaData.metadataContrast.color = 'TODO: fix this'
        _metaData.metadataContrast.minInt = int(np.min(imgData))
        _metaData.metadataContrast.maxInt = int(np.max(imgData))
        minContrast, maxContrast = getAutoContrast(imgData)
        _metaData.metadataContrast.minContrast = minContrast
        _metaData.metadataContrast.maxContrast = maxContrast

        # self._metadata[time].append([channel, _metaData])
        self._metadata[time] = _metaData
        
        # self.paths.append([time, channel, path])

    def build(self) -> ImageLoader:
        images = {}
        # metadata = {}

        for time, values in self._images.items():
            # if not (time in self._metadata):
            #     raise ValueError(f"Metadata not found for time point {time}")

            maxChannel = max(channel for channel, _ in values) + 1
            maxSlice, maxX, maxY = values[0][1].shape
            dimensions = [maxChannel, maxSlice, maxX, maxY]
            images[time] = np.zeros(dimensions, dtype=np.uint16)
            for channel, image in values:
                images[time][channel] = image

                # abb
                # metadata[time][channel] = Metadata()
                # logger.warning('TODO set MetaData()')

        return _MultiImageLoader(images, self._metadata)

    # abb TODO: this is never called?
    def readMetadata(self, metadata: Union[Metadata, str], time: int = 0):
        """
        Set the metadata for the given time index.

        Args:
          time (int): The time index.
          metadata (Metadata): The metadata.
        """

        if isinstance(metadata, str):
            with open(metadata, "r") as metadataFile:
                metadata = Metadata.from_json(metadataFile)

        self._metadata[time] = metadata

class _MultiImageLoader(ImageLoader):
    """
    A loader class for loading from imageio supported formats.
    """

    def __init__(self, images: dict[int, np.ndarray], metadata: dict[int, Metadata]):
        """
        Initialize the BaseImage class.

        Args:
          images (np.ndarray): [time, channel, slice].

        """
        super().__init__()
        self._imagesSrcs = images
        self._metadata = metadata

    def timePoints(self) -> Iterator[int]:
        """
        Returns an iterator over the time points of the images.

        Returns:
            An iterator that yields the time points of the images.
        """
        return self._imagesSrcs.keys()

    def _images(self, t: int) -> np.ndarray:
        return self._imagesSrcs[t]
