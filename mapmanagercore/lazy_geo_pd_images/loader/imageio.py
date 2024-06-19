import json
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from .base import ImageLoader
from typing import Iterator, Union
import numpy as np

class MultiImageLoader(ImageLoader):
    """
    Class for building an MultiImageLoader.
    """

    def __init__(self):
        super().__init__()
        self._images = {}
        self.paths = [] # for logging only
        
    def __str__(self):
        return f"Multi image Loader paths: {self.paths}"

    def read(self, path: Union[str, np.ndarray], time: int = 0, channel: int = 0):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): Either the path to the image file or a np array.
          time (int): The time index.
          channel (int): The channel index.
        """
        from imageio import imread
        if time not in self._images:
            self._images[time] = []

        if isinstance(path, str):
            imgData = imread(path)
        else:
            imgData = path

        self._images[time].append([channel, imgData])
        self.paths.append([time, channel, path])

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

    def build(self) -> ImageLoader:
        images = {}

        for time, values in self._images.items():
            # if not (time in self._metadata):
            #     raise ValueError(f"Metadata not found for time point {time}")

            maxChannel = max(channel for channel, _ in values) + 1
            maxSlice, maxX, maxY = values[0][1].shape
            dimensions = [maxChannel, maxSlice, maxX, maxY]
            images[time] = np.zeros(dimensions, dtype=np.uint16)
            for channel, image in values:
                images[time][channel] = image

        return _MultiImageLoader(images, self._metadata)


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
