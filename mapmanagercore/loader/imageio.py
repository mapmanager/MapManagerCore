import json
import pandas as pd

from mapmanagercore.config import Metadata
from .base import ImageLoader, Loader
from typing import Iterator, Union
import numpy as np


class MultiImageLoader(Loader):
    """
    Class for building an MultiImageLoader.
    """

    def __init__(self, lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(), points: Union[str, pd.DataFrame] = pd.DataFrame()):
        super().__init__(lineSegments, points)
        self._images = {}
        self._metadata = {}

    def imread(path: str) -> ImageLoader:
        """
        Load an image using imageio.imread.

        Args:
          path (str): The path to the image.
        """
        from imageio import imread
        return _MultiImageLoader(imread(path))

    def read(self, path, time: int = 0, channel: int = 0):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): The path to the image file.
          time (int): The time index.
          channel (int): The channel index.
        """
        from imageio import imread
        if time not in self._images:
            self._images[time] = []

        self._images[time].append([channel, imread(path)])

    def readMetadata(self, metadata: Union[Metadata, str], time: int = 0):
        """
        Set the metadata for the given time index.

        Args:
          time (int): The time index.
          metadata (Metadata): The metadata.
        """

        if isinstance(metadata, str):
            with open(metadata, "r") as metadataFile:
                metadata = json.load(metadataFile)

        self._metadata[time] = metadata

    def images(self) -> ImageLoader:
        images = {}

        for time, values in self._images.items():
            if not (time in self._metadata):
                raise ValueError(f"Metadata not found for time point {time}")

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
        return self._imagesSrcs.keys()

    def _images(self, t: int) -> np.ndarray:
        return self._imagesSrcs[t]

    def metadata(self, t: int) -> Metadata:
        return self._metadata[t] if t in self._metadata else Metadata()
