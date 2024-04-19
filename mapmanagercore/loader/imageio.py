import pandas as pd

from mapmanagercore.config import Metadata
from .base import ImageLoader, Loader
from typing import Dict, List, Tuple, Union
import numpy as np
import zarr


class MultiImageLoader(Loader):
    """
    Class for building an MultiImageLoader.
    """

    def __init__(self, lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(), points: Union[str, pd.DataFrame] = pd.DataFrame(), metadata: Union[str, Metadata] = Metadata()):
        super().__init__(lineSegments, points, metadata)
        self._images = []

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
        self._images.append([time, channel, imread(path)])

    def images(self) -> ImageLoader:
        maxChannel = max(channel for _, channel, _ in self._images) + 1
        maxSlice, maxX, maxY = self._images[0][2].shape

        dimensions = [maxChannel, maxSlice, maxX, maxY]
        images = { t: np.zeros(dimensions, dtype=np.uint16) for t, _, i in  self._images}

        for time, channel, image in self._images:
            images[time][channel] = image

        return _MultiImageLoader(images)


class _MultiImageLoader(ImageLoader):
    """
    A loader class for loading from imageio supported formats.
    """

    def __init__(self, images: Dict[np.ndarray]):
        """
        Initialize the BaseImage class.

        Args:
          images (np.ndarray): [time, channel, slice].

        """
        super().__init__()
        self._images = images

    def shape(self) -> Tuple[int, int, int, int, int]:
        return self._images.shape

    def saveTo(self, group: zarr.Group):
        for time, images in self._images:
            group.create_dataset(f"t-{time}", data=images, dtype=images.dtype)

    def loadSlice(self, time: int, channel: int, slice: int) -> np.ndarray:
        return self._images[time][channel][slice]

    def fetchSlices(self, time: int, channel: int, sliceRange: Tuple[int, int]) -> np.ndarray:
        return np.max(self._images[time][channel][sliceRange[0]:sliceRange[1]], axis=0)
