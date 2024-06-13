from typing import List, Tuple
import numpy as np


class ImageSlice:
    """
    Represents an image slice with plotting utilities.
    """

    def __init__(self, image: np.ndarray):
        self._image = image

    def __getitem__(self, key):
        return self._image[key]

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the image data.

        Returns:
          tuple: The shape of the image data.
        """
        return self._image.shape

    def data(self) -> np.ndarray:
        """
        Returns the image data.

        Returns:
          np.ndarray: The image data.
        """
        return self._image.flatten()

    def extent(self) -> Tuple[int, int]:
        """
        The range of the image data

        Returns:
            Tuple[int, int]: min and max range of the image data
        """
        return (self._image.min(), self._image.max())

    def bins(self, binCount: int = 256) -> List[Tuple[int, int]]:
        """
        Calculate the histogram bins for the image.

        Args:
          binCount (int): The number of bins to use for the histogram. Default is 256.

        Returns:
          list: A list of tuples representing the histogram bins. Each tuple contains the bin center and the count.
        """
        counts, bounds = np.histogram(self._image, binCount)
        return [((bounds[i] + bounds[i + 1]) / 2, int(counts[i])) for i in range(0, len(counts))]

    def plot(self, ax=None, cmap: str = 'viridis', **kwargs):
        """
        Plots the image data.

        Args:
          ax: The axis to plot the image on.
          cmap (str): The colormap to use for the image. Default is 'viridis'.
        """
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()

        ax.imshow(self._image, cmap=cmap, interpolation='nearest', **kwargs)
