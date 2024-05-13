from functools import lru_cache
from typing import Iterator, List, Self, Tuple, Union
import numpy as np
import pandas as pd
from ..config import Metadata
from ..utils import shapeIndexes
import geopandas as gp
import zarr
from shapely.geometry.base import BaseGeometry
from shapely import wkt

from mapmanagercore.analysis_params import AnalysisParams

from mapmanagercore.logger import logger

class ImageLoader:
    """
    Base class for image loaders.
    """

    def timePoints(self) -> Iterator[int]:
        ("implemented by subclass")

    def _images(self, t: int) -> np.ndarray:
        ("implemented by subclass", t)

    def metadata(self, t: int) -> Metadata:
        ("implemented by subclass")

    def loadSlice(self, time: int, channel: int, slice: int) -> np.ndarray:
        """
        Loads a slice of data for the given time, channel, and slice index.

        Args:
          time (int): The time index.
          channel (int): The channel index.
          slice (int): The slice index.

        Returns:
          np.ndarray: The loaded slice of data.
        """
        return self._images(time)[channel][slice]

    def dtype(self, t: int) -> np.dtype:
        """
        Returns the data type of the image data.

        Returns:
          np.dtype: The data type of the image data.
        """
        np.dtype(str.lower(self.metadata(t)["dtype"]))

    def shape(self, t: int) -> Tuple[int, int, int, int]:
        """
        Returns the shape of the image data.

        Returns:
          Tuple[int, int, int, int]: The shape of the image data, (c,z,x,y).
        """
        return self._images(t).shape

    def channels(self, t: int = None) -> int:
        """
        Returns the number of channels in the image data.

        Returns:
          int: The number of channels in the image data.
        """

        if t is None:
            return min(self.shape(t)[0] for t in self.timePoints())

        return self.shape(t)[0]

    def slices(self, t: int) -> int:
        """
        Returns the number of channels in the image data.

        Returns:
          int: The number of channels in the image data.
        """
        return self.shape(t)[1]

    def saveTo(self, group: zarr.Group):
        """
        Saves the image data to a store.

        Args:
          store: The store to save the data to.
        """
        for t in self.timePoints():
            image = self._images(t)
            group.create_dataset(f"img-{t}", data=image, dtype=image.dtype)
            group.attrs[f"metadata-{t}"] = self.metadata(t)

        group.attrs["timePoints"] = list(self.timePoints())

    def fetchSlices(self, time: int, channel: int, sliceRange: Tuple[int, int]) -> np.ndarray:
        """
        Fetches a range of slices for the given time, channel, and slice range.

        Args:
          time (int): The time index.
          channel (int): The channel index.
          sliceRange (tuple): The range of slice indices.

        Returns:
          np.ndarray: The fetched slices.
        """
        if sliceRange[0] == sliceRange[1] - 1:
            return self.loadSlice(time, channel, sliceRange[0])

        return np.max(self._images(time)[channel][sliceRange[0]:sliceRange[1]], axis=0)

    def cached(self, maxsize=15) -> Self:
        """
        Adds a cache to a subset of methods method.
        """
        cache = lru_cache(maxsize=maxsize)
        self.fetchSlices = cache(self.fetchSlices)
        self.fetchSlices2 = cache(self.fetchSlices2)
        return self

    def get(self, time: int, channel: int, z: Union[Tuple[int, int], int, np.ndarray], x: Union[Tuple[int, int, np.ndarray], int], y: Union[Tuple[int, int], int, np.ndarray]) -> np.array:
        """
        Fetches a range of slices for the given time, channel, and slice range.

        Args:
          time (int): The time index.
          channel (int): The channel index.
          z (tuple): The range of slice indices.
          x (tuple): The range of x indices.
          y (tuple): The range of y indices.

        Returns:
          np.ndarray: The fetched slices.
        """
        z = z if isinstance(z, tuple) else (z, z + 1)

        if isinstance(x, np.ndarray):
            x = bounds(x)
        else:
            x = x if isinstance(x, tuple) else (x, x + 1)

        if isinstance(y, np.ndarray):
            y = bounds(y)
        else:
            y = y if isinstance(y, tuple) else (y, y + 1)

        if z[0] == z[1] - 1:
            slices = self.loadSlice(time, channel, int(z[0]))
        else:
            slices = self.fetchSlices(time, channel, z)

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return slices[x, y]
        if isinstance(x, np.ndarray):
            return slices[x, y[0]:y[1]]
        if isinstance(y, np.ndarray):
            return slices[x[0]:x[1], y]

        return slices[x[0]:x[1], y[0]:y[1]]

    def getShapePixels(self, shape: gp.GeoDataFrame, zSpread: int = 0, channel: Union[int, List[int]] = 0, time=None, z: int = None):
        """
        Retrieve image slices corresponding to the given shape.

        Args:
            shape (gp.GeoDataFrame): GeoDataFrame containing the shape under the column polygon, along with `z` and time `t`.
            zSpread (int, optional): Number of slices to expand in the z-direction. Defaults to 0.
            channel (int, optional): Channel index. Defaults to 0.

        Returns:
            pd.Series: Series containing the image slices corresponding to the shape.
        """
        results = []
        indexes = []
        if isinstance(shape, list):
            shape = gp.GeoDataFrame(shape, columns=["shape"], geometry="shape")

        if isinstance(shape, pd.Series) or isinstance(shape, gp.GeoSeries):
            shape = shape.to_frame("shape")

        if "t" in shape.index.names:
            if not "t" in shape.columns:
                shape.reset_index("t", inplace=True)
            else:
                shape.drop("t", axis=1, inplace=True)

        if time is not None:
            shape["t"] = time

        if not "z" in shape:
            if z is None:
                coords = shape["shape"].get_coordinates(include_z=True)
                shape["z"] = coords["z"].groupby(coords.index).mean()
            else:
                shape["z"] = z

        shape["z"] = shape["z"].astype(int)

        if isinstance(channel, list):
            for (t, z), group in shape.groupby(by=["t", "z"]):
                images = [self.fetchSlices(
                    t, c, (z - zSpread, z + zSpread + 1)) for c in channel]

                for idx, row in group.iterrows():
                    xs, ys = shapeIndexes(row["shape"])
                    xLim, yLim = images[0].shape
                    xBase = np.clip(xs, 0, xLim-1)
                    yBase = np.clip(ys, 0, yLim-1)

                    results.append(
                        [image[xBase, yBase] for image in images])
                    indexes.append(idx)
            return pd.DataFrame(results, indexes, columns=channel)

        for (t, z), group in shape.groupby(by=["t", "z"]):
            image = self.fetchSlices(
                t, channel, (z - zSpread, z + zSpread + 1))

            # logger.info(f'   t:{t} z:{z} image:{image.shape}')
            # print('group')
            # print(group)

            for idx, row in group.iterrows():
                xs, ys = shapeIndexes(row["shape"])
                xLim, yLim = image.shape

                results.append(
                    image[np.clip(xs, 0, xLim-1), np.clip(ys, 0, yLim-1)])
                indexes.append(idx)

        return pd.Series(results, indexes, name=channel)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class Loader:
    def __init__(self,
                 lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(),
                 points: Union[str, pd.DataFrame] = pd.DataFrame(),
                 metadata: Union[str, Metadata] = Metadata(),
                 analysisParams: AnalysisParams = AnalysisParams()):
        
        if not isinstance(lineSegments, gp.GeoDataFrame):
            if not isinstance(lineSegments, pd.DataFrame):
                lineSegments = pd.read_csv(lineSegments, index_col=False)

        if not isinstance(points, gp.GeoDataFrame):
            if not isinstance(points, pd.DataFrame):
                points = pd.read_csv(points, index_col=False)

        self._lineSegments = lineSegments
        self._points = points

    def points(self) -> gp.GeoDataFrame:
        return self._points

    def segments(self) -> gp.GeoDataFrame:
        return self._lineSegments

    def images(self) -> ImageLoader:
        "abstract method"


def bounds(x: np.array):
    return (x.min(), int(x.max()) + 1)
