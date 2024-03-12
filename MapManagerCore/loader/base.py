from functools import lru_cache
from typing import Tuple, Union
import numpy as np
import pandas as pd
from ..utils import polygonIndexes, toGeoData
import geopandas as gp


class ImageLoader:
    """
    Base class for image loaders.
    """

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
        ("implemented by subclass", time, channel, slice)

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
        sls = [self.loadSlice(time, channel, i)
               for i in range(int(sliceRange[0]), int(sliceRange[1]))]

        if len(sls) == 1:
            return sls[0]

        return np.max(sls, axis=0)

    def cached(self, maxsize=15):
        """
        Adds a cache to a subset of methods method.
        """
        cache = lru_cache(maxsize=maxsize)
        self.fetchSlices = cache(self.fetchSlices)

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

    def getPolygons(self, polygons: gp.GeoDataFrame, zExpand: int = 0, channel: int = 0):
        """
        Retrieve image slices corresponding to the given polygons.

        Args:
            polygons (gp.GeoDataFrame): GeoDataFrame containing the polygons under the column polygon, along with `z` and time `t`.
            zExpand (int, optional): Number of slices to expand in the z-direction. Defaults to 0.
            channel (int, optional): Channel index. Defaults to 0.

        Returns:
            pd.Series: Series containing the image slices corresponding to the polygons.
        """
        results = []
        indexes = []

        polygons["z"] = polygons["z"].astype(int)

        for (t, z), group in polygons.groupby(by=["t", "z"]):
            if zExpand == 0:
                image = self.loadSlice(t, channel, z)
            else:
                image = self.fetchSlices(
                    t, channel, (z - zExpand, z + zExpand + 1))

            for idx, row in group.iterrows():
                xs, ys = polygonIndexes(row["polygon"])
                xLim, yLim = image.shape

                results.append(
                    image[np.clip(xs, 0, xLim-1), np.clip(ys, 0, yLim-1)])
                indexes.append(idx)

        return pd.Series(results, indexes)


class Loader:
    def __init__(self, lineSegments: Union[str, pd.DataFrame], points: Union[str, pd.DataFrame]):
        if isinstance(lineSegments, str):
            lineSegments = pd.read_csv(
                lineSegments, index_col="segmentID", dtype={'segmentID': str})
        else:
            lineSegments = lineSegments.set_index('segmentID', drop=True)
            lineSegments["segmentID"] = lineSegments["segmentID"].astype(str)

        if isinstance(points, str):
            points = pd.read_csv(points, index_col="spineID", dtype={
                'spineID': str, 'segmentID': str})
        else:
            points = points.set_index('spineID', drop=True)
            points["spineID"] = points["spineID"].astype(str)
            points["segmentID"] = points["segmentID"].astype(str)

        lineSegments = toGeoData(lineSegments, ['segment'])
        points = toGeoData(points, ['point', 'anchor'])

        lineSegments["modified"] = lineSegments["modified"].astype(
            'datetime64[ns]')
        points["modified"] = points["modified"].astype('datetime64[ns]')

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
