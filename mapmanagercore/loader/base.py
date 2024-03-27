from functools import lru_cache
from typing import Self, Tuple, Union
import numpy as np
import pandas as pd

from mapmanagercore.config import LineSegment, Spine
from ..utils import shapeIndexes
import geopandas as gp
import zarr
from shapely.geometry.base import BaseGeometry
from shapely import wkt


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

    def dtype(self) -> np.dtype:
        """
        Returns the data type of the image data.

        Returns:
          np.dtype: The data type of the image data.
        """
        np.dtype("uint16")

    def shape(self) -> Tuple[int, int, int, int, int]:
        """
        Returns the shape of the image data.

        Returns:
          Tuple[int, int, int, int, int]: The shape of the image data, (t,c,z,x,y).
        """
        ("implemented by subclass")

    def channels(self) -> int:
        """
        Returns the number of channels in the image data.

        Returns:
          int: The number of channels in the image data.
        """
        return self.shape()[1]

    def saveTo(self, store: zarr.Group):
        """
        Saves the image data to a store.

        Args:
          store: The store to save the data to.
        """
        ("implemented by subclass")

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

    def cached(self, maxsize=15) -> Self:
        """
        Adds a cache to a subset of methods method.
        """
        cache = lru_cache(maxsize=maxsize)
        self.fetchSlices = cache(self.fetchSlices)
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

    def getShapePixels(self, shape: gp.GeoDataFrame, zSpread: int = 0, channel: int = 0):
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

        shape["z"] = shape["z"].astype(int)

        for (t, z), group in shape.groupby(by=["t", "z"]):
            if zSpread == 0:
                image = self.loadSlice(t, channel, z)
            else:
                image = self.fetchSlices(
                    t, channel, (z - zSpread, z + zSpread + 1))

            for idx, row in group.iterrows():
                xs, ys = shapeIndexes(row["shape"])
                xLim, yLim = image.shape

                results.append(
                    image[np.clip(xs, 0, xLim-1), np.clip(ys, 0, yLim-1)])
                indexes.append(idx)

        return pd.Series(results, indexes)


def loadShape(shape: Union[str, BaseGeometry]):
    if isinstance(shape, BaseGeometry):
        return shape
    return wkt.loads(shape)


def setColumnTypes(df: pd.DataFrame, types: Union[LineSegment, Spine]) -> gp.GeoDataFrame:
    defaults = types.defaults()
    types = types.__annotations__
    df = gp.GeoDataFrame(df)
    for key, valueType in types.items():
        if issubclass(valueType, np.datetime64):
            valueType = "datetime64[ns]"

        if key in df.index.names:
            df.index = df.index.astype(valueType)
            continue
        if not isinstance(valueType, str) and issubclass(valueType, BaseGeometry):
            df[key] = gp.GeoSeries(df[key].apply(
                loadShape)) if key in df.columns else gp.GeoSeries()
        else:
            if int == valueType:
                valueType = 'Int64'

            df[key] = df[key].astype(
                valueType) if key in df.columns else pd.Series(dtype=valueType)

        if key in defaults:
            df[key] = df[key].fillna(defaults[key])

    return df


class Loader:
    def __init__(self, lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(), points: Union[str, pd.DataFrame] = pd.DataFrame()):
        if not isinstance(lineSegments, gp.GeoDataFrame):
            if not isinstance(lineSegments, pd.DataFrame):
                lineSegments = pd.read_csv(lineSegments, index_col=False)

        lineSegments = setColumnTypes(lineSegments, LineSegment)
        if lineSegments.index.name != "segmentID":
            lineSegments.set_index("segmentID", drop=True, inplace=True)

        if not isinstance(points, gp.GeoDataFrame):
            if not isinstance(points, pd.DataFrame):
                points = pd.read_csv(points, index_col=False)

        points = setColumnTypes(points, Spine)
        if points.index.name != "spineID":
            points.set_index("spineID", drop=True, inplace=True)

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
