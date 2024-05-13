from functools import lru_cache
import json
from typing import Self, Tuple, Union
import numpy as np
import pandas as pd

from ..config import Metadata, Segment, Spine
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
        shape = shape.reset_index()
        shape["z"] = shape["z"].astype(int)

        # logger.info('shape:')
        # print(shape)
        # logger.info(f'zSpread:{zSpread}')
        # logger.info(f'channel:{channel}')
        
        for (t, z), group in shape.groupby(by=["t", "z"]):
            if zSpread == 0:
                image = self.loadSlice(t, channel, z)
            else:
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

        return pd.Series(results, indexes)


def loadShape(shape: Union[str, BaseGeometry]):
    if shape is None:
        return None
    if isinstance(shape, BaseGeometry):
        return shape
    return wkt.loads(shape)


def setColumnTypes(df: pd.DataFrame, types: Union[Segment, Spine]) -> gp.GeoDataFrame:
    defaults = types.defaults()
    types = types.__annotations__
    df = gp.GeoDataFrame(df)
    for key, valueType in types.items():
        if issubclass(valueType, np.datetime64):
            valueType = "datetime64[ns]"

        if key in df.index.names:
            if int == valueType:
                valueType = 'Int64'

            if len(df.index.names) == 1:
                df.index = df.index.astype(valueType)
            else:
                i = df.index.names.index(key)
                df.index = df.index.set_levels(
                    df.index.levels[i].astype(valueType), level=i)
            continue
        if not isinstance(valueType, str) and issubclass(valueType, BaseGeometry):
            df[key] = gp.GeoSeries(df[key].apply(
                loadShape)) if key in df.columns else gp.GeoSeries()
        else:
            if int == valueType:
                valueType = 'Int64'
                if key in df.columns:
                    df[key] = np.trunc(df[key])

            # abb 03/2024
            # see: https://stackoverflow.com/questions/62899860/how-can-i-resolve-typeerror-cannot-safely-cast-non-equivalent-float64-to-int6
            try:
                df[key] = df[key].astype(
                    valueType) if key in df.columns else pd.Series(dtype=valueType)
            except (TypeError):
                if key in df.columns:
                    df[key] = np.floor(pd.to_numeric(df[key], errors='coerce')).astype('Int64')
                else:
                    df[key] = pd.Series(dtype=valueType)
                
                # logger.warning(e)
                # logger.warning(f'df[key].dtype:{df[key].dtype}')
                # logger.warning(f'key:{key} valueType:{valueType}')

        if key in defaults:
            df[key] = df[key].fillna(defaults[key])

    return df


class Loader:
    def __init__(self,
                 lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(),
                 points: Union[str, pd.DataFrame] = pd.DataFrame(),
                 metadata: Union[str, Metadata] = Metadata(),
                 analysisParams: AnalysisParams = AnalysisParams()):
        
        if not isinstance(lineSegments, gp.GeoDataFrame):
            if not isinstance(lineSegments, pd.DataFrame):
                lineSegments = pd.read_csv(lineSegments, index_col=False)

        lineSegments = setColumnTypes(lineSegments, Segment)
        if not "segmentID" in lineSegments.index.names or not "t" in lineSegments.index.names:
            lineSegments.set_index(["segmentID", "t"], drop=True, inplace=True)
        lineSegments.sort_index(level=0, inplace=True)

        # logger.info('core lineSegments')
        # print(lineSegments)

        if not isinstance(points, gp.GeoDataFrame):
            if not isinstance(points, pd.DataFrame):
                points = pd.read_csv(points, index_col=False)

        points = setColumnTypes(points, Spine)
        if not "spineID" in points.index.names or not "t" in points.index.names:
            points.set_index(["spineID", "t"], drop=True, inplace=True)
        points.sort_index(level=0, inplace=True)

        lineSegments["modified"] = lineSegments["modified"].astype(
            'datetime64[ns]')
        points["modified"] = points["modified"].astype('datetime64[ns]')

        self._lineSegments = lineSegments
        self._points = points

        if isinstance(metadata, str):
            with open(metadata, "r") as metadataFile:
                metadata = json.load(metadataFile)

        self._metadata = metadata

        # abb 202404
        self._analysisParams = analysisParams

    def points(self) -> gp.GeoDataFrame:
        return self._points

    def segments(self) -> gp.GeoDataFrame:
        return self._lineSegments

    def images(self) -> ImageLoader:
        "abstract method"

    def metadata(self) -> Metadata:
        return self._metadata

    def analysisParams(self) -> AnalysisParams:
        return self._analysisParams

def bounds(x: np.array):
    return (x.min(), int(x.max()) + 1)
