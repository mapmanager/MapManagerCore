from enum import IntEnum
from functools import lru_cache
from typing import Iterator, List, Self, Tuple, TypedDict, Union
from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from mapmanagercore.logger import logger

from dataclasses import asdict
# import json
import numpy as np
import pandas as pd
import geopandas as gp
import zarr
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
import shapely
import skimage.draw


class Position(IntEnum):
    OVER = 0
    AT = 1


class DataTreeNodeChannel(TypedDict):
    channel: int
    slices: int
    width: int
    height: int
    name: str


class DataTreeNode(TypedDict):
    name: str
    channels: List[DataTreeNodeChannel]


def shapeIndexes(d: Union[Polygon, LineString]) -> Tuple[np.ndarray, np.ndarray]:
    """ Get the x and y indexes of the pixels in a shape."""

    d = shapely.force_2d(d)

    # TODO: fully support multi-polygon
    if isinstance(d, MultiPolygon):
        d = d.geoms[0]
    if isinstance(d, GeometryCollection):
        d = next(s for s in d.geoms if isinstance(s, Polygon))

    if isinstance(d, Polygon):
        coords = d.exterior.coords.xy
    else:
        coords = d.coords.xy

    x, y = np.array(coords[0]), np.array(coords[1])
    # Since skimage.draw does not support negative coordinates, we need to shift the coordinates.
    minx, miny = int(min(min(x), 0)), int(min(0, min(y)))
    x, y = x - minx, y - miny

    if isinstance(d, Polygon):
        xs, ys = skimage.draw.polygon(x, y)
    else:
        xs, ys = skimage.draw.line(int(x[0]), int(y[0]), int(x[1]), int(y[1]))

    # Shift the coordinates back to the original position.
    xs, ys = xs + minx, ys + miny
    return xs, ys


class ImageLoader:
    """
    Base class for image loaders.
    """
    _metadata: List[Metadata]
    _analysisParams: AnalysisParams

    def __init__(self):
        self._metadata = []
        self._analysisParams = AnalysisParams()

    def __str__(self):
        return f"ImageLoader: time points: {self.timePoints()}"

    def merge(self, loader: Self):
        ("implemented by subclass", loader)
        pass

    def analysisParams(self):
        return self._analysisParams

    def metadata(self, t: int) -> Metadata:
        return self._metadata[t] if t < len(self._metadata) else Metadata()

    def timePoints(self) -> Iterator[int]:
        ("implemented by subclass")
        return []

    def _images(self, t: int, channel: int) -> np.ndarray:
        ("implemented by subclass", t, channel)
        return np.array([])

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
        return self._images(time, channel)[slice]

    def dtype(self, t: int) -> np.dtype:
        """
        Returns the data type of the image data.

        Returns:
          np.dtype: The data type of the image data.
        """
        return np.dtype(str.lower(self.metadata(t)["dtype"]))

    def shape(self, t: int, channel: int = None) -> Tuple[int, int, int]:
        """
        Returns the shape of the image data.

        Returns:
          Tuple[int, int, int]: The shape of the image data, (z,x,y).
        """
        if t not in self.timePoints():
            return (0, 0, 0)
        if channel is None:
            channels = self.channels(t)

            if len(channels) == 0:
                return (0, 0, 0)

            channel = channels[0]
        return self._images(t, channel).shape

    def channels(self, t: int) -> List[int]:
        metaData = self.metadata(t)
        return list(metaData.channelNames.keys())

    # abb TODO depreciate
    def maxChannels(self) -> int:
        return self._analysisParams.getValue("maxChannels")

    # abb TODO depreciate
    def setMaxChannels(self, maxChannels: int) -> bool:
        if self.maxChannels() == maxChannels:
            return False
        self._analysisParams.setValue("maxChannels", maxChannels)  # abb md3 depreciated
        return True

    def slices(self, t: int, channel: int = 0) -> int:
        """
        Returns the number of slices in the image data.

        Returns:
          int: The number of slices in the image data.
        """
        return self.shape(t, channel)[0]

    def saveTo(self, group: zarr.Group):
        """
        Saves the image data to a store.

        Args:
          store: The store to save the data to.
        """
        
        deleteGroups = set(group.keys())
        for t in self.timePoints():
            channels = self.channels(t)
            # abb
            tStr = str(t)
            if tStr in group.keys():
                # logger.error(f'abb {tStr} already exists in group {group}')
                # logger.error('TODO check if we have new channels and save them')
                timePoint = group[tStr]
            else:
                timePoint = group.create_group(str(t))
            
            metaData = self.metadata(t)
            timePoint.attrs[f"metadata"] = asdict(metaData)

            deleteChannels = set(timePoint.keys())
            for channel in channels:
                strChannel = str(channel)
                image = self._images(t, channel)
                if strChannel in timePoint.keys():
                    # logger.error(f'channel {strChannel} is already in timepoint {timePoint}')
                    pass
                else:
                    timePoint.create_dataset(
                        str(channel), data=image, dtype=image.dtype)

    def getAutoContrast_qt(self, time: int, channel: int) -> Tuple[int, int, int, int]:
        """Get the auto contrast from the entire image volume.

        Used in PyQt interface.
        """

        # logger.info(f'{self._images(time)[channel].shape} {np.min(self._images(time)[channel]), np.max(self._images(time)[channel])}')
        
        imgData = self._images(time, channel)

        # _percent_low = 25.0  #20.0 #0.5  # .30
        # _percent_high = 99.8  # 99.95  #100 - 0.5
        # percentiles = np.percentile(imgData, (_percent_low, _percent_high))
        # theMin = int(percentiles[0])
        # theMax = int(percentiles[1])

        # not working for a stack???
        from mapmanagercore.utils import getAutoContrast
        _maxProject = np.max(imgData, axis=0)
        theMin, theMax = getAutoContrast(_maxProject)

        globalMin = np.min(imgData)
        globalMax = np.max(imgData)

        logger.warning(f'EXPENSIVE --> channel:{channel} shape:{imgData.shape} dtype:{imgData.dtype} globalMin:{globalMin} globalMax:{globalMax}')

        return theMin, theMax, globalMin, globalMax
    
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

        # abb fetchSlices() is getting called multiple times when editing one spine?
        # logger.info(f'=== time:{time} channel:{channel} sliceRange:{sliceRange}')

        # logger.warning(f'xxx {self._images(time)[channel].shape}')

        z, _x, _y = self.shape(time, channel)
        sliceRange = (max(0, sliceRange[0]), min(z, sliceRange[1]))

        if sliceRange[0] == sliceRange[1] - 1:
            return self.loadSlice(time, channel, sliceRange[0])

        return np.max(self._images(time, channel)[sliceRange[0]:sliceRange[1]], axis=0)

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

    def getShapePixels(self, shape: gp.GeoDataFrame, zSpread: int = 0, channel: Union[int, List[int]] = 0, time=None, z: int = None):
        """
        Retrieve image slices corresponding to the given shape.

        Args:
            shape (gp.GeoDataFrame): GeoDataFrame containing the shape under the column polygon, along with `z` and time `t`.
            zSpread (int, optional): Number of slices to expand in the z-direction. Defaults to 0.
            channel (int, optional): Channel index. Defaults to 0.
            time (int, optional): Time index. Defaults to None. If provided, the time index will be used instead of the `t` column in the shape.
            z (int, optional): Z index. Defaults to None. If provided, the z index will be used instead of the `z` column in the shape.

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
                    xLim, yLim = images[0].shape
                    xs, ys = shapeIndexes(row["shape"])
                    # Clip the coordinates to the image bounds.
                    inBounds = (xs >= 0) & (xs < xLim) & (
                        ys >= 0) & (ys < yLim)
                    xs = np.clip(xs, 0, xLim - 1)
                    ys = np.clip(ys, 0, yLim - 1)

                    # inject the nan values where the shape is out of bounds.
                    results.append(
                        [np.where(inBounds, image[xs, ys], np.nan) for image in images])
                    indexes.append(idx)
            return pd.DataFrame(results, indexes, columns=channel)

        for (t, z), group in shape.groupby(by=["t", "z"]):
            image = self.fetchSlices(
                t, channel, (z - zSpread, z + zSpread + 1))

            for idx, row in group.iterrows():
                xLim, yLim = image.shape
                xs, ys = shapeIndexes(row["shape"])
                # Clip the coordinates to the image bounds.
                inBounds = (xs >= 0) & (xs < xLim) & (ys >= 0) & (ys < yLim)
                xs = np.clip(xs, 0, xLim - 1)
                ys = np.clip(ys, 0, yLim - 1)

                # inject the nan values where the shape is out of bounds.
                results.append(np.where(inBounds, image[xs, ys], np.nan))
                indexes.append(idx)

        return pd.Series(results, indexes, name=channel)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def dataTree(self) -> List[DataTreeNode]:
        tree = []
        timePoints = sorted(self.timePoints())
        for t in timePoints:
            channels = []
            metadata = self.metadata(t)
            channels_ = sorted(self.channels(t))
            for channel in channels_:
                shape = self.shape(t, channel)
                name = metadata.channelNames[channel] if channel in metadata.channelNames else f"Unnamed Channel"

                channels.append(DataTreeNodeChannel({
                    "channel": channel,
                    "name": name,
                    "slices": shape[0],
                    "width": shape[1],
                    "height": shape[2],
                }))

            tree.append(DataTreeNode({
                "name": metadata.name if metadata.name != "" else f"Unnamed Time Point",
                "channels": channels
            }))

        return tree

    def createTimePoint(self) -> bool:
        return False

    def appendChannelToTimePoint(self, srcTimePoint: int, srcChannel: int, destTimePoint: int) -> bool:
        ("implemented by subclass", srcTimePoint, srcChannel, destTimePoint)
        return False

    def moveChannel(self, srcTimePoint: int, srcChannel: int, destTimePoint: int, destChannel: int) -> bool:
        ("implemented by subclass", srcTimePoint,
         srcChannel, destTimePoint, destChannel)
        return False

    def moveTimePoint(self, srcTimePoint: int, destTimePoint: int, position: Position = Position.OVER) -> bool:
        ("implemented by subclass", srcTimePoint, destTimePoint, position)
        return False

    def deleteTimePoint(self, timePoint: int) -> bool:
        ("implemented by subclass", timePoint)
        return False

    def deleteChannel(self, timePoint: int, channel: int) -> bool:
        ("implemented by subclass", channel)
        return False

    def updateChannel(self, timePoint: int, channel: int, updates: dict) -> bool:
        metadata = self.metadata(timePoint)
        if "name" in updates:
            metadata.channelNames[channel] = updates["name"]

        newTimePoint = int(updates["timePoint"]) - 1
        newChannel = int(updates["channel"]) - 1
        if timePoint != newTimePoint or channel != newChannel:
            self.moveChannel(timePoint, channel, newTimePoint, newChannel)

        return True

    def updateTimePoint(self, timePoint: int, updates: dict) -> bool:
        metadata = self.metadata(timePoint)
        metadata.name = updates.get("name", metadata.name)
        metadata.physicalSize.x = float(updates.get(
            "physicalSizeX", metadata.physicalSize.x))
        metadata.physicalSize.y = float(updates.get(
            "physicalSizeY", metadata.physicalSize.y))
        metadata.physicalSize.unit = updates.get(
            "physicalSizeUnit", metadata.physicalSize.unit)
        metadata.voxel.x = float(updates.get("voxelX", metadata.voxel.x))
        metadata.voxel.y = float(updates.get("voxelY", metadata.voxel.y))
        metadata.voxel.z = float(updates.get("voxelZ", metadata.voxel.z))

        newTimePoint = int(updates["timePoint"]) - 1
        if timePoint != newTimePoint:
            self.moveTimePoint(timePoint, newTimePoint)

        return True


def bounds(x: np.array):
    return (x.min(), int(x.max()) + 1)
