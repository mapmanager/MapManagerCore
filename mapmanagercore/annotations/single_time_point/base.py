import numpy as np
import pandas as pd
import geopandas as gp
from mapmanagercore.benchmark import timer
from mapmanagercore.config import Metadata, Segment, SegmentId, Spine, SpineId
from mapmanagercore.image_slices import ImageSlice
from .. import Annotations
from typing import Any, Tuple, Union
from copy import copy


# note hack to inherit types from Annotations
# this is a container for annotations at a single time point and does not
# actually inherit from Annotations directly
class _SingleTimePointAnnotationsBase(Annotations):
    _annotations: Annotations
    _t: int

    def __init__(self, annotations: Annotations, t: int) -> None:
        self._annotations = annotations
        self._t = t

    @property
    def timeSeries(self) -> Annotations:
        return self._annotations

    @timer
    def __getattr__(self, name: str) -> Any:
        if name == "_points":
            try:
                return self._annotations._points.xs(self._t, level=1)
            except KeyError:
                empty = self._annotations._lineSegments.head(0)
                return empty.set_index(empty.index.get_level_values(0), drop=True)
        if name == "_lineSegments":
            try:
                return self._annotations._lineSegments.xs(self._t, level=1)
            except KeyError:
                empty = self._annotations._lineSegments.head(0)
                return empty.set_index(empty.index.get_level_values(0), drop=True)
        result = getattr(self._annotations, name)
        return result

    def __getitem__(self, key: Any) -> Any:
        filtered = copy(self._annotations)
        filtered._points = filtered._points.xs(
            self._t, level=1, drop_level=False)
        filtered._lineSegments = filtered._lineSegments.xs(
            self._t, level=1, drop_level=False)

        result = filtered[key]

        if isinstance(result, pd.DataFrame) or isinstance(result, gp.GeoDataFrame) or isinstance(result, pd.Series) or isinstance(result, gp.GeoSeries):
            if result.index is not None and isinstance(result.index, pd.MultiIndex):
                if result.shape[0] == 1:
                    if isinstance(key, tuple) and len(key) > 1 and isinstance(key[0], int) and (len(key) == 1 or isinstance(key[1], str)):
                        return result.values[0]
                return result.xs(self._t, level=1)

        if isinstance(result, Annotations):
            return self.__class__(result, self._t)
        return result


Key = SpineId
Keys = Union[Key, list[Key]]


class SingleTimePointAnnotationsBase(_SingleTimePointAnnotationsBase):

    def getPixels(self, channel: int, zRange: Tuple[int, int] = None, z: int = None, zSpread: int = 0) -> ImageSlice:
        """
        Loads the image data for a slice.

        Args:
          channel (int): The channel index.
          zRange (Tuple[int, int]): The visible z slice range.
          z (int): The z slice index.
          zSpread (int): The amount to offset z +/-.

        Returns:
          ImageSlice: The image slice.
        """
        return self._annotations.getPixels(self._t, channel, zRange, z, zSpread)

    def getShapePixels(self, shapes: gp.GeoSeries, channel: int = 0, zSpread: int = 0, ids: pd.Index = None, id: str = None) -> pd.Series:
        return self._annotations.getShapePixels(shapes, channel, zSpread, ids, id, self._t)

    def _mapKeys(self, keys: Keys) -> Keys:
        if isinstance(keys, list):
            return [(id, self._t) for id in keys]
        else:
            return (keys, self._t)

    def deleteSpine(self, spineId: Keys, skipLog=False) -> None:
        return self._annotations.deleteSpine(self._mapKeys(spineId), skipLog)

    def deleteSegment(self, segmentId: Keys, skipLog=False) -> None:
        return self._annotations.deleteSegment(self._mapKeys(segmentId), skipLog)

    def updateSegment(self, segmentId: Keys, value: Segment, replaceLog=False, skipLog=False):
        return self._annotations.updateSegment(self._mapKeys(segmentId), value, replaceLog, skipLog)

    def updateSpine(self, spineId: Keys, value: Spine, replaceLog=False, skipLog=False):
        return self._annotations.updateSpine(self._mapKeys(spineId), value, replaceLog, skipLog)

    def connect(self, spineKey: SpineId, toSpineKey: Tuple[SpineId, int]):
        return self._annotations.connect((spineKey, self._t), toSpineKey)

    def disconnect(self, spineKey: SpineId):
        return self._annotations.disconnect((spineKey, self._t))

    def connectSegment(self, segmentKey: SegmentId, toSegmentKey: Tuple[SegmentId, int]):
        return self._annotations.connectSegment((segmentKey, self._t), toSegmentKey)

    def disconnectSegment(self, segmentKey: SegmentId):
        return self._annotations.disconnectSegment((segmentKey, self._t))

    def metadata(self) -> Metadata:
        return self._images.metadata(self._t)
    
