# This file predominantly contains classes that are used to represent
# annotations at a single time point via proxy-ing request to the multi 
# time-point equivalent of the classes. Note that new functionality should not 
# be added directly to this file, but rather to the multi time-point or a
# subclass of the classes in this file.

from collections.abc import Sequence
import pandas as pd
import geopandas as gp
from ...config import Metadata, SegmentId, SpineId
from ...schemas import Segment, Spine
from ...image_slices import ImageSlice
from ...lazy_geo_pandas.attributes import ColumnAttributes
from ...lazy_geo_pandas.lazy import LazyGeoFrame
from ...lazy_geo_pandas.schema import Schema
from .. import Annotations
from typing import Any, Callable, Hashable, List, Self, Tuple, Union
from copy import copy

# abb analysisparams
from mapmanagercore.analysis_params import AnalysisParams

class SingleTimePointFrame(LazyGeoFrame):
    def __init__(self, frame: LazyGeoFrame, t: int):
        if isinstance(frame, SingleTimePointFrame):
            self._root = copy(frame._root)
        else:
            self._root = copy(frame)
        self._t = t

    def __getitem__(self, items: Any) -> Any:
        filter = self._root._filterIdx
        try:
            self._root._filterIdx = self._root._df.xs(
                self._t, level=1, drop_level=False).index
            self._root._currentVersion = -1

            result = self._root[items]
            if isinstance(result, pd.DataFrame) or isinstance(result, gp.GeoDataFrame) or isinstance(result, pd.Series) or isinstance(result, gp.GeoSeries):
                if result.index is not None and result.index.nlevels > 1:
                    result = result.xs(self._t, level=1, drop_level=True)

            if isinstance(result, LazyGeoFrame):
                return SingleTimePointFrame(result, self._t)

            # extract single values if index is precisely one row
            if result.shape[0] <= 1:
                row, _key = self._parseKeyRow(items)
                if self._root._schema.isIndexType(row):
                    if result.empty:
                        return None
                    return result.values[0]

            return result
        finally:
            self._root._filterIdx = filter

    @property
    def index(self):
        return self._root.index.droplevel(1)

    def loadData(self, data: gp.GeoDataFrame) -> None:
        return self._root.loadData(data)

    def addComputed(self, column: str, attribute: ColumnAttributes, func: Callable[[], Union[gp.GeoSeries, gp.GeoDataFrame]], dependencies: Union[List[str], dict[str, list[str]]] = {}, skipUpdate=False) -> None:
        return self._root.addComputed(column, attribute, func,
                                      dependencies, skipUpdate)

    def updateComputed(self):
        return self._root.updateComputed()

    def getFrame(self, key: str):
        return self._root.getFrame(key)

    def pendingColumns(self) -> list[str]:
        if len(self._computingColumns) == 0:
            return []
        return self._computingColumns[-1]

    @property
    def columns(self):
        return self._root._columns

    @property
    def columnsAttributes(self):
        return self._root.columnsAttributes

    def undo(self) -> None:
        return self._root.undo()

    def redo(self) -> None:
        return self._root.redo()

    def drop(self, id: Union[Hashable, Sequence[Hashable], pd.Index], skipLog=False) -> None:
        return self._root.drop(id, skipLog)

    def update(self, ids: Union[Hashable, Sequence[Hashable], pd.Index], value: Schema, replaceLog=False, skipLog=False) -> None:
        if isinstance(ids, list):
            ids = [(id, self._t) for id in ids]
        else:
            ids = (ids, self._t)
        return self._root.update(ids, value, replaceLog, skipLog)

    def invalidClone(self, depKey: str) -> Union[None, Self]:
        return self._root.invalidClone(depKey)


# note hack to inherit types from Annotations
# this is a container for annotations at a single time point and does not
# actually inherit from Annotations directly
class _SingleTimePointAnnotationsBase(Annotations):
    _annotations: Annotations
    _t: int

    def __init__(self, annotations: Annotations, t: int) -> None:
        self._annotations = copy(annotations)

        self._annotations._segments = SingleTimePointFrame(
            self._annotations._segments, t)
        self._annotations._points = SingleTimePointFrame(
            self._annotations._points, t)

        self._t = t

    @property
    def points(self) -> LazyGeoFrame:
        return self._annotations.points

    @property
    def segments(self) -> LazyGeoFrame:
        return self._annotations.segments

    # abb analysisparams
    @property
    def analysisParams(self) -> AnalysisParams:
        return self._annotations._analysisParams

    @property
    def timeSeries(self) -> Annotations:
        return self._annotations


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

    def getShapePixels(self, shapes: gp.GeoDataFrame, channel: Union[int, List[int]] = 0, zSpread: int = 0, z: int = None) -> pd.Series:
        return self._annotations.getShapePixels(shapes, channel, zSpread, self._t, z=z)

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
