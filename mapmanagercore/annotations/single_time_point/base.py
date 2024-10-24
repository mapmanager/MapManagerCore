# This file predominantly contains classes that are used to represent
# annotations at a single time point via proxy-ing request to the multi
# time-point equivalent of the classes. Note that new functionality should not
# be added directly to this file, but rather to the multi time-point or a
# subclass of the classes in this file.

from collections.abc import Sequence
import pandas as pd
import geopandas as gp

from mapmanagercore.benchmark import timer
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from ...config import SegmentId, SpineId
from ...schemas import Segment, Spine
from ...lazy_geo_pd_images.image_slices import ImageSlice
from ...lazy_geo_pandas.attributes import ColumnAttributes
from ...lazy_geo_pandas.lazy import LazyGeoFrame
from ...lazy_geo_pandas.schema import Schema
from .. import Annotations
from typing import Any, Callable, Hashable, List, Self, Tuple, Union
from copy import copy

# abb analysisparams
from mapmanagercore.analysis_params import AnalysisParams

from mapmanagercore.logger import logger

class SingleTimePointFrame(LazyGeoFrame):
    def __init__(self, frame: LazyGeoFrame, t: int):
        
        # frame does not have any computed values (columns)
        # 
        # logger.warning(f'SingleTimePointFrame constructor from frame: {type(frame)}')
        # # abb no __str__ rep for LazyGeoFrame
        # logger.warning(f'frame is: {frame}')
        # print('frame._rootDf is:')
        # print(frame._rootDf.columns)

        if isinstance(frame, SingleTimePointFrame):
            # logger.warning('CONSTRUCTING FROM SingleTimePointFrame')
            # logger.error(f'abb copy memory #3 frame:{type(frame)}')
            self._root = copy(frame._root)
            # logger.warning('abb SingleTimePointFrame removed copy v1')
            # self._root = frame._root
        else:
            # why is this getting called so much???
            # logger.warning('NOT CONSTRUCTING FROM SingleTimePointFrame')
            # logger.error(f'abb copy memory #4 frame:{type(frame)}')
            # print('   frame:', type(frame))
            # frame: mapmanagercore.lazy_geo_pandas.lazy.LazyGeoFrame
            self._root = copy(frame)
            # logger.warning('abb SingleTimePointFrame removed copy v2')
            # self._root = frame

        # abb
        # if isinstance(frame, SingleTimePointFrame):
        #     self._root = frame._root
        # else:
        #     self._root = frame

        self._currentVersion = -1
        self._t = t
        self._refreshIndex()

    @timer
    def _refreshIndex(self):
        if self._root._state.version == self._currentVersion:
            return

        self._currentVersion = self._root._state.version

        if self._t not in self._root._df.index.get_level_values(1):
            self._root._setFilterIndex(pd.Index([]))
            return

        self._root._setFilterIndex(self._root._df.xs(
            self._t, level=1, drop_level=False).index)

    @timer
    def __getitem__v0(self, items: Any) -> Any:
        self._refreshIndex()

        result = self._root[items]
        if isinstance(result, pd.DataFrame) or isinstance(result, gp.GeoDataFrame) or isinstance(result, pd.Series) or isinstance(result, gp.GeoSeries):
            if result.index is not None and result.index.nlevels > 1:
                #abj: only check Dataframes since Series/ gp.Series due to AttributeError: 'GeoSeries' object has no attribute 'set_index'
                if result.empty and (isinstance(result, pd.DataFrame) or isinstance(result, gp.GeoDataFrame)):        
                # if result.empty:
                    # this can only be called by a dataframe
                    return result.set_index(result.index.droplevel(1), inplace=False, drop=True)

                #abj
                if not result.empty:
                    result = result.xs(self._t, level=1, drop_level=True)

        if isinstance(result, LazyGeoFrame):
            # logger.info('  (2) return SingleTimePointFrame')
            return SingleTimePointFrame(result, self._t)

        # extract single values if index is precisely one row
        if result.shape[0] <= 1:
            if (len(result.shape) == 1 or result.shape[1] <= 1):
                row, _key = self._parseKeyRow(items)
                if self._root._schema.isIndexType(row):
                    if result.empty:
                        return None
                    return result.values[0]
            else:
                if result.empty:
                    return None
                return result.iloc[0]

        return result

    # abj johnson version
    @timer
    def __getitem__(self, items: Any) -> Any:
        """
        Notes
        -----
        When called with [:], items = slice(None, None, None)
        """

        # print('')
        # logger.info(f'=== abj version SingleTimePointFrame base.py items: {items}')
        
        self._refreshIndex()

        result = self._root[items]

        # logger.info(f'  at start result._root[items] is type {type(result)}')
        # print(result)
        
        if isinstance(result, pd.DataFrame) or isinstance(result, gp.GeoDataFrame) or isinstance(result, pd.Series) or isinstance(result, gp.GeoSeries):
            if result.index is not None and result.index.nlevels > 1:
                #abj: only check Dataframes since Series/ gp.Series cause AttributeError: 'GeoSeries' object has no attribute 'set_index'
                if result.empty and (isinstance(result, pd.DataFrame) or isinstance(result, gp.GeoDataFrame)):
                # if result.empty:
                    # this can only be called by a dataframe
                    # logger.info(f'  (1) return result.set_index')
                    return result.set_index(result.index.droplevel(1), inplace=False, drop=True)

                #abj: fix for key error: 0
                if not result.empty:
                    # logger.info('  1.5) result = result.xs(self._t, level=1, drop_level=True)')
                    result = result.xs(self._t, level=1, drop_level=True)

        if isinstance(result, LazyGeoFrame):
            logger.info('  (2) return SingleTimePointFrame')
            print('self._t:', self._t, type(self._t))
            print('result:', result)

            return SingleTimePointFrame(result, self._t)

        # extract single values if index is precisely one row
        if result.shape[0] <= 1 and (len(result.shape) == 1 or result.shape[1] <= 1):
            # logger.info(f"check 2 is instance type: {type(result)}")
            row, _key = self._parseKeyRow(items)
            if self._root._schema.isIndexType(row):
                if result.empty:
                    # logger.info('  (3) return NONE -->> ERROR')
                    # logger.info(f'  row:{row}')
                    # logger.info(f'  result: {type(result)}')
                    # print(result)
                    # logger.info(f'result.values: {type(result.values)}')
                    # print(result.values)
                    
                    from shapely.geometry import LineString
                    return LineString([])
                    
                    # abb was this
                    # return None
                
                # logger.info('  (4) return result.values[0]')
                # print(result.values[0])
                
                return result.values[0]

        #abj
        # error: empty spineLines returns series instead of geoseries
        # force pd.series into geoseries
        if isinstance(result, pd.Series) and len(result) == 0:
            # logger.error('  (5) result = gp.GeoSeries(result)')
            # logger.error(f'result:{result} {type(result)}')
            result = gp.GeoSeries(result)

        # abb never seems to get here???
        # logger.info(f'  (6) final return result: {type(result)}')
        # print(result)
        
        return result
    
    @property
    def index(self):
        self._refreshIndex()
        return self._root.index.droplevel(1)

    def loadData(self, data: gp.GeoDataFrame):
        return self._root.loadData(data)

    def addComputed(self, column: str, attribute: ColumnAttributes, func: Callable[[], Union[gp.GeoSeries, gp.GeoDataFrame]], dependencies: Union[List[str], dict[str, list[str]]] = {}, skipUpdate=False):
        return self._root.addComputed(column, attribute, func,
                                      dependencies, skipUpdate)

    def updateComputedDependencies(self):
        return self._root.updateComputedDependencies()

    def getFrame(self, key: str):
        return self._root.getFrame(key)

    def pendingColumns(self) -> list[str]:
        """Returns the columns that are currently being computed."""
        return [] if len(self._computingColumns) == 0 else self._computingColumns[-1]

    @property
    def columns(self):
        return self._root._columns

    @property
    def columnsAttributes(self):
        return self._root.columnsAttributes

    @property
    def shape(self):
        self._refreshIndex()
        return self._root.shape

    def __len__(self):
        return self.shape[0]

    def undo(self):
        return self._root.undo()

    def redo(self):
        return self._root.redo()

    def drop(self, id: Union[Hashable, Sequence[Hashable], pd.Index], skipLog=False):
        return self._root.drop(id, skipLog)

    def update(self, ids: Union[Hashable, Sequence[Hashable], pd.Index], value: Schema, replaceLog=False, skipLog=False):
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
class _SingleTimePointAnnotationsBase:
    _annotations: Annotations
    _t: int

    def __init__(self, annotations: Annotations, t: int):
        # abb dubug
        # this is where we run into problems on add/update spine and segment
        # when we add, we update self._annotations
        # but then self.points and self.segments are not updated?
        # they are another copy from SingleTimePointFrame

        # was this
        # self._annotations = copy(annotations)
        logger.warning('abb turned OFF copy of Annotations in _SingleTimePointAnnotationsBase()')
        self._annotations = annotations

        self._segments = SingleTimePointFrame(
            self._annotations._segments, t)
        self._points = SingleTimePointFrame(
            self._annotations._points, t)

        self._t = t
    
    @property
    def points(self) -> LazyGeoFrame:
        # logger.warning('SINGLE TIMEPOINT')
        return self._points

    @property
    def segments(self) -> LazyGeoFrame:
        # logger.warning('SINGLE TIMEPOINT')
        # print(self._annotations._segments)

        return self._segments

    # abb
    @property
    def analysisParams(self) -> AnalysisParams:
        return self._annotations._analysisParams

    # abb not used
    @property
    def timeSeries(self) -> Annotations:
        return self._annotations


Key = SpineId
Keys = Union[Key, list[Key]]


class SingleTimePointAnnotationsBase(_SingleTimePointAnnotationsBase):

    # abb
    def __str__(self):        
        numTimepoints = f'single timepoint ({self._t})'
        numPnts = len(self.points)
        numSegments = len(self.segments)
                
        return f't:{numTimepoints}, points:{numPnts} segments:{numSegments} images:{self.shape}'
        
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

    # abb
    def getAutoContrast_qt(self, channel: int) -> Tuple[int, int]:
        """Get the auto contrast from the entire image volume.
        
        Used in PyQt interface.
        """
        theMin, theMax = self._annotations.getAutoContrast_qt(time=self._t, channel=channel)

        return theMin, theMax
    
    @property
    def shape(self):
        return self._annotations._images.shape(self._t)

    # abb
    @property
    def numChannels(self) -> int:
        """Get the number of image channels.
        """
        return self._annotations._images.shape(self._t)[0]

    def getShapePixels(self, shapes: gp.GeoDataFrame, channel: Union[int, List[int]] = 0, zSpread: int = 0, z: int = None) -> pd.Series:
        return self._annotations.getShapePixels(shapes, channel, zSpread, self._t, z=z)

    def _mapKeys(self, keys: Keys) -> Keys:
        if isinstance(keys, list):
            _ret = [(id, self._t) for id in keys]
        else:
            _ret = (keys, self._t)
        # logger.warning(f'keys:{keys} is returning _ret:{_ret}')
        return _ret
    
    def deleteSpine(self, spineId: Keys, skipLog=False):
        return self._annotations.deleteSpine(self._mapKeys(spineId), skipLog)

    def getNumSpines(self, segmentId: Keys):
        return self._annotations.getNumSpines(self._mapKeys(segmentId))

    def deleteSegment(self, segmentId: Keys, skipLog=False):
        return self._annotations.deleteSegment(self._mapKeys(segmentId), skipLog)

    def updateSegment(self, segmentId: Keys, value: Segment, replaceLog=False, skipLog=False):
        # logger.warning(f'segmentId:{segmentId} self._mapKeys(segmentId):{self._mapKeys(segmentId)}')
        return self._annotations.updateSegment(self._mapKeys(segmentId), value, replaceLog, skipLog)

    def updateSpine(self, spineId: Keys, value: Spine, replaceLog=False, skipLog=False):
        
        _keys = self._mapKeys(spineId)  # yields (spineID, self._t)

        # if self._t in [1,2]:
        #     logger.info(f'{self}')
        #     logger.info(f'self._t:{self._t}')
        #     logger.info(f'_keys:{_keys}')
        #     logger.info(f'value:{value}')
            
        return self._annotations.updateSpine(_keys, value, replaceLog, skipLog)

    def connect(self, spineKey: SpineId, toSpineKey: Tuple[SpineId, int]):
        return self._annotations.connect((spineKey, self._t), toSpineKey)

    def disconnect(self, spineKey: SpineId):
        return self._annotations.disconnect((spineKey, self._t))

    def connectSegment(self, segmentKey: SegmentId, toSegmentKey: Tuple[SegmentId, int]):
        return self._annotations.connectSegment((segmentKey, self._t), toSegmentKey)

    def disconnectSegment(self, segmentKey: SegmentId):
        return self._annotations.disconnectSegment((segmentKey, self._t))

    def metadata(self) -> Metadata:
        return self._annotations._images.metadata(self._t)

    def getColors(self, colorOn: str = None, function=False) -> pd.Series:
        return Annotations.getColors(self, colorOn, function)

    def getSymbols(self, shapeOn: str, function=False) -> pd.Series:
        return Annotations.getSymbols(self, shapeOn, function)
