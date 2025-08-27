from typing import Tuple, Union
# from shapely.geometry import Point

from ..schemas import Spine, Segment
from ..config import SegmentId, SpineId
from .base import AnnotationsBase

from mapmanagercore.logger import logger

Key = Union[SpineId, Tuple[SpineId, int]]
Keys = Union[Key, list[Key]]


# abb 202508 this is imported as MapAnnotations, like:
# from mapmanagercore.annotations import MapAnnotations
class AnnotationsBaseMut(AnnotationsBase):
    
    def deleteSpine(self, spineId: Keys, skipLog=False):
        """
        Delete the spine with the given ID.
        """
        self._drop("Spine", spineId, skipLog=skipLog)

    # abb convenience
    def getNumSpines(self, segmentId : Keys) -> int:
        """Get number of spines on a segment.
        
        TODO: Too complicated
        """
        try:
            _spines = self.points[["segmentID"]].reset_index().set_index(["segmentID", "t"]).loc[segmentId]
            _numSpines = len(_spines)
        except (KeyError) as e:
            _numSpines = 0
        return _numSpines
    
    def deleteSegment(self, segmentId: Keys, skipLog=False, forceDelete=False):
        """
        Delete the segment with the given ID.
        
        If forceDelete is True, delete the segment even if it has an attached spine.

        abb forceDelete is not implemented, only segment is deleted not spines!
        """
        try:
            # abb TODO use getNumSpines(segmentId)
            # abb multi timepoint error
            if not forceDelete and not self.points[["segmentID"]].reset_index().set_index(["segmentID", "t"]).loc[segmentId].empty:
                logger.warning(f'Cannot delete segment(s) {segmentId} as it has an attached spine(s)')
                return False
                # raise ValueError(
                #     f"Cannot delete segment(s) {segmentId} as it has an attached spine(s)")
        except (KeyError) as e:
            logger.error(f'deleteSegment error: {e}')
            pass

        self._drop("Segment", segmentId, skipLog=skipLog)
        return True
    
    def updateSpine(self, spineId: Keys, value: Spine, replaceLog=False, skipLog=False):
        """
        Set the spine with the given ID to the specified value.            
        """
        return self._update("Spine", spineId, value, replaceLog, skipLog)

    def updateSegment(self, segmentId: Keys, value: Segment, replaceLog=False, skipLog=False):
        """Set the segment with the given ID to the specified value.

        Args:
            segmentId (str): The ID of the spine.
            value (Union[dict, gp.Series, pd.Series]): The value to set for the spine.
        """
        return self._update("Segment", segmentId, value, replaceLog, skipLog)

    def newUnassignedSpineId(self) -> SpineId:
        """
        Returns a new unassigned spine ID.
        """
        if len(self.points) == 0:
            return 1 # start from 1 to avoid confusion with counting from 0
        return max(1, self.points.index.get_level_values(0).max() + 1)

    def newUnassignedSegmentId(self) -> SegmentId:
        """
        Returns a new unassigned segment ID.
        """
        if len(self.segments) == 0:
            return 1 # start from 1 to avoid confusion with counting from 0
        return max(1, self.segments.index.get_level_values(0).max() + 1)

    def connect(self, spineKey: Tuple[SpineId, int], toSpineKey: Tuple[SpineId, int]):
        
        # ValueError: Can only compare identically-labeled Series objects
        # if self.points[toSpineKey, "segmentID"] != self.points[spineKey, "segmentID"]:
        
        _segmentID = self.points[spineKey, "segmentID"]
        # abb multi timepoint error was this
        # _segmentID = self.points[spineKey, "segmentID"].values[0]
        
        _toSegmentID = self.points[toSpineKey, "segmentID"]
        
        if _segmentID != _toSegmentID:
            logger.warning(f'Cannot connect spines from different segments. Got segments {_segmentID} and {_toSegmentID}')
            return False
        
        # check if the key already exists in the time point
        existingKey = (spineKey[0], toSpineKey[1])
        #existingKey = (toSpineKey[0], spineKey[1])
        if existingKey in self.points.index:
            logger.info(f'disconnecting existingKey:{existingKey}')
            self.disconnect(existingKey)

        # Propagate the spine ID to all future time points
        # _slice = slice(spineKey, spineKey[0])
        _slice = toSpineKey  # this will not get toSpineKEy[0] at future timepoints

        _spine = Spine(
            # spineID=toSpineKey[0],
            spineID=spineKey[0],
        )
        
        # logger.info(f'   _slice:{_slice}')
        # logger.info(f'   _spine:{_spine}')
        
        self.updateSpine(_slice, _spine)

        return True
    
    def disconnect(self, spineKey: Tuple[SpineId, int]):
        newID = self.newUnassignedSpineId()

        # Propagate the spine ID change to all future time points
        self.updateSpine(slice(spineKey, spineKey[0]), Spine(
            spineID=newID,
        ))

    def connectSegment(self, segmentKey: Tuple[SegmentId, int], toSegmentKey: Tuple[SegmentId, int]):
        logger.info(f'segmentKey:{segmentKey}')
        logger.info(f'toSegmentKey:{toSegmentKey}')
        # logger.info('self.segments.index')
        # print(self.segments.index)

        if segmentKey[1] == toSegmentKey[1]:
            # raise ValueError(
            #     "Cannot connect segments in the same time points.")
            logger.warning('Cannot connect segments in the same time points.')
            return
        
        newPostKey = (segmentKey[0], toSegmentKey[1])
        if newPostKey in self.segments.index:
            logger.warning(f'newPostKey:{newPostKey} already in index')
            return

        # TODO if segment key is connect downstream -> disconnect
        # TODO if toSegmentKey key is connect upstream -> disconnect
        # check if the key already exists in the time point
        existingKey = (toSegmentKey[0], segmentKey[1])
        if 0 and existingKey in self.segments.index:
            logger.info(f'   calling disconnectSegment() for existingKey:{existingKey}')
            self.disconnectSegment(existingKey)

        # Propagate the segment ID to all future time points
        # was this
        # _slice = slice(segmentKey, segmentKey[0])
        # logger.info(f'   _slice:{(segmentKey, segmentKey[0])}')
        # abb multi timepoint error connectSegment works for transient (1 tp) segments,
        #   does not get any other downstream
        _slice = toSegmentKey  # ('bar',)

        _segment = Segment(
            # was this
            # segmentID=toSegmentKey[0],
            # abb multi timepoint error was this
            segmentID=segmentKey[0],
        )
        # logger.info(f'   _segment:{_segment}')

        self.updateSegment(_slice, _segment)

        spineRows = self.points[ self.points['segmentID'] == toSegmentKey[0]].index
        _spine = Spine(
            segmentID=segmentKey[0],
        )

        # logger.warning(f'todo: update all SPINES with segmentID:{toSegmentKey} to {segmentKey}')
        # print('spineRows:')
        # print(spineRows)
        # print('_spine')
        # print(_spine)

        self.updateSpine(spineRows, _spine)

        return True
    
    def disconnectSegment(self, segmentKey: Tuple[SegmentId, int]):
        newID = self.newUnassignedSegmentId()

        # Propagate the segment ID change to all future time points
        self.updateSegment(slice(segmentKey, segmentKey[0]), Segment(
            segmentID=newID,
        ))

    # abc 20250806 - Backward compatibility loading methods
    @classmethod
    def load_with_hybrid_backward_compatibility(cls, path: Union[str, None], lazy=False):
        """
        Load MapAnnotations with hybrid backward compatibility for schema changes.
        
        This method handles:
        1. Loading data from older file formats that may be missing current columns
        2. Filtering out deprecated columns that are no longer in the current schema
        3. Automatically computing missing computed columns after load
        
        Args:
            path: Path to the .mmap file
            lazy: Whether to use lazy loading (currently unused but kept for compatibility)
            
        Returns:
            MapAnnotations instance with up-to-date schema
        """
        logger.info(f'Loading with hybrid backward compatibility from: {path}')
        
        # 1. Load raw data using existing load logic
        annotations = cls._load_raw_data_with_backward_compatibility(path, lazy)
        
        # 2. Filter to current schema columns
        annotations._filter_to_current_schema()
        
        # 3. Trigger computation of missing computed columns
        annotations._trigger_missing_computed_columns()
        
        logger.info('Successfully loaded with backward compatibility')
        return annotations

    @classmethod
    def _load_raw_data_with_backward_compatibility(cls, path: Union[str, None], lazy=False):
        """
        Load raw data from file using existing load logic but with enhanced error handling.
        
        This is essentially the same as the existing load() method but with better
        error handling and logging for backward compatibility scenarios.
        """
        from ..lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader
        import os
        import zarr
        from io import BytesIO
        import geopandas as gp
        import pyarrow
        from pyarrow.lib import ArrowInvalid
        
        # Create loader
        loader = mmMapLoader(path)

        if path.endswith('.tif'):
            logger.warning(f'MapAnnotations from tif file:{path}')
            lineSegments = gp.GeoDataFrame()
            points = gp.GeoDataFrame()
            lastSaveTime = ''
            return cls(loader, lineSegments, points, path, lastSaveTime)
        
        # Determine store type
        if os.path.isdir(path):
            _zipStore = False
        else:
            _zipStore = True
        
        # Load data from zarr
        with zarr.ZipStore(path, mode="r") if _zipStore else zarr.DirectoryStore(path) as store:
            _group: zarr.hierarchy.Group = zarr.open(store=store, mode='r')
            
            # Load points with enhanced error handling
            if "points" in _group:
                try:
                    points = gp.read_parquet(BytesIO(_group["points"][:].tobytes()))
                    points = gp.GeoDataFrame(points, geometry="point")
                    logger.info(f'Loaded points with {len(points)} rows and columns: {list(points.columns)}')
                except (ArrowInvalid) as e:
                    logger.error(f'Error reading points: {e}')
                    points = gp.GeoDataFrame()
            else:
                logger.info('No points found in file')
                points = gp.GeoDataFrame()

            # Load segments with enhanced error handling
            if "lineSegments" in _group:
                try:
                    lineSegments = gp.read_parquet(BytesIO(_group["lineSegments"][:].tobytes()))
                    lineSegments = gp.GeoDataFrame(lineSegments, geometry="segment")
                    logger.info(f'Loaded segments with {len(lineSegments)} rows and columns: {list(lineSegments.columns)}')
                except (ArrowInvalid) as e:
                    logger.error(f'Error reading lineSegments: {e}')
                    lineSegments = gp.GeoDataFrame()
            else:
                logger.info('No segments found in file')
                lineSegments = gp.GeoDataFrame()
            
            # Load lastSaveTime
            try:
                lastSaveTime = _group.attrs['lastSaveTime']
            except (KeyError) as e:
                logger.error(f'KeyError: did not get "lastSaveTime":{e}')
                lastSaveTime = ""

        return cls(loader, lineSegments, points, path, lastSaveTime)

    def _filter_to_current_schema(self):
        """
        Filter loaded data to only include columns that are in the current runtime schema.
        This handles both missing columns (by adding them with None values) and
        deprecated columns (by removing them).
        """
        # Get current runtime schema columns (basic columns only)
        current_point_columns = Spine.getColumnNames(include_computed=False)
        current_segment_columns = Segment.getColumnNames(include_computed=False)
        
        logger.info(f'Current point schema columns: {current_point_columns}')
        logger.info(f'Current segment schema columns: {current_segment_columns}')
        
        # Filter points to current schema
        if len(self.points) > 0:
            # Find deprecated columns (in loaded data but not in current schema)
            deprecated_point_columns = set(self.points.columns) - set(current_point_columns)
            if deprecated_point_columns:
                logger.warning(f'Removing deprecated point columns: {deprecated_point_columns}')
            
            # Use reindex to handle missing columns gracefully (adds None values)
            filtered_points = self.points.reindex(columns=current_point_columns, fill_value=None)
            self.points.loadData(filtered_points)
            logger.info(f'Filtered points to {len(filtered_points.columns)} columns')
        
        # Filter segments to current schema
        if len(self.segments) > 0:
            # Find deprecated columns (in loaded data but not in current schema)
            deprecated_segment_columns = set(self.segments.columns) - set(current_segment_columns)
            if deprecated_segment_columns:
                logger.warning(f'Removing deprecated segment columns: {deprecated_segment_columns}')
            
            # Use reindex to handle missing columns gracefully (adds None values)
            filtered_segments = self.segments.reindex(columns=current_segment_columns, fill_value=None)
            self.segments.loadData(filtered_segments)
            logger.info(f'Filtered segments to {len(filtered_segments.columns)} columns')

    def _trigger_missing_computed_columns(self):
        """
        Trigger computation of missing computed columns after loading.
        This ensures all computed columns are available even if they weren't in the saved file.
        """
        # Get all computed columns from schemas
        computed_point_columns = Spine.getColumnNames(include_computed=True, include_basic=False)
        computed_segment_columns = Segment.getColumnNames(include_computed=True, include_basic=False)
        
        # Find missing computed columns
        missing_point_columns = set(computed_point_columns) - set(self.points.columns)
        missing_segment_columns = set(computed_segment_columns) - set(self.segments.columns)
        
        # Trigger computation for points
        if missing_point_columns and len(self.points) > 0:
            logger.info(f'Computing {len(missing_point_columns)} missing point columns: {missing_point_columns}')
            _ = self.points[list(missing_point_columns)]  # This triggers computation
        elif missing_point_columns:
            logger.info(f'No point data to compute {len(missing_point_columns)} missing columns')
        
        # Trigger computation for segments
        if missing_segment_columns and len(self.segments) > 0:
            logger.info(f'Computing {len(missing_segment_columns)} missing segment columns: {missing_segment_columns}')
            _ = self.segments[list(missing_segment_columns)]  # This triggers computation
        elif missing_segment_columns:
            logger.info(f'No segment data to compute {len(missing_segment_columns)} missing columns')
