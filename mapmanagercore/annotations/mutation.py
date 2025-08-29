from typing import Tuple, Union, List
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
    def load_backward_compatible(cls, path: Union[str, None], lazy=False):
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
                    
                    # debug load
                    # logger.error('checking loaded points geodataframe (loaded from old file)')
                    # print('points is:')
                    # print(points)
                    # print('points.columns is:')
                    # print(points.columns)
                    # print('points.index is:')
                    # print(points.index)
                    # print('points.geometry is:')
                    # print(points.geometry)
                    # sys.exit(1)
                    # end debug load

                    points = gp.GeoDataFrame(points, geometry="point")
                    logger.info(f'Loaded points GeoDataFrame with {len(points)} rows and {len(points.columns)} columns')
                    # print(list(points.columns))
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
                    logger.info(f'Loaded lineSegments GeoDataFrame with {len(lineSegments)} rows and {len(lineSegments.columns)} columns')
                    # print(list(lineSegments.columns))
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
        Single-pass schema evolution: classify each column and apply appropriate action.
        This handles both missing columns (by adding them with None values) and
        deprecated columns (by removing them).
        """
        logger.info('Performing single-pass schema evolution...')
        
        # Single-pass evolution for points
        if len(self.points) > 0:
            self._evolve_schema_single_pass(self.points, "Spine")
        
        # Single-pass evolution for segments  
        if len(self.segments) > 0:
            self._evolve_schema_single_pass(self.segments, "Segment")

    def _evolve_schema_single_pass(self, lazy_frame, schema_name: str):
        """
        Single-pass schema evolution: classify each column and apply appropriate action.
        
        Args:
            lazy_frame: The LazyGeoFrame to evolve (points or segments)
            schema_name: The schema name ("Spine" or "Segment")
        """
        logger.info(f'Evolving schema for {schema_name} with single-pass approach...')
        
        # 1. Get current schema expectations
        current_basic_columns = Spine.getColumnNames(include_computed=False) if schema_name == "Spine" else Segment.getColumnNames(include_computed=False)
        current_computed_columns = Spine.getColumnNames(include_computed=True, include_basic=False) if schema_name == "Spine" else Segment.getColumnNames(include_computed=True, include_basic=False)
        current_image_columns = self._get_expected_image_columns(lazy_frame, schema_name)
        
        # logger.info(f'Current basic columns: {current_basic_columns}')
        logger.info(f'Current computed columns: {len(current_computed_columns)} expected')
        logger.info(f'Current image columns: {len(current_image_columns)} expected')
        
        # 2. Get the DataFrame and index columns to avoid duplicates
        current_df = lazy_frame._rootDf.copy()
        index_columns = list(current_df.index.names) if hasattr(current_df, 'index') and current_df.index.names else []
        logger.info(f'Index columns: {index_columns}')
        
        # 3. Classify each loaded column and build new DataFrame
        columns_to_remove = []
        columns_to_keep = []
        
        for column in current_df.columns:
            if column in current_basic_columns:
                # Keep basic schema columns
                columns_to_keep.append(column)
            elif column in current_computed_columns:
                # Keep stored computed columns (preserve pre-computed values)
                columns_to_keep.append(column)
                # logger.info(f'Preserving stored computed column: {column}')
            elif column in current_image_columns:
                # Keep image columns (preserve values)
                columns_to_keep.append(column)
            else:
                # Remove deprecated columns
                columns_to_remove.append(column)
        
        # 4. Remove deprecated columns
        if columns_to_remove:
            logger.warning(f'Removing {len(columns_to_remove)} deprecated {schema_name} columns: {columns_to_remove}')
            current_df = current_df.drop(columns=columns_to_remove)
        
        # 5. Add missing basic columns (but avoid index columns that already exist in index)
        missing_basic_columns = set(current_basic_columns) - set(current_df.columns)
        # Filter out index columns that are already in the index
        missing_basic_columns = missing_basic_columns - set(index_columns)
        
        if missing_basic_columns:
            logger.warning(f'Adding {len(missing_basic_columns)} missing basic {schema_name} columns: {missing_basic_columns}')
            for column in missing_basic_columns:
                current_df[column] = None
        
        # 6. Add missing image columns using addSchema()
        self._add_missing_image_columns(lazy_frame, schema_name)
        
        # 7. Update the DataFrame
        lazy_frame._rootDf = current_df
        
        # logger.info(f'Completed {schema_name} schema evolution: {len(current_df.columns)} total columns')

    def _get_expected_image_columns(self, lazy_frame, schema_name: str) -> List[str]:
        """
        Get expected image columns for a given schema.
        """
        store = lazy_frame.getStore()
        if hasattr(store, 'getImageColumnNames'):
            schema_class = Spine if schema_name == "Spine" else Segment
            return store.getImageColumnNames(schema_class)
        return []

    def _add_missing_image_columns(self, lazy_frame, schema_name: str):
        """
        Add missing image columns using the existing addSchema logic.
        """
        store = lazy_frame.getStore()
        if hasattr(store, 'addSchema'):
            # Get channel keys from the metadata
            # Get the first timepoint from the metadata
            timepoints = list(self.loader.timePoints())
            if timepoints:
                timepoint = timepoints[0]
                # Get channel keys for this timepoint
                channelKeys = self.loader.metadata.getTimepoint(timepoint).channelKeys
                logger.info(f'Using timepoint {timepoint} with {len(channelKeys)} channels from metadata for {schema_name}')
            else:
                logger.warning(f'No timepoints in metadata for {schema_name}')
                channelKeys = []
            
            if channelKeys:
                # logger.info(f'Adding missing image columns for {schema_name} using addSchema...')
                store.addSchema(lazy_frame, channelKeys)
            else:
                logger.warning(f'No channel keys available for {schema_name}, skipping addSchema')
        else:
            logger.warning(f'Store does not have addSchema method for {schema_name}')

    # OLD COMPLEX METHODS - REMOVED
    # The following methods have been replaced by the simplified single-pass approach:
    # - _filter_basic_schema_columns()
    # - _regenerate_image_columns_with_values() 
    # - _remove_remaining_deprecated_columns()
    # - _capture_existing_image_values()
    # - _restore_image_values()
    # - _get_expected_columns()

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
            logger.info(f'Computing {len(missing_point_columns)} missing point columns:')
            print(missing_point_columns)
            _ = self.points[list(missing_point_columns)]  # This triggers computation
        elif missing_point_columns:
            logger.info(f'No point data to compute {len(missing_point_columns)} missing columns')
        
        # Trigger computation for segments
        if missing_segment_columns and len(self.segments) > 0:
            logger.info(f'Computing {len(missing_segment_columns)} missing segment columns:')
            print(missing_segment_columns)
            _ = self.segments[list(missing_segment_columns)]  # This triggers computation
        elif missing_segment_columns:
            logger.info(f'No segment data to compute {len(missing_segment_columns)} missing columns')

    # abc 20250806 - Enhanced saving methods with computed columns
    def save_with_computed_columns(self, path: str = None):
        """
        Save the mmap with all computed columns pre-computed.
        
        This ensures that all computed columns are calculated and saved to the file,
        making loading more efficient and ensuring backward compatibility.
        
        Args:
            path: Path to save to (if None, uses self.path)
        """
        logger.info(f'Saving with computed columns to: {path or self.path}')
        
        # 1. Trigger computation of all computed columns
        self._compute_all_computed_columns()
        
        # 2. Call the existing save method
        self.save(path)
        
        logger.info('Successfully saved with computed columns')

    def _compute_all_computed_columns(self):
        """
        Force computation of all computed columns in both points and segments.
        This ensures all computed data is available for saving.
        """
        logger.info('Computing all computed columns before save...')
        
        # Get all computed columns from schemas
        computed_point_columns = Spine.getColumnNames(include_computed=True, include_basic=False)
        computed_segment_columns = Segment.getColumnNames(include_computed=True, include_basic=False)
        
        total_computed = 0
        
        # Trigger computation for points
        if computed_point_columns and len(self.points) > 0:
            logger.info(f'Computing {len(computed_point_columns)} point columns before save: {computed_point_columns}')
            _ = self.points[computed_point_columns]  # This triggers computation
            total_computed += len(computed_point_columns)
        elif computed_point_columns:
            logger.info(f'No point data to compute {len(computed_point_columns)} columns')
        
        # Trigger computation for segments  
        if computed_segment_columns and len(self.segments) > 0:
            logger.info(f'Computing {len(computed_segment_columns)} segment columns before save: {computed_segment_columns}')
            _ = self.segments[computed_segment_columns]  # This triggers computation
            total_computed += len(computed_segment_columns)
        elif computed_segment_columns:
            logger.info(f'No segment data to compute {len(computed_segment_columns)} columns')
        
        logger.info(f'Completed computation of {total_computed} computed columns')

    def _get_computed_columns_status(self):
        """
        Get the status of computed columns for debugging and verification.
        
        Returns:
            dict: Status information about computed columns
        """
        # Get all computed columns from schemas
        computed_point_columns = Spine.getColumnNames(include_computed=True, include_basic=False)
        computed_segment_columns = Segment.getColumnNames(include_computed=True, include_basic=False)
        
        # Check which computed columns are currently available
        available_point_columns = set(self.points.columns) if len(self.points) > 0 else set()
        available_segment_columns = set(self.segments.columns) if len(self.segments) > 0 else set()
        
        # Find missing computed columns
        missing_point_columns = set(computed_point_columns) - available_point_columns
        missing_segment_columns = set(computed_segment_columns) - available_segment_columns
        
        status = {
            'total_computed_point_columns': len(computed_point_columns),
            'total_computed_segment_columns': len(computed_segment_columns),
            'available_point_columns': len(available_point_columns),
            'available_segment_columns': len(available_segment_columns),
            'missing_point_columns': list(missing_point_columns),
            'missing_segment_columns': list(missing_segment_columns),
            'all_computed_available': len(missing_point_columns) == 0 and len(missing_segment_columns) == 0
        }
        
        return status

    def columnIsComputed(self, column: str, frame_type: str = "points") -> bool:
        """
        Check if a column is computed (has valid data) or needs computation.
        
        Args:
            column: Name of the column to check
            frame_type: Either "points" or "segments"
            
        Returns:
            bool: True if column is computed (valid), False if needs computation
        """
        if frame_type == "points":
            frame = self.points
        elif frame_type == "segments":
            frame = self.segments
        else:
            raise ValueError(f"frame_type must be 'points' or 'segments', got {frame_type}")
        
        if frame is None or len(frame) == 0:
            return False
            
        df = frame._rootDf
        valid_key = f"{column}.valid"
        
        # Check if the .valid column exists and has values
        if valid_key in df.columns:
            # Column is computed if .valid column exists and has non-null values
            return not df[valid_key].isna().all()
        
        return False

    def verify_computed_columns_before_save(self):
        """
        Verify that all computed columns are available before saving.
        This is a debugging method to check the status of computed columns.
        
        Returns:
            dict: Status information about computed columns
        """
        return self._get_computed_columns_status()


