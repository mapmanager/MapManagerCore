from typing import Tuple, Union, List
# from shapely.geometry import Point

from ..schemas import Spine, Segment
from ..config import SegmentId, SpineId
from ..lazy_geo_pandas import LazyGeoFrame
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
    def load_backward_compatible(cls, path: str, lazy: bool = False) -> 'AnnotationsBaseMut':
        """
        Load MapAnnotations with hybrid backward compatibility for schema evolution.
        
        This method handles loading data from older file formats by:
        1. Loading raw data with enhanced error handling
        2. Filtering to current schema (removing deprecated columns, adding missing ones)
        3. Marking loaded computed columns as valid
        4. Triggering computation of missing computed columns
        
        Args:
            path: Path to the .mmap file (required)
            lazy: Whether to use lazy loading (currently unused, kept for compatibility)
            
        Returns:
            AnnotationsBaseMut: Fully loaded MapAnnotations instance with up-to-date schema
            
        Raises:
            FileNotFoundError: If the specified path doesn't exist
            ValueError: If the file format is invalid or corrupted
            
        Example:
            >>> mmap = MapAnnotations.load_backward_compatible("old_file.mmap")
            >>> print(f"Loaded {len(mmap.points)} points and {len(mmap.segments)} segments")
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
    def _load_raw_data_with_backward_compatibility(cls, path: str, lazy: bool = False) -> 'AnnotationsBaseMut':
        """
        Load raw data from file with enhanced error handling for backward compatibility.
        
        This is the core loading method that handles the actual file I/O operations.
        It provides better error handling and logging compared to the standard load() method.
        
        Args:
            path: Path to the .mmap file (required)
            lazy: Whether to use lazy loading (currently unused, kept for compatibility)
            
        Returns:
            AnnotationsBaseMut: Partially loaded instance (schema evolution not yet applied)
            
        Raises:
            FileNotFoundError: If the specified path doesn't exist
            ArrowInvalid: If the parquet data is corrupted
            KeyError: If required file attributes are missing
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

    def _filter_to_current_schema(self) -> None:
        """
        Apply single-pass schema evolution to both points and segments.
        
        This method performs schema evolution by:
        1. Removing deprecated columns that are no longer in the current schema
        2. Adding missing basic columns with None values
        3. Adding missing image columns using the store's addSchema method
        
        The evolution is applied to both points (Spine schema) and segments (Segment schema)
        in a single pass for efficiency.
        
        Note:
            This method modifies the underlying DataFrames in-place.
            It should be called after loading raw data but before accessing computed columns.
        """
        logger.info('Performing single-pass schema evolution...')
        
        # Single-pass evolution for points
        if len(self.points) > 0:
            self._evolve_schema_single_pass(self.points, "Spine")
        
        # Single-pass evolution for segments  
        if len(self.segments) > 0:
            self._evolve_schema_single_pass(self.segments, "Segment")

    def _evolve_schema_single_pass(self, lazy_frame: LazyGeoFrame, schema_name: str) -> None:
        """
        Perform single-pass schema evolution on a specific frame.
        
        This method classifies each column in the frame and applies the appropriate action:
        - Keep basic schema columns (preserve values)
        - Keep stored computed columns (preserve pre-computed values)
        - Keep image columns (preserve values)
        - Remove deprecated columns (no longer in current schema)
        - Add missing basic columns (with None values)
        - Add missing image columns (using store's addSchema)
        
        Args:
            lazy_frame: The LazyGeoFrame to evolve (points or segments)
            schema_name: The schema name ("Spine" or "Segment")
            
        Note:
            This method modifies the lazy_frame._rootDf in-place.
            Index columns are preserved and not duplicated.
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

    def _get_expected_image_columns(self, lazy_frame: LazyGeoFrame, schema_name: str) -> List[str]:
        """
        Get the list of expected image columns for a given schema.
        
        This method queries the store to determine what image columns should exist
        for the given schema (Spine or Segment) based on the available channels.
        
        Args:
            lazy_frame: The LazyGeoFrame to check (points or segments)
            schema_name: The schema name ("Spine" or "Segment")
            
        Returns:
            List[str]: List of expected image column names for the schema
            
        Note:
            Returns empty list if the store doesn't support getImageColumnNames.
        """
        store = lazy_frame.getStore()
        if hasattr(store, 'getImageColumnNames'):
            schema_class = Spine if schema_name == "Spine" else Segment
            return store.getImageColumnNames(schema_class)
        return []

    def _add_missing_image_columns(self, lazy_frame: LazyGeoFrame, schema_name: str) -> None:
        """
        Add missing image columns to a frame using the store's addSchema method.
        
        This method uses the store's addSchema functionality to create missing image columns
        (like spineRoi_ch1_sum, spineRoi_ch1_mean, etc.) based on the available channels
        in the metadata.
        
        Args:
            lazy_frame: The LazyGeoFrame to add columns to (points or segments)
            schema_name: The schema name ("Spine" or "Segment")
            
        Note:
            This method requires the store to have an addSchema method.
            It uses the first timepoint's channel keys from the metadata.
            If no timepoints or channels are available, the operation is skipped.
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

    def _trigger_missing_computed_columns(self) -> None:
        """
        Trigger computation of missing computed columns after loading.
        
        This method ensures all computed columns are available by:
        1. Identifying computed columns that are missing from the loaded data
        2. Triggering their computation through the lazy system
        3. Logging which columns are being computed
        
        The method handles both points (Spine schema) and segments (Segment schema).
        Only columns that are missing (not in the loaded data) are computed.
        
        Note:
            This method should be called after schema evolution is complete.
            It only computes columns that weren't loaded from the file.
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


    def _compute_all_computed_columns(self) -> None:
        """
        Force computation of all computed columns in both points and segments.
        
        This method ensures all computed data is available for saving by:
        1. Getting all computed columns from both Spine and Segment schemas
        2. Triggering computation of all computed columns (not just missing ones)
        3. Logging the computation progress and results
        
        Unlike _trigger_missing_computed_columns(), this method computes ALL computed
        columns regardless of whether they were loaded from the file or not.
        
        Note:
            This method is typically called before saving to ensure all computed
            values are written to the file rather than placeholder values.
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

    def _get_computed_columns_status(self) -> dict:
        """
        Get comprehensive status information about computed columns.
        
        This method provides debugging and verification information about the state
        of computed columns in both points and segments frames.
        
        Returns:
            dict: Status information containing:
                - total_computed_point_columns: Number of computed columns expected for points
                - total_computed_segment_columns: Number of computed columns expected for segments
                - available_point_columns: Number of columns currently available in points
                - available_segment_columns: Number of columns currently available in segments
                - missing_point_columns: List of missing computed columns for points
                - missing_segment_columns: List of missing computed columns for segments
                - all_computed_available: Boolean indicating if all computed columns are available
                
        Example:
            >>> status = mmap._get_computed_columns_status()
            >>> print(f"Missing point columns: {status['missing_point_columns']}")
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
        Check if a specific column is computed (has valid data) or needs computation.
        
        This method checks the lazy system's version tracking to determine if a column
        has been computed and is up-to-date. It looks for the corresponding `.valid`
        column that contains version information.
        
        Args:
            column: Name of the column to check (e.g., 'spineLength', 'spineRoi_ch1_sum')
            frame_type: Frame to check - either "points" or "segments"
            
        Returns:
            bool: True if column is computed and valid, False if it needs computation
            
        Raises:
            ValueError: If frame_type is not "points" or "segments"
            
        Example:
            >>> mmap.columnIsComputed('spineLength', 'points')
            True
            >>> mmap.columnIsComputed('spineRoi_ch1_shape', 'points')
            False
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

    def verify_computed_columns_before_save(self) -> dict:
        """
        Verify the status of computed columns before saving.
        
        This is a debugging method that provides detailed information about the state
        of computed columns to help diagnose issues before saving.
        
        Returns:
            dict: Comprehensive status information about computed columns
                (same format as _get_computed_columns_status())
                
        Example:
            >>> status = mmap.verify_computed_columns_before_save()
            >>> if not status['all_computed_available']:
            ...     print(f"Missing columns: {status['missing_point_columns']}")
        """
        return self._get_computed_columns_status()


