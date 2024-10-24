from datetime import datetime
import os
from copy import copy
from io import BytesIO
from typing import Any, Tuple, Union, Optional
import zipfile
import numpy as np
import pandas as pd

from mapmanagercore.benchmark import timer
from mapmanagercore.config import Colors, scaleColors, symbols
from mapmanagercore.lazy_geo_pd_images.loader.zarr import ZarrLoader
from ..lazy_geo_pandas import LazyGeoFrame
from ..schemas import Segment, Spine
from ..lazy_geo_pd_images import LazyImagesGeoPandas, ImageLoader
from ..lazy_geo_pd_images.image_slices import ImageSlice
import zarr
import warnings
from plotly.express.colors import sample_colorscale
import geopandas as gp

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.logger import logger

class AnnotationsBase(LazyImagesGeoPandas):
    _images: ImageLoader

    def __init__(self,
                 loader: ImageLoader,
                 lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(),
                 points: Union[str, pd.DataFrame] = pd.DataFrame(),
                 analysisParams: AnalysisParams = AnalysisParams(),
                 lastSaveTime: str = ""
                 ):

        super().__init__(loader)

        if not isinstance(lineSegments, gp.GeoDataFrame):
            if not isinstance(lineSegments, pd.DataFrame):
                lineSegments = pd.read_csv(lineSegments, index_col=False)

        if not isinstance(points, gp.GeoDataFrame):
            if not isinstance(points, pd.DataFrame):
                points = pd.read_csv(points, index_col=False)

        # abj
        self._lastSaveTime = lastSaveTime

        # abb analysisparams
        self._analysisParams: AnalysisParams = analysisParams

        self._segments = LazyGeoFrame(
            Segment, data=lineSegments, store=self)
        
        # logger.warning(f'base.py AnnotationsBase is making points from: {type(points)}')
        # print(points.columns)
        # print('points is:')
        # print(points)

        self._points = LazyGeoFrame(Spine, data=points, store=self)

        self.loader = loader

    # abj
    def getLastSaveTime(self):
        """
        """
        # get last save time from attributes
        return self._lastSaveTime 

    # abb
    def getNumTimepoints(self):
        return len(self._images.timePoints())

    # abb
    def getPointDataFrame(self, t : Optional[int] = None) -> pd.DataFrame:
        """Get the full points dataframe.
        """
        pointsDf = self.points[:]
        
        if t is not None:

            # move (,t) index into a column
            pointsDf = pointsDf.reset_index(level=1)
            # reduce my t==t
            pointsDf = pointsDf[ pointsDf['t']==t ]
        
        return pointsDf
    
    # abb
    def __str__(self):
        """Print info about the map.
        
        See: _SingleTimePointAnnotationsBase()
        """
        timePoints = self._images.timePoints()
        numTimepoints = len(timePoints)
        numPnts = len(self.points)
        numSegments = len(self.segments)
        
        theRet =  f'mmmap t:[{numTimepoints}], points:{numPnts} segments:{numSegments}\n'
        for tpIdx in timePoints:
            tp = self.getTimePoint(time=tpIdx)
            theRet += f'      {tp}\n'
        return theRet
    
    @property
    def segments(self) -> LazyGeoFrame:
        return self._segments

    @property
    def points(self) -> LazyGeoFrame:
        return self._points
    
    @property
    def analysisParams(self) -> AnalysisParams:
        return self._analysisParams
    
    def filterPoints(self, filter: Any):
        """
        Filters the points.
        """
        # logger.error(f'abb copy memory #1 filter:{filter} {type(filter)}')
        
        c = copy(self)
        c._points = c._points[filter]
        return c

    def filterSegments(self, filter: Any):
        """
        Filters the segments.
        """
        # logger.error(f'abb copy memory #2 filter:{filter} {type(filter)}')
        c = copy(self)
        c._segments = c._segments[filter]
        return c

    def getTimePoint(self, time: int):
        """
        Returns the annotations for a single time point.
        """
        from .single_time_point import SingleTimePointAnnotations
        return SingleTimePointAnnotations(self, time)
    
    def getPixels(self, time: int, channel: int, zRange: Tuple[int, int] = None, z: int = None, zSpread: int = 0) -> ImageSlice:
        """
        Loads the image data for a slice.

        Args:
          time (int): The time slot index.
          channel (int): The channel index.
          zRange (Tuple[int, int]): The visible z slice range.
          z (int): The z slice index.
          zSpread (int): The amount to offset z +/-.

        Returns:
          ImageSlice: The image slice.
        """

        if zRange is None:
            if z is not None:
                zRange = (z-zSpread, z+zSpread)
            else:
                zRangeDf = self.points["z"]
                zRange = (int(zRangeDf.min()),
                          int(zRangeDf.max()))

        return super().getPixels(time, channel, zRange)

    # Serialization

    # abb
    @classmethod
    def checkFile(cls, path: str, lazy=True, verbose=False) -> bool:
        """Check if a zarr file is valid to load.

        This is a complex function, Python has never been good at this?
        Is there a better way to write it?
        """
        from json import JSONDecodeError
        from pickle import UnpicklingError
        from mapmanagercore.lazy_geo_pd_images.metadata import Metadata

        _errors = 0

        if verbose:
            logger.info(f'inspecting zarr file: {path}')

        # (1) taken from ZarrLoader (it loads images)
        # loader = ZarrLoader(path, lazy=lazy)
        # error when trying to load a zipstore
        # zarr.errors.FSPathExistNotDir: path exists but is not a directory: %r
        if os.path.isdir(path):
            store = zarr.DirectoryStore(path)
        else:
            store = zarr.ZipStore(path, mode="r")
        
        group = zarr.group(store=store)

        if verbose:
            logger.info('file has the following groups')
            for _key, _value in group.items():
                logger.info(f'  {_key}: {_value}')

            logger.info('file has the following attrs keys')
            for _key, _value in group.attrs.items():
                logger.info(f'  {_key}: {_value}')

        _imagesSrcs = {}
        _metadata = {}
        for t in group.attrs["timePoints"]:
            try:
                images = group[f"img-{t}"]
            except (KeyError) as e:
                logger.error(f'did not find group "img-{t}"')
                logger.error(f'   {e}')
                _errors += 1
            finally:
                _imagesSrcs[t] = images if lazy else images[:]
                if verbose:
                    logger.info(f'img-{t}: {_imagesSrcs[t].shape}')

            try:
                group.attrs[f"metadata-{t}"]
            except (KeyError) as e:
                logger.error(f'did not find group "metadata-{t}"')
                logger.error(f'   {e}')
                _errors += 1
            finally:
                _metadata[t] = Metadata.from_json(group.attrs[f"metadata-{t}"])
                if verbose:
                    logger.info(f'metadata-{t}: {_metadata[t]}')

        # (2) points
        try:
            _points = group["points"]  # zarr.core.Array '/points' (255865,) uint8
        except (KeyError) as e:
            logger.error('did not find group "points"')
            logger.error(f'   {e}')
            _errors += 1
        finally:
            try:
                _points = pd.read_pickle(BytesIO(_points[:].tobytes()))
                if verbose:
                    logger.info(f'points: {len(_points)}')
                    # print(_points.head())
            except (UnpicklingError) as e:
                logger.error('error reading pickel from points')
                logger.error(f'   {e}')
                _errors += 1

        # (3) lineSegments
        try:
            _lineSegments = group["lineSegments"]
        except (KeyError) as e:
            logger.error('did not find group "lineSegments"')
            logger.error(f'   {e}')
            _errors += 1
        finally:
            try:
                _lineSegments = pd.read_pickle(BytesIO(_lineSegments[:].tobytes()))
                if verbose:
                    logger.info(f'lineSegments: {len(_lineSegments)}')
                    # print(_lineSegments.head())
            except (UnpicklingError) as e:
                logger.error('error reading pickel from lineSegments')
                logger.error(f'   {e}')
                _errors += 1

        # (4) analysisParams
        try:
            _analysisParams_json = group.attrs['analysisParams']
        except (KeyError) as e:
            logger.error('did not find "analysisParams"')
            logger.error(f'   {e}')
            _errors += 1
        finally:
            try:
                analysisParams = AnalysisParams(loadJson=_analysisParams_json)
            except (JSONDecodeError) as e:
                logger.error('did not parse json into AnalysisParams()')
                logger.error(f'   {e}')
                _errors += 1

        if verbose:
            logger.info(f'encountered {_errors} errors while inspecting {path}')

        return _errors == 0
    
    @classmethod
    def load(cls, path: str, lazy=False, version:int=0):
        """
        Parameters
        ----------
        version : int
            Verion to load, 0 (default) is original
        """
        logger.info(f'lazy:{lazy} path:{path}')

        loader = ZarrLoader(path, lazy=lazy)
        
        # logger.info(f'abb tweaking load/save version:{version}')
    
        points = pd.read_pickle(BytesIO(loader.group["points"][:].tobytes()))
        points = gp.GeoDataFrame(points, geometry="point")
        
        lineSegments = pd.read_pickle(
            BytesIO(loader.group["lineSegments"][:].tobytes()))
        lineSegments = gp.GeoDataFrame(lineSegments, geometry="segment")

        # abb analysisparams
        _analysisParams_json = loader.group.attrs['analysisParams']  # json str
        # abj added path argument
        analysisParams = AnalysisParams(loadJson=_analysisParams_json, path = path)

        # abj last save
        try:
            lastSaveTime = loader.group.attrs['lastSaveTime']
        except:
            lastSaveTime = ""

        return cls(loader, lineSegments, points, analysisParams, lastSaveTime)

    def save(self, path: str, compression=zipfile.ZIP_STORED, version:int=0):
        if not path.endswith(".mmap"):
            path += ".mmap"

        # abj - dont save if path is empty
        if path == ".mmap":
            return
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logger.info(f'saving to {path}')
            
            fileExists = os.path.isdir(path)
            # fs = zarr.ZipStore(path, mode="w", compression=compression)
            fs = zarr.DirectoryStore(path)
            with fs as store:
                group = zarr.group(store=store)
                if not fileExists:
                    self._images.saveTo(group)
        
                # self.points : LazyGeoFrame
                _dPoints = group.create_dataset(
                        "points", overwrite = True, data=self.points.toBytes(), dtype=np.uint8)
                
                # self.segments : LazyGeoFrame
                _dSegment = group.create_dataset(
                    "lineSegments", overwrite = True, data=self.segments.toBytes(), dtype=np.uint8)

                group.attrs["version"] = 1

                # abb analysisparams
                group.attrs['analysisParams'] = self._analysisParams.getJson()

                # abj
                group.attrs["lastSaveTime"] = self.getCurrentTime()

    def getCurrentTime(self):
        currentTime = datetime.now()
        # Format the current time
        formatted_time = currentTime.strftime('%Y%m%d %H:%M')
        logger.info(f"storeLastSaveTime {formatted_time}")
        return formatted_time
    
    # Context manager
    def __enter__(self):
        self._images = self._images.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._images.__exit__(exc_type, exc_value, traceback)

    def close(self):
        self._images.close()
        return

    # Utility functions

    @timer
    def getColors(self, colorOn: str = None, function=False) -> pd.Series:
        """
        Returns the colors of the points.
        """
        if colorOn is None:
            if function:
                return lambda _: Colors.spine
            return pd.Series([Colors.spine] * len(self.points), index=self.points.index)

        categorical = False
        if colorOn not in self.points.columnsAttributes:
            raise ValueError(f"Column {colorOn} has no color attributes.")

        attr = self.points.columnsAttributes[colorOn]
        if "colors" in attr:
            colors = attr["colors"]
        elif "categorical" in attr and attr["categorical"]:
            colors = Colors.categorical
            categorical = True
        elif "divergent" in attr and attr["divergent"]:
            colors = Colors.divergent
        else:
            colors = Colors.scalar

        if colorOn in self.points.index.names:
            values = pd.Series(self.points.index.get_level_values(
                colorOn).to_list(), index=self.points.index)
        else:
            values = self.points[colorOn]

        if categorical and not isinstance(colors, dict):
            keys = list(values.unique())
            keys.sort()
            originalColors = colors
            colors = {key: originalColors[i % len(
                originalColors)] for i, key in enumerate(keys)}

        if isinstance(colors, dict):
            if function:
                return lambda x: colors[x]

            def extractColor(x):
                color = colors[x]
                if isinstance(color, list):
                    return tuple(color)
                if isinstance(color, tuple):
                    return color
                return color.values[0]

            return values.apply(extractColor)

        valuesMin = values.min()
        valuesMax = values.max()

        colors = scaleColors(colors, 1.0/255.0)
        if function:
            return lambda x: scaleColors(sample_colorscale(colors, (values[x]-valuesMin)/(valuesMax-valuesMin), colortype="tuple"), 255)

        normalized = (values-valuesMin)/(valuesMax-valuesMin)
        return pd.Series(scaleColors(sample_colorscale(colors, normalized, colortype="tuple"), 255), index=values.index)

    @timer
    def getSymbols(self, shapeOn: str = None, function=False) -> pd.Series:
        """
        Returns the symbols of the points.
        """
        if shapeOn is None:
            if function:
                return lambda _: "circle"
            return pd.Series(["circle"] * len(self.points), index=self.points.index)

        if shapeOn not in self.points.columnsAttributes:
            raise ValueError(f"Column {shapeOn} has no shape attributes.")

        attr = self.points.columnsAttributes[shapeOn]
        if "symbols" in attr:
            symbols_ = attr["symbols"]
        elif "categorical" in attr and attr["categorical"]:
            symbols_ = symbols
        else:
            raise ValueError(
                f"Column {shapeOn} is scalar and cannot be used as a shape.")

        if shapeOn in self.points.index.names:
            values = pd.Series(self.points.index.get_level_values(
                shapeOn).to_list(), index=self.points.index)
        else:
            values = self.points[shapeOn]

        if not isinstance(symbols_, dict):
            keys = list(values.unique())
            keys.sort()
            originalSymbols = symbols_
            symbols_ = {key: originalSymbols[i % len(
                originalSymbols)] for i, key in enumerate(keys)}

        if function:
            return lambda x: symbols_[x]

        return values.apply(lambda x: symbols_[x])
