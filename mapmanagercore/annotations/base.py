from datetime import datetime
from copy import copy
from io import BytesIO
import weakref
from typing import Any, Tuple, Union, Optional
import zipfile
import zarr.storage

# from mapmanagercore.lazy_geo_pandas.lazy import LazyGeoSeries
# from mapmanagercore.lazy_geo_pd_images.loader.base import Position
# abb reverting analysisparams to class (not lazy)
# from mapmanagercore.schemas.analysis_params import AnalysisParameters
import numpy as np
import pandas as pd

from mapmanagercore.benchmark import timer
from mapmanagercore.config import Colors, scaleColors, symbols
# from mapmanagercore.lazy_geo_pd_images.loader.imageio import MultiImageLoader
# abb depreciated
# from mapmanagercore.lazy_geo_pd_images.loader.zarr import ZarrLoader
from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader

from ..lazy_geo_pandas import LazyGeoFrame
from ..schemas import Segment, Spine
from ..lazy_geo_pd_images import LazyImagesGeoPandas, ImageLoader
from ..lazy_geo_pd_images.image_slices import ImageSlice
import zarr
import warnings
from plotly.express.colors import sample_colorscale
import geopandas as gp

# from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.logger import logger

# abb imageImporter converting from pickle to parquet
from ..lazy_geo_pandas.lazy import toBytes

class AnnotationsBase(LazyImagesGeoPandas):
    # _images: MultiImageLoader  # abb this is either MultiImageLoader | ZarrLoader !!!
    # 202504 this is what it is now
    # adding channel to mmMapLoader does not trigger _ch2 for points
    _image : mmMapLoader
    
    def __init__(self,
                #  loader: MultiImageLoader,  # depreciated, this is either MultiImageLoader | ZarrLoader !!!
                 loader: mmMapLoader,
                 lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(),
                 points: Union[str, pd.DataFrame] = pd.DataFrame(),
                #  analysisParams: AnalysisParams = AnalysisParams(),
                 path: str = None,
                 lastSaveTime: str = ""):

        super().__init__(loader)

        # if analysisParams is None:
        #     analysisParams = AnalysisParams()
        #     # analysisParams = AnalysisParameters()
        if lineSegments is None:
            lineSegments = pd.DataFrame()
        if points is None:
            points = pd.DataFrame()

        if not isinstance(lineSegments, gp.GeoDataFrame):
            if not isinstance(lineSegments, pd.DataFrame):
                lineSegments = pd.read_csv(lineSegments, index_col=False)

        if not isinstance(points, gp.GeoDataFrame):
            if not isinstance(points, pd.DataFrame):
                points = pd.read_csv(points, index_col=False)

        self._lastSaveTime = lastSaveTime

        # abb reverting analysisparams to class (not lazy)
        # self._analysisParameters = LazyGeoSeries(
        #     AnalysisParameters, store=weakref.ref(self), context=self
        # )
        # self._analysisParameters.update(analysisParams, skipLog=True)
        # self._analysisParams = analysisParams
        
        self._segments = LazyGeoFrame(
            Segment, data=lineSegments, store=weakref.ref(self), context=self)
        self._points = LazyGeoFrame(
            Spine, data=points, store=weakref.ref(self), context=self)

        self.loader = loader
        self.path = path

        # To invalidate columns that were miss-computed in previous version
        # we can conditionally check the version number
        # if version === 0:
        #  then we can invalidate the invalid columns by name
        # self._segments.invalidateColumns([... columns ...])
    
    def getLastSaveTime(self):
        """
        """
        # get last save time from attributes
        return self._lastSaveTime

    # abb convenience
    def getNumTimepoints(self):
        # return len(self._images.timePoints())
        return self._images.numTimepoints

    # abb convenience
    def getPointDataFrame(self, t: Optional[int] = None) -> pd.DataFrame:
        """Get the full points dataframe.
        """
        pointsDf = self.points[:]

        if t is not None:

            # move (,t) index into a column
            pointsDf = pointsDf.reset_index(level=1)
            # reduce my t==t
            pointsDf = pointsDf[pointsDf['t'] == t]

        return pointsDf

    # abj
    # def getChannelTotal(self, t : Optional[int] = None) -> int:
    #     """Get total number of channel
    #     """
    #     return self._images.channels()

    # abb convenience
    def __str__(self):
        """Print info about the map.

        See: _SingleTimePointAnnotationsBase()
        """
        timePoints = self._images.timePoints()
        numTimepoints = len(timePoints)
        numPnts = len(self.points)
        numSegments = len(self.segments)

        theRet = f'mmmap t:[{numTimepoints}], points:{numPnts} segments:{numSegments}\n'
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

    # @property
    # def analysisParams(self) -> LazyGeoSeries:
    #     return self._analysisParams

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

    def getPixels(self, time: int,
                  channel: int,
                  zRange: Tuple[int, int] = None,
                  z: int = None,
                  zSpread: int = 0,
                  threeD: bool = False) -> ImageSlice:
        """
        Loads the image data for a slice.

        Args:
          time (int): The time slot index.
          channel (int): The channel index.
          zRange (Tuple[int, int]): The visible z slice range.
          z (int): The z slice index.
          zSpread (int): The amount to offset z +/-.
          threeD (bool): Get full 3D np.array when true

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
        return super().getPixels(time, channel, zRange, threeD=threeD)

    # Serialization

    def merge(self, loader: ImageLoader):
        self.loader.merge(loader)

    @classmethod
    def load(cls, path: Union[str, None], lazy=False):
        # logger.warning(f'abb creating ZarrLoader from path:{path}')
        # logger.warning(f'  cls:{cls}')

        # abb was this
        # from mapmanagercore.lazy_geo_pd_images.loader.zarr import ZarrLoader
        # loader = ZarrLoader(path, lazy=lazy)
        # now this 20250317
        # logger.info(f'TODO: switch to mm_map_loader mmMapLoader !!!')
        loader = mmMapLoader(path)

        # abb read_pickle() is failing if we have an older version of numpy
        # when building for pyinstaller, we end up with numpy==1.26.4
        if "points" in loader.group:
            # points = pd.read_pickle(
            points = gp.read_parquet(
                BytesIO(loader.group["points"][:].tobytes()))
            points = gp.GeoDataFrame(points, geometry="point")
        else:
            points = gp.GeoDataFrame()

        if "lineSegments" in loader.group:
            # lineSegments = pd.read_pickle(
            lineSegments = gp.read_parquet(
                BytesIO(loader.group["lineSegments"][:].tobytes()))
            lineSegments = gp.GeoDataFrame(lineSegments, geometry="segment")
        else:
            lineSegments = gp.GeoDataFrame()

        # if "analysisParams" in loader.group.attrs:
        #     analysisParams = loader.group.attrs["analysisParams"]  # this is json
        #     # analysisParams = AnalysisParameters(**analysisParams)
        #     analysisParams = AnalysisParams(loadedDict=analysisParams)
        # else:
        #     # analysisParams = AnalysisParameters()
        #     analysisParams = AnalysisParams()
        
        # abb lastSaveTime will ALWAYS be in file
        # abb when using try/catch, ALWAYS name an exception, Do not use bare `except`RuffE722
        try:
            lastSaveTime = loader.group.attrs['lastSaveTime']
        except:
            lastSaveTime = ""

        # _ret = cls(loader, lineSegments, points, analysisParams, path, lastSaveTime)
        _ret = cls(loader, lineSegments, points, path, lastSaveTime)
        # logger.warning(f'  returning {type(_ret)}')
        return _ret

    # abb TODO put in one place (this function is repeated elsewhere)
    def _group_exists(self, store, group_path) -> bool:
        """Check if a zarr group exists.
        """
        try:
            with zarr.open_group(store, mode='r', path=group_path):
                return True
        except (zarr.errors.GroupNotFoundError, KeyError):
            return False

    def save(self, path: str = None):
        """Save the mmap

        Parameters:
        path : str
            Path to save to, if a folder then save as zarr DirectoryStore, otherwise save as single file zip.
        """
        if path is None:
            path = self.path

        # if not path.endswith(".mmap"):
        #     path += ".mmap"

        # # abj - dont save if path is empty
        # if path == ".mmap":
        #     logger.warning(f'did not save:{path}')
        #     return
        
        _lastSaveTime = self.getCurrentTime()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logger.info(f'saving to {path}')
            # if os.path.isdir(path):
            if path.endswith('.mmap'):
                # logger.info('   as a DirectoryStore')
                # fs = zarr.DirectoryStore(path)
                _zipStore = False
            elif path.endswith('.mmap.zip'):
                # fs = zarr.ZipStore(path, mode="w", compression=zipfile.ZIP_STORED)
                # logger.info('   as a ZipStore')
                _zipStore = True
            else:
                logger.error(f'error: expecting mmap or mmap.zip')
                return

            with zarr.ZipStore(path, mode="w", compression=zipfile.ZIP_STORED) if _zipStore \
                 else zarr.DirectoryStore(path) as store:
                
                group = zarr.group(store=store)  # creates a group

                # abb we need to interogate the existing mmap and
                # see if we have some new image channels or timepoints to save
                # # abb error on saving changes to existing mmap
                # zarr.errors.ContainsGroupError: path 'images' contains a group
                # if not self._group_exists(store, "images"):
                
                # v1, replaced by ZarrLoader2
                # if "images" not in group.keys():
                #     images = group.create_group("images")
                # else:
                #     images = zarr.group(store=store)['images']
                # self._images.saveTo(images)
                
                #v2
                self._images.saveAs(path)

                # abb added overwrite=True, need to implement dirty flag for points and segments
                # zarr.errors.ContainsArrayError: path 'points' contains an array
                # logger.warning('TODO: abb imageImporter saving points/segments as text')
                # v1
                # group.create_dataset(
                #     "points", data=self.points.toBytes(), dtype=np.uint8, overwrite=True)
                # group.create_dataset(
                #     "lineSegments", data=self.segments.toBytes(), dtype=np.uint8, overwrite=True)
                # group.attrs["version"] = 1
                
                # GeoDataFrame
                # logger.error(f'self.points[:] is:{type(self.points[:])}')
                # GeoDataFrame
                # logger.error(f"self.points[:].set_geometry('point') is: {type(self.points[:].set_geometry('point'))}")
                
                # v1.1
                _points = self.points[:].set_geometry('point')  # _points is GeoPandas, NOT our lazy
                # logger.error(f'_points is:{type(_points)}')
                group.create_dataset(
                    "points",
                    data=toBytes(_points),
                    # data=self.points[:].to_json(orient='index'),
                    dtype=np.uint8,
                    overwrite=True)
                
                _segments = self.segments[:].set_geometry('segment')  # _segments is GeoPandas, NOT our lazy
                group.create_dataset(
                    "lineSegments",
                    data=toBytes(_segments),
                    # data=self.segments[:].to_json(orient='index'),                    
                    dtype=np.uint8,
                    overwrite=True)
                group.attrs["version"] = 1.1

                # # v1.2
                # _rootDf = self.points[:]  #self.points._rootDf  # or use self.points[:]
                # # _rootDf = _rootDf.set_geometry('point')
                # logger.warning(f'_rootDf is: {type(_rootDf)}')
                # print(_rootDf.columns)
                # print(_rootDf)
                # # logger.warning(f'   _rootDf.geometry.name:{_rootDf.geometry.name}')
                # _pointJson = _rootDf.to_json()
                # logger.warning('_pointJson is:')
                # print(_pointJson)
                # group.create_dataset(
                #     "points", data=self.points.toBytes(), dtype=np.uint8, overwrite=True)
                # group.create_dataset(
                #     "lineSegments", data=self.segments.toBytes(), dtype=np.uint8, overwrite=True)
                # group.attrs["version"] = 1.2

                # abb analysisparams
                # group.attrs['analysisParams'] = self._analysisParams.getDict()

                # abj
                group.attrs["lastSaveTime"] = _lastSaveTime

        self._lastSaveTime = _lastSaveTime

    def getCurrentTime(self):
        currentTime = datetime.now()
        # Format the current time
        formatted_time = currentTime.strftime('%Y%m%d %H:%M')
        # logger.info(f"storeLastSaveTime {formatted_time}")
        return formatted_time

    # Context manager
    def __enter__(self):
        logger.warning('context manager')
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

    # abj
    # def loadInNewChannel(self, path: Union[str, np.ndarray], time: int = 0, channel: int = 0):
    #     """ Load in new channel (tif image)
    #     This function is called by fullMap within pymapmanager Desktop

    #     Args:
    #         path: directory str of tif file
    #         time: time in series
    #         channel: new channel value
    #     """
    #     # functions = [func for func in dir(self._images) if callable(getattr(self._images, func))]
    #     # print(functions)

    #     self._images.readNewImages(path = path, channel = channel)
        # self._images.appendChannelToTimePoint()

    def getDendrogramReplot(self, newSegmentID: int, spineAngleChecked: bool, spineLengthChecked: bool, spineLengthConstant: int):
        """ calculate necessary values to replot dendrogram widget

        Args:
            newSegmentID: segment ID being plotted
            spineAngleChecked: True when showing spine angle, False when not (will show perpendicular line instead)
            spineLengthChecked: True when showing spine length, otherwise use spineLengthConstant
            spineLengthConstant: value used to plot spine line spineLengthCheck if False

        Return:
            plotDF: df
            spineLineDF: df
            segmentLength:

        """
        newSegmentID = int(newSegmentID)
        self._paDF = self.getPointDataFrame()

        filteredPointDF = self._paDF[self._paDF["segmentID"] == newSegmentID]
        spinePositions, spineLength, spineAngle, spineSide, spineIndex = \
            [filteredPointDF[col] for col in ["spinePosition", "spineLength", "spineAngle", "spineSide"]] + [filteredPointDF.index]

        anchorX, anchorY = [0] * len(spinePositions), spinePositions.tolist() 
        spineX, spineY, savedSpineIndex = [None] * len(spineIndex), [], []
        xVal = np.where(spineLengthChecked, spineLength, spineLengthConstant)

        # Handle direction adjustments for xVal
        direction_map = {"Left": -1, "Right": 1, "Undefined": np.nan}
        spineX = np.array([xVal[i] * direction_map.get(direction, 1) for i, direction in enumerate(spineSide)])
        savedSpineIndex = np.array(spineIndex)

        # Handle Y calculation if the spine angle checkbox is checked
        if spineAngleChecked:
            # Define undefined angles for tangent calculation
            undefinedList = [270, 90, 180, 0, 360]

            # Vectorized angle-based Y calculation
            angles = spineAngle
            anchorYVal = np.array(anchorY)

            # Handle undefined angles (where tan function would break)
            is_undefined_angle = np.isin(np.floor(angles), undefinedList)
            angledY = np.where(is_undefined_angle, anchorYVal, xVal * np.tan(np.radians(angles)))

            # Adjust Y based on the angle ranges
            conditions = [
                (0 <= angles) & (angles <= 90), 
                (90 < angles) & (angles <= 180), 
                (180 < angles) & (angles <= 270), 
                (270 < angles) & (angles <= 360)
            ]

            adjustments = [
                lambda y, diff: y + abs(diff),
                lambda y, diff: y - abs(diff),
                lambda y, diff: y - abs(diff),
                lambda y, diff: y + abs(diff)
            ]

            # Apply the appropriate adjustment for each angle range
            for cond, adjust in zip(conditions, adjustments):
                diff = np.abs(anchorYVal) - np.abs(angledY)
                angledY = np.where(cond, adjust(anchorYVal, diff), angledY)

            spineY = angledY
        else:
            spineY = np.array(anchorY) 

        spineLineX = []
        spineLineY = []
        for i in range(len(anchorX)):
            spineLineX.extend([anchorX[i], spineX[i], np.nan])
            spineLineY.extend([anchorY[i], spineY[i], np.nan])

        laDF = self.segments[:]
        filteredLineDF = laDF.loc[laDF.index.get_level_values(0) == newSegmentID] # only compare segments id, not time

        segmentLength = filteredLineDF["length"].iloc[0]
        plotDF = pd.DataFrame({"spineX": spineX, "spineY": spineY, "spineIndex": savedSpineIndex})
        spineLineDF = pd.DataFrame({"spineLineX": spineLineX, "spineLineY": spineLineY})
        return plotDF, spineLineDF, segmentLength