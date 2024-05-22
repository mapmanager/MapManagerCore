from copy import copy
from typing import Any, Tuple
import zipfile
import numpy as np
from ..lazy_geo_pandas import LazyGeoFrame
from ..schemas import Segment, Spine
from ..lazy_geo_pd_image import LazyImagesGeoPandas
from ..image_slices import ImageSlice
from ..loader.base import ImageLoader, Loader
import zarr
import warnings

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.logger import logger

class AnnotationsBase(LazyImagesGeoPandas):
    _images: ImageLoader

    def __init__(self, loader: Loader):
        super().__init__(loader.images())

        self._segments = LazyGeoFrame(
            Segment, data=loader.segments(), store=self)
        self._points = LazyGeoFrame(Spine, data=loader.points(), store=self)

        # abb analysisparams
        self._analysisParams : AnalysisParams = loader.analysisParams()

        self._zarrPath : str = loader.getZarrPath()

    def getZarrPath(self):
        return self._zarrPath
    
    def __str__(self):
        """Print info about the map.
        
        See: _SingleTimePointAnnotationsBase()
        """
        zarrPath = self.getZarrPath()
        numTimepoints = len(self._images._imagesSrcs.keys())
        numPnts = len(self.points._rootDf)
        numSegments = len(self.segments._rootDf)
        return f't:{numTimepoints}, points:{numPnts} segments:{numSegments} zarr:{zarrPath}'
        
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
        c = copy(self)
        c._points = c._points[filter]
        return c
    
    def filterSegments(self, filter: Any):
        c = copy(self)
        c._segments = c._segments[filter]
        return c

    def getTimePoint(self, time: int):
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

    def save(self, path: str, compression=zipfile.ZIP_STORED):
        if not path.endswith(".mmap"):
            raise ValueError(
                "Invalid file format. Please provide a path ending with '.mmap'.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logger.info(f'saving to {path}')
            fs = zarr.ZipStore(path, mode="w", compression=compression)
            with fs as store:
                group = zarr.group(store=store)
                self._images.saveTo(group)
                group.create_dataset("points", data=self.points.toBytes(), dtype=np.uint8)
                group.create_dataset("lineSegments", data=self.segments.toBytes(), dtype=np.uint8)
                group.attrs["version"] = 1

                # abb analysisparams
                group.attrs['analysisParams'] = self._analysisParams.getJson()

    def __enter__(self):
        self._images = self._images.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._images.__exit__(exc_type, exc_value, traceback)
        
    def close(self):
        self._images.close()
        return

