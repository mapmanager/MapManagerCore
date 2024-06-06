from copy import copy
from typing import Any, Tuple
import zipfile
import numpy as np
import pandas as pd

from mapmanagercore.benchmark import timer
from mapmanagercore.config import COLORS, scaleColors, symbols
from ..lazy_geo_pandas import LazyGeoFrame
from ..schemas import Segment, Spine
from ..lazy_geo_pd_image import LazyImagesGeoPandas
from ..image_slices import ImageSlice
from ..loader.base import ImageLoader, Loader
import zarr
import warnings
from plotly.express.colors import sample_colorscale

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.logger import logger

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
        try:
            return self._zarrPath
        except (AttributeError) as e:
            logger.warning(f'{e}')

    # abb
    def __str__(self):
        """Print info about the map.
        
        See: _SingleTimePointAnnotationsBase()
        """
        zarrPath = self.getZarrPath()
        numTimepoints = len(self._images._imagesSrcs.keys())
        numPnts = len(self.points._rootDf)
        numSegments = len(self.segments._rootDf)
        
        # images
        # numTimepoints = self.numTimepoints()

        return f't:{numTimepoints}, points:{numPnts} segments:{numSegments} zarr:{zarrPath}'
    
    #abb
    # def numTimepoints(self) -> int:
    #     """Get the number of timepoints.
    #     """
    #     return self._images.shape(t=0)[0]
    
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
                group.create_dataset(
                    "points", data=self.points.toBytes(), dtype=np.uint8)
                group.create_dataset(
                    "lineSegments", data=self.segments.toBytes(), dtype=np.uint8)
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

    @timer
    def getColors(self, colorOn: str = None, function=False) -> pd.Series:
        if colorOn is None:
            if function:
                return lambda _: COLORS["spine"]
            return pd.Series([COLORS["spine"]] * len(self.points), index=self.points.index)

        categorical = False
        if colorOn not in self.points.columnsAttributes:
            raise ValueError(f"Column {colorOn} has no color attributes.")

        attr = self.points.columnsAttributes[colorOn]
        if "colors" in attr:
            colors = attr["colors"]
        elif "categorical" in attr and attr["categorical"]:
            colors = COLORS["categorical"]
            categorical = True
        elif "divergent" in attr and attr["divergent"]:
            colors = COLORS["divergent"]
        else:
            colors = COLORS["scalar"]

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
