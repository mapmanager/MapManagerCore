import time
from io import BytesIO
import pandas as pd
from mapmanagercore.config import Metadata
from .base import ImageLoader, Loader
from typing import Iterator, Tuple
import numpy as np
import zarr
import geopandas as gp

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.logger import logger

class MMapLoaderLazy(Loader, ImageLoader):
    def __init__(self, path: str):
        self.store = zarr.ZipStore(path, mode="r")
        group = zarr.group(store=self.store)
        points = pd.read_pickle(BytesIO(group["points"][:].tobytes()))
        points = gp.GeoDataFrame(points, geometry="point")
        lineSegments = pd.read_pickle(
            BytesIO(group["lineSegments"][:].tobytes()))
        lineSegments = gp.GeoDataFrame(lineSegments, geometry="segment")

        # abb
        _analysisParams_json = group.attrs['analysisParams']  # json str
        analysisParams = AnalysisParams(loadJson=_analysisParams_json)

        super().__init__(lineSegments, points, analysisParams)

        self._imagesSrcs = {}
        self._metadata = {}
        for t in group.attrs["timePoints"]:
            self._imagesSrcs[t] = group[f"img-{t}"]
            self._metadata[t] = group.attrs[f"metadata-{t}"]

    def images(self) -> ImageLoader:
        return self

    def timePoints(self) -> Iterator[int]:
        return self._imagesSrcs.keys()

    def _images(self, t: int) -> np.ndarray:
        return self._imagesSrcs[t]

    def metadata(self, t: int) -> Metadata:
        return self._metadata[t] if t in self._metadata else Metadata()

    def close(self):
        self.store.close()

    # abb, why is this duplicated in multiimageloader?
    def fetchSlices2(self, time: int, channel: int, sliceRange: Tuple[int, int]) -> np.ndarray:
        # _imgData = self._images[time][channel][sliceRange[0]:sliceRange[1]]
        if isinstance(sliceRange, tuple):
            return self._images[time][channel][sliceRange[0]:sliceRange[1]]
        else:
            return self._images[time][channel][sliceRange]

class MMapLoader(MMapLoaderLazy):
    def __init__(self, path: str):
        super().__init__(path)
        self._imagesSrcs = {key: value[:]
                            for key, value in self._imagesSrcs.items()}
