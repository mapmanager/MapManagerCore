from io import BytesIO
import json
import pandas as pd

from mapmanagercore.config import Metadata
from .base import ImageLoader, Loader
from typing import Iterator, Tuple
import numpy as np
import zarr
import geopandas as gp


class MMapLoaderLazy(Loader, ImageLoader):
    def __init__(self, path: str):
        store = zarr.ZipStore(path, mode="r")
        group = zarr.group(store=store)
        points = pd.read_pickle(BytesIO(group["points"][:].tobytes()))
        points = gp.GeoDataFrame(points, geometry="point")
        points["anchor"] = gp.GeoSeries(points["anchor"])
        points["point"] = gp.GeoSeries(points["point"])

        lineSegments = pd.read_pickle(
            BytesIO(group["lineSegments"][:].tobytes()))
        lineSegments = gp.GeoDataFrame(lineSegments, geometry="segment")
        lineSegments["segment"] = gp.GeoSeries(lineSegments["segment"])

        super().__init__(lineSegments, points)

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


class MMapLoader(MMapLoaderLazy):
    def __init__(self, path: str):
        super().__init__(path)
        self._imagesSrcs = {key: value[:] for key, value in self._imagesSrcs.items()}
