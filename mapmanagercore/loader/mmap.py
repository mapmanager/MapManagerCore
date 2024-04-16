from io import BytesIO
import pandas as pd
from .base import ImageLoader, Loader
from typing import Tuple
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
        metadata = group.attrs["metadata"]
        
        super().__init__(lineSegments, points, metadata)
        self._images = group["images"]

    def images(self) -> ImageLoader:
        return self

    def shape(self) -> Tuple[int, int, int, int, int]:
        return self._images.shape

    def dtype(self) -> np.dtype:
        return self._images.dtype

    def saveTo(self, group: zarr.Group):
        group.create_dataset("images", data=self._images,
                             dtype=self._images.dtype)

    def loadSlice(self, time: int, channel: int, slice: int) -> np.ndarray:
        return self._images[time][channel][slice]

    def fetchSlices(self, time: int, channel: int, sliceRange: Tuple[int, int]) -> np.ndarray:
        return np.max(self._images[time][channel][sliceRange[0]:sliceRange[1]], axis=0)


class MMapLoader(MMapLoaderLazy):
    def __init__(self, path: str):
        super().__init__(path)
        self._images = self._images[:]