from typing import Tuple
import zipfile
import geopandas as gp
import numpy as np
import pandas as pd

from ..config import Metadata
from ..image_slices import ImageSlice
from ..loader.base import ImageLoader, Loader
import zarr
import warnings
import io


class AnnotationsBase:
    images: ImageLoader  # used for the brightest path
    _points: gp.GeoDataFrame
    _lineSegments: gp.GeoDataFrame
    _metadata: Metadata

    def __init__(self, loader: Loader):
        self._lineSegments = loader.segments()
        self._points = loader.points()
        self.images = loader.images()
        self._metadata = loader.metadata()
        
    def metadata(self) -> Metadata:
        return self._metadata

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
                zRange = (int(self._points["z"].min()),
                          int(self._points["z"].max()))

        return ImageSlice(self.images.fetchSlices(time, channel, (zRange[0], zRange[1] + 1)))

    def getShapePixels(self, shapes: gp.GeoSeries, channel: int = 0, zSpread: int = 0, ids: pd.Index = None, id: str = None, time=None) -> pd.Series:
        if id:
            ids = [id]

        if isinstance(shapes, list):
            shapes = gp.GeoSeries(shapes, index=ids)
            z = shapes.apply(lambda x: x.coords[0][2])

        singleRow = not isinstance(shapes, gp.GeoSeries)
        if singleRow:
            z = self._points.loc[ids] if ids else [shapes.coords[0][2]]
            shapes = gp.GeoSeries(shapes, index=ids)
        else:
            if shapes.iloc[0].has_z:
                z = shapes.apply(lambda x: x.coords[0][2])
            else:
                z = self._points.loc[ids if ids else shapes.index, "z"]
        shapes = shapes.to_frame(name="shape")
        shapes["z"] = z
        if time is not None:
            shapes["t"] = time

        r = self.images.getShapePixels(
            shapes, channel=channel, zSpread=zSpread)
        if singleRow:
            return r.iloc[0]
        return r

    def save(self, path: str, compression=zipfile.ZIP_STORED):
        if not path.endswith(".mmap"):
            raise ValueError(
                "Invalid file format. Please provide a path ending with '.mmap'.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            store = zarr.ZipStore(path, mode="w", compression=compression)
            group = zarr.group(store=store)
            self.images.saveTo(group)

            group.create_dataset("points", data=toBytes(self._points),
                                 dtype=np.uint8)
            group.create_dataset("lineSegments", data=toBytes(self._lineSegments),
                                 dtype=np.uint8)
            group.attrs["metadata"] = self.metadata()
            store.close()


def toBytes(df: gp.GeoDataFrame):
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)
