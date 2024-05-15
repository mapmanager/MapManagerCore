# adds image slices to lazy geo pandas

from typing import Callable, List, Tuple, Union, Unpack
import numpy as np
from mapmanagercore.image_slices import ImageSlice
from mapmanagercore.lazy_geo_pandas.attributes import ColumnAttributes
from mapmanagercore.lazy_geo_pandas.lazy import LazyGeoFrame
from .loader.base import ImageLoader
from .lazy_geo_pandas import LazyGeoPandas
import geopandas as gp
import pandas as pd

from mapmanagercore.logger import logger


class ImageColumnAttributes(ColumnAttributes):
    _aggregate: list[str]
    zSpread: int
    t: str


def parseColumns(columns: List[str], prefix: str) -> Tuple[set[int], set[str]]:
    channels = set()
    aggregates = set()
    for column in columns:
        if not column.startswith(prefix):
            continue

        parts = column.split("_")
        if len(parts) < 3:
            continue

        channels.add(int(parts[1][2:]) - 1)
        aggregates.add(parts[2])

    return channels, aggregates


def applyAgg(x, agg):
    try:
        return getattr(np, agg)(x)
    except (ValueError) as e:
        logger.error(f'ValueError: {e}')
        logger.error(f'  x:{x} agg:{agg}')
        return np.nan


class LazyImagesGeoPandas(LazyGeoPandas):
    # A Lazy geo pandas store with image data
    _images: ImageLoader

    def __init__(self, images: ImageLoader, overrideDefault=True) -> None:
        super().__init__()
        self._images = images

        if overrideDefault:
            LazyGeoPandas.setDefaultStore(self)

    def _genWrappedFunc(self, method, attributes, frame: LazyGeoFrame):
        name = attributes["key"]
        func = method

        zSpread = attributes["zSpread"] if "zSpread" in attributes else 0
        tColumn = attributes["t"] if "t" in attributes else "t"
        timeIndexLevel = frame._schema._index.index(
            tColumn) if tColumn in frame._schema._index else None

        def wrappedFunc(frame: LazyGeoFrame):
            (channels, aggregates) = parseColumns(
                frame.pendingColumns(), name)
            if len(channels) == 0 or len(aggregates) == 0:
                return gp.GeoDataFrame()

            shapes: gp.GeoDataFrame = func(frame)
            shapeKey = shapes.columns.symmetric_difference(["t", "z"])[0]
            shapes.rename(columns={shapeKey: "shape"}, inplace=True)

            shapes["t"] = frame["t"] if timeIndexLevel is None else frame._df.index.get_level_values(
                timeIndexLevel)
            channels = list(channels) if len(
                channels) > 1 else next(channels)

            # Compute the aggregates over the pixels
            pixels = self.getShapePixels(
                shapes, channel=channels, zSpread=zSpread)

            if isinstance(pixels, pd.Series):
                # one channel was returned
                return pixels.apply(lambda x: pd.Series(
                    {f"{name}_ch{pixels.name + 1}_{agg}": applyAgg(x, agg) for agg in aggregates}), index=pixels.index)

            # multiple channels were returned
            # abb, ValueError: zero-size array to reduction operation maximum which has no identity
            try:
                return pd.DataFrame({
                    f"{name}_ch{channel + 1}_{agg}": pixels.loc[:, channel].apply(lambda x: applyAgg(x, agg)) for agg in aggregates for channel in channels
                }, index=pixels.index)
            except (ValueError) as e:
                logger.error(f'ValueError: {e}')
                logger.error(
                    f'  name:{name} channel:{channels} agg:{aggregates}')
                # print('pixels.index:')
                # print(pixels.index)
                # print('pixels')
                # print(pixels)
                return None

        return wrappedFunc

    def addSchema(self, frame: LazyGeoFrame) -> None:
        # inject image computed columns
        cls = frame._schema.__bases__[1]
        for method in cls.__dict__.values():
            if not hasattr(method, "_imageComputed"):
                continue

            attributes: ImageColumnAttributes = method._imageComputed
            if "_aggregate" not in attributes:
                continue
            name = attributes["key"]
            wrappedFunc = self._genWrappedFunc(method, attributes, frame)
            for channel in range(self._channels()):
                for agg in attributes["_aggregate"]:
                    frame.addComputed(
                        f"{name}_ch{channel + 1}_{agg}",
                        {
                            **attributes,
                            "title": f"{name} Channel {channel + 1} - {agg.capitalize()}",
                        },
                        wrappedFunc,
                        skipUpdate=True
                    )

            frame.updateComputed()

        return super().addSchema(frame)

    def _channels(self):
        return self._images.channels()

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
                raise ValueError("zRange or z must be provided")

        return ImageSlice(self._images.fetchSlices(time, channel, (zRange[0], zRange[1] + 1)))

    def getShapePixels(self, shapes: gp.GeoDataFrame, channel: Union[int, List[int]] = 0, zSpread: int = 0, time=None, z: int = None) -> Union[pd.Series, pd.DataFrame]:
        return self._images.getShapePixels(shapes, channel=channel, zSpread=zSpread, time=time, z=z)


def calculatedROI(dependencies: Union[List[str], dict[str, list[str]]] = {}, aggregate: list[str] = [], **attributes: Unpack[ImageColumnAttributes]):
    def wrapper(func: Callable[[], Union[pd.Series, pd.DataFrame]]):
        func._imageComputed = {
            "key": func.__name__,
            "_aggregate": aggregate,
            **attributes,
            "_dependencies": dependencies,
        }
        return func
    return wrapper
