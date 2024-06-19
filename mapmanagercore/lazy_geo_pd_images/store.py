# Adds image slices to lazy geo pandas

from typing import Callable, List, Self, Tuple, Union, Unpack
import numpy as np
from mapmanagercore.lazy_geo_pd_images.image_slices import ImageSlice
from mapmanagercore.lazy_geo_pandas.attributes import ColumnAttributes
from mapmanagercore.lazy_geo_pandas.lazy import LazyGeoFrame
from mapmanagercore.lazy_geo_pd_images.loader import ImageLoader
from ..lazy_geo_pandas import LazyGeoPandas
import geopandas as gp
import pandas as pd

from mapmanagercore.logger import logger

class ImageColumnAttributes(ColumnAttributes):
    """Attributes for image computed columns."""

    """The list of aggregates function names to compute."""
    _aggregate: list[str]

    """The z spread to use when computing the pixels."""
    zSpread: int

    """The time column to use when computing the pixels."""
    t: str


def parseColumns(columns: List[str], prefix: str) -> Tuple[set[int], set[str]]:
    """Parse the roi computed columns to get the channels and aggregates."""
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
    """Apply an aggregate function to the data."""
    try:
        return getattr(np, agg)(x)
    except (ValueError) as e:
        logger.error(f'ValueError: {e}')
        logger.error(f'  x:{x} agg:{agg}')
        return np.nan


class LazyImagesGeoPandas(LazyGeoPandas):
    """A Lazy geo pandas store with image data"""
    _images: ImageLoader

    def __init__(self, images: ImageLoader, overrideDefault=True):
        super().__init__()
        self._images = images

        if overrideDefault:
            LazyGeoPandas.setDefaultStore(self)

    def _genWrappedFunc(self, method, attributes, frame: LazyGeoFrame[Self]):
        """Generate a wrapped function for the computed column."""

        name = attributes["key"]
        func = method

        zSpread = attributes["zSpread"] if "zSpread" in attributes else 0
        tColumn = attributes["t"] if "t" in attributes else "t"
        timeIndexLevel = frame._schema._index.index(
            tColumn) if tColumn in frame._schema._index else None

        def wrappedFunc(frame: LazyGeoFrame[Self]):
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

            return pd.DataFrame({
                f"{name}_ch{channel + 1}_{agg}": pixels[channel].apply(lambda x: getattr(np, agg)(x)) for agg in aggregates for channel in channels
            }, index=pixels.index)
        return wrappedFunc

    def addSchema(self, frame: LazyGeoFrame[Self]):
        """Add a schema frame to the store. Essentially, this adds a new data frame to the store."""

        # Inject computed columns that use the image to calculate roi stats
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

            frame.updateComputedDependencies()

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
        """ Get the pixels that are in the shapes.

        Args:
            shapes (gp.GeoDataFrame): The shapes to get the pixels for.
                Shapes can contain a 't' column to specify the time and/or a 'z' column to specify the z.
                Alternatively, the time and z can be specified as arguments.
            channel (Union[int, List[int]], optional): The channel to get the pixels for. Defaults to 0.
            zSpread (int, optional): The z spread to get the pixels for. Defaults to 0.
            time ([type], optional): The time to get the pixels for. Defaults to None.
            z (int, optional): The z to get the pixels for. Defaults to None.
        """
        return self._images.getShapePixels(shapes, channel=channel, zSpread=zSpread, time=time, z=z)


def aggregateROI(dependencies: Union[List[str], dict[str, list[str]]] = {}, aggregate: list[str] = [], **attributes: Unpack[ImageColumnAttributes]):
    """A decorator that adds image based computed column to the schema.

    Args:
        dependencies (Union[List[str], dict[str, list[str]], optional):
            The dependencies for the computed column. Defaults to {}.
            The dictionary can be used to specify dependencies across different schemas.
            {"schemaName": ["column1", "column2"], "schemaName2": ["column3", "column4"]}
        aggregate (list[str], optional): The aggregates to compute. Defaults to [].
        **attributes (Unpack[ImageColumnAttributes]): The attributes for the computed column.

    Returns:
        A function that returns a geo pandas data frame with a shape column and a z column.
    """
    def wrapper(func: Callable[[], Union[pd.Series, pd.DataFrame]]):
        func._imageComputed = {
            "key": func.__name__,
            "_aggregate": aggregate,
            **attributes,
            "_dependencies": dependencies,
        }
        return func
    return wrapper
