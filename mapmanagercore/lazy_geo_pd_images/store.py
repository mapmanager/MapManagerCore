# Adds image slices to lazy geo pandas

from typing import Callable, List, Self, Tuple, Union, Unpack
import weakref
import numpy as np
from mapmanagercore.lazy_geo_pd_images.image_slices import ImageSlice
from mapmanagercore.lazy_geo_pandas.attributes import ColumnAttributes
from mapmanagercore.lazy_geo_pandas.lazy import LazyGeoFrame
from mapmanagercore.lazy_geo_pd_images.loader import ImageLoader
from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader
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
    logger.error(f'columns:{columns} prefix:{prefix}')
    
    channels = set()
    aggregates = set()
    for column in columns:
        if not column.startswith(prefix):
            continue

        parts = column.split("_")
        if len(parts) < 3:
            continue

        # logger.warning('  === abb removing -1 in channel')
        # logger.warning(f'    column:{column} prefix:{prefix} parts:{parts}')
        # channels.add(int(parts[1][2:]) - 1)
        # abb, removed -1
        channels.add(int(parts[1][2:]))

        aggregates.add(parts[2])

    return channels, aggregates

# abj
def parseColumns2(columns: List[str], prefix: str) -> Tuple[set[int], set[str]]:
    """Parse the roi computed columns to get the channels and aggregates.
    Optimized to only get requested channels columns
    """
    # logger.error(f'columns:{columns} prefix:{prefix}')
    
    channels = set()
    aggregates = set()
    for column in columns:
        if not column.startswith(prefix):
            continue

        parts = column.split("_")
        if len(parts) < 3:
            continue

        # channels.add(int(parts[1][2:]))
        aggregates.add(parts[2])

    return aggregates

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
    # _images: ImageLoader
    _image = mmMapLoader

    # abb TODO: switch to mmMapLoader
    def __init__(self, images: mmMapLoader, overrideDefault=True):
        super().__init__()
        # logger.info(f'abb creating LazyImagesGeoPandas() with images:{type(images)}')
        self._images = images

        self._recordShapePixels = {} # dict

        if overrideDefault:
            LazyGeoPandas.setDefaultStore(self)

    # TODO: optimize by lazily aggregating the pixels for the requested columns (aggregate, channel)
    # TODO: Insure that ^ optimization also doesn't recompute the same pixels multiple times
    def _genWrappedFunc(self, method, attributes, frame: LazyGeoFrame[Self]):
        """Generate a wrapped function for the computed column."""

        name = attributes["key"]
        func = method

        zSpread = attributes["zSpread"] if "zSpread" in attributes else 0
        tColumn = attributes["t"] if "t" in attributes else "t"

        # logger.info(f"check zSpread {zSpread} tColumn {tColumn}")
        timeIndexLevel = frame._schema._index.index(
            tColumn) if tColumn in frame._schema._index else None

        weakSelf = weakref.ref(self)

        def wrappedFunc(frame: LazyGeoFrame[Self]):
            # abb pending columns only contains _ch1 (after append channel, should contain _ch2)
            # logger.error(f'    frame.pendingColumns():{frame.pendingColumns()}')
            # logger.error(f'    name:{name}')

            # (channels, aggregates) = parseColumns(
            #             frame.pendingColumns(), name)        
            # logger.error(f'    before channels are:{channels}')
            # logger.error(f'    timeIndexLevel:{timeIndexLevel}')

            channels = weakSelf().getActivatedChannels(timeIndexLevel)
            # logger.info(f"activated channels are {channels}")
            aggregates = parseColumns2(
                frame.pendingColumns(), name)
            if len(channels) == 0 or len(aggregates) == 0:
                return gp.GeoDataFrame()

            shapes: gp.GeoDataFrame = func(frame)
            shapeKey = shapes.columns.symmetric_difference(["t", "z"])[0]
            shapes.rename(columns={shapeKey: "shape"}, inplace=True)

            # shapes["t"] = frame["t"] if timeIndexLevel is None else frame._df.index.get_level_values(
            #     timeIndexLevel)
            
            # logger.error('too complicated')
            # logger.warning(f'  original channels is: {channels}')
            
            # abb >= 1
            # FIXME: S: Channels should only be a list if there are multiple channels in which case we return a DataFrame instead of a single Series
            # if the use case for a single channel isn't needed, we can remove the check and always return a DataFrame
            # which will simplify the code base
            # abb, 1 channel case should just be list of len 1
            # was this
            # channels = list(channels) if len(channels) >= 1 else next(channels)
            channels = list(channels)

            # logger.warning('  abb agreed ... why is this so complicated???')
            # logger.warning(f'    channels is:{channels}')
            
            # abb, what timepoint are we in???
            # self._images.metadata

            # logger.error(f'post channels:{channels}')
            # logger.error(f'next(channels):{next(channels)}')
            
            # Do a simple save
            # Check if shapes channels and zspread is the same
            if not len(self._recordShapePixels) == 0:
                oldChannels = self._recordShapePixels["channels"]
                oldShapes = self._recordShapePixels["shapes"]
                oldZSpread = self._recordShapePixels["zSpread"]
            else:
                oldChannels = oldShapes = oldZSpread = None

            # Compute the aggregates over the pixels
            # logger.info(f"oldChannels {oldChannels}, channels {channels}")
            # logger.info(f"oldZSpread {oldZSpread}, zSpread {zSpread}")

            self._recordShapePixels["channels"] = channels
            self._recordShapePixels["shapes"] = shapes
            self._recordShapePixels["zSpread"] = zSpread
            # issue is that shapes will have time as a second index

            if oldShapes is not None:
                # get actual shape values
                oS = oldShapes["shape"].values
                s = shapes["shape"].values
                # logger.info(f"oldShapes {oS}, shapes {s}")

            if oldChannels == channels and np.array_equal(oS, s) and oldZSpread == zSpread:
                # logger.info(F"old pixels")
                try:
                    pixels =  self._recordShapePixels["pixels"]
                    # logger.info(f"using old pixels")
                except:
                    # logger.info(f"no old pixels, must calculate new one")
                    pixels = weakSelf().getShapePixels(
                    shapes, channel=channels, zSpread=zSpread)
            
            else:
                # logger.info(F"new pixels")
                pixels = weakSelf().getShapePixels(
                    shapes, channel=channels, zSpread=zSpread)
                
            self._recordShapePixels["pixels"] = pixels

            if isinstance(pixels, pd.Series):
                # # one channel was returned
                return pixels.apply(lambda x: pd.Series(
                    # {f"{name}_ch{pixels.name + 1}_{agg}": applyAgg(x, agg) for agg in aggregates}), index=pixels.index)
                    {f"{name}_ch{pixels.name}_{agg}": applyAgg(x, agg) for agg in aggregates}), index=pixels.index)

            # logger.error(f'REMOVE {channels} -> channels = [1]')
            _channels = channels
            logger.info(f'  _channels  {_channels}:')
            return pd.DataFrame({
                # f"{name}_ch{channel + 1}_{agg}": pixels[channel].apply(lambda x: getattr(np, agg)(x)) for agg in aggregates for channel in channels
                f"{name}_ch{channel}_{agg}": pixels[channel].apply(lambda x: getattr(np, agg)(x)) for agg in aggregates for channel in _channels
            }, index=pixels.index)

        return wrappedFunc

    def addSchema(self, frame: LazyGeoFrame[Self], newChannelKeys: List = None):
        """Add a new channel schema frame to the store.
        Essentially, this adds a new data frame to the store.
        This function is called after append channel
        
        abb this creates all spine intensity analysis columns

        Args:
            frame: spine DF
            newChannelsKeys: List of new channels that are appended
        """

        # TODO: check if channel schema was already added
        if frame is None:
            return
        
        # logger.warning('!!==!! abb in store.py LazyImagesGeoPandas')

        # abai 20250806: Use metadata3 API for current channel keys
        currentChannelKeys = self._images.metadata.getTimepoint(1).channelKeys  # abai 20250806

        # Inject computed columns that use the image to calculate roi stats
        # logger.info(f"frame check {frame}")
        for method in frame._schema.__dict__.values():
            if not hasattr(method, "_imageComputed"):
                continue

            attributes: ImageColumnAttributes = method._imageComputed
            logger.info(f'checking attributes {attributes}')
            if "_aggregate" not in attributes:
                continue

            name = attributes["key"]
            wrappedFunc = self._genWrappedFunc(method, attributes, frame)

            for channel in currentChannelKeys:  # abai 20250806
                for agg in attributes["_aggregate"]:
                    col_name = f"{name}_ch{channel}_{agg}"  # abai 20250806
                    # abai 20250806: Check for existing column before adding
                    if col_name in frame.columns:  # abai 20250806
                        continue  # abai 20250806: Skip if already present
                    frame.addComputed(
                        col_name,
                        {
                            **attributes,
                            "title": f"{name} Channel {channel} - {agg.capitalize()}",
                        },
                        wrappedFunc,
                        skipUpdate=True
                    )

        frame.updateComputedDependencies()

        return super().addSchema(frame)

    def _maxChannels(self):
        return self._images.maxChannels()
    
    def getActivatedChannels(self, t):
        return self._images.activatedChannels(t)

    def getInActiveChannels(self, t):
        return self._images.inActiveChannels(t)

    # abb this is overly complex, why is this using next() and iter() ???
    def imageBounds(self, t: int = None, channel: int = None) -> gp.GeoSeries:
        """Get the image bounds."""
        if t == None:
            t = next(iter(self._images.timePoints()))
        if channel == None:
            channel = next(iter(self._images.channels(t=t)))
        # abb all channels within a given timepoint will have the same shape
        return self._images.shape(t, channel)

    def getPixels(self,
                  time: int,
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
                raise ValueError("zRange or z must be provided")

        return ImageSlice(self._images.fetchSlices(time, channel, (zRange[0], zRange[1] + 1), threeD))

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
        # logger.error(f'channel:{channel}')
        return self._images.getShapePixels(shapes, channel=channel, zSpread=zSpread, time=time, z=z)

def computeAggregateImage(dependencies: Union[List[str], dict[str, list[str]]] = {}, aggregate: list[str] = [], **attributes: Unpack[ImageColumnAttributes]):
    """A decorator that adds image based computed column to the schema.

    Args:
        dependencies (Union[List[str], dict[str, list[str]], optional):
            The dependencies of the computed column. 
            Use a dictionary to specify dependencies across multiple schemas with the schema name being the key and an array of dependency columns being the column.
            An array to specify dependencies within the same schema.
            The dependencies for the computed column. Defaults to {}.
            The dictionary can be used to specify dependencies across different schemas.
            {"schemaName": ["column1", "column2"], "schemaName2": ["column3", "column4"]}
        aggregate (list[str], optional): The aggregates to compute. Defaults to [].
            Aggregates must be numpy functions.
            For example, ["mean", "std", "min", "max"]
        title (str): The title of the column.
        categorical (bool): Indicates whether the column is categorical or not.
        divergent (bool): Indicates whether the column is divergent or not.
        description (str): The description of the column.
        group (str): The group to which the column belongs.
        colors (Union[List[Color], Dict[Any, Color]]): The colors associated with the column.
        symbols (Union[List[Symbol], Dict[Any, Symbol]]): The symbols associated with the column.
        plot (bool): Indicates whether the column should be plotted or not.
        version(int): The version of the computed column. When a computed column is updated, the version should be incremented so that the older versions of the columns are automatically invalidated and are recomputed.
        type (str): The type of the column.


    Returns:
        A function that returns a geo pandas data frame with a shape column with any name along with a z column.
    """
    # TODO: extend aggregates to allow user defined functions as well. Use the function name as the name of the aggregate
    def wrapper(func: Callable[[], Union[pd.Series, pd.DataFrame]]):
        func._imageComputed = {
            "key": func.__name__,
            "_aggregate": aggregate,
            **attributes,
            "_dependencies": dependencies,
        }
        return func
    return wrapper
