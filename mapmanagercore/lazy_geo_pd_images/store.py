# Adds image slices to lazy geo pandas

from typing import Callable, List, Self, Tuple, Union, Unpack, Set
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
def parseColumns2(columns: List[str],
                  prefix: str) -> Set[str]:
    """
    Parse the roi computed columns to get the channels and aggregates.
    
    Optimized to only get requested channels columns
    """    
    aggregates = set()
    for column in columns:
        if not column.startswith(prefix):
            continue
        # spineRoi_ch1_sum
        parts = column.split("_")
        if len(parts) < 3:
            continue
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

        # abb 20250825 this is getting called for things like Spine.roi and Spine.denRoi
        # but never for something simple like spineLen
        # logger.info(f'!!! DEBUG 20250825 _genWrappedFunc() called with:')
        # logger.info(f'  method:{method}')
        # logger.info(f'  attributes:{attributes}')
        # logger.info(f'  frame:{frame}')
        
        name = attributes["key"]
        func = method

        zSpread = attributes["zSpread"] if "zSpread" in attributes else 0
        tColumn = attributes["t"] if "t" in attributes else "t"

        # logger.info(f"check zSpread {zSpread} tColumn {tColumn}")
        timeIndexLevel = frame._schema._index.index(
            tColumn) if tColumn in frame._schema._index else None

        weakSelf = weakref.ref(self)

        def wrappedFunc(frame: LazyGeoFrame[Self]):

            # abj
            # channels = weakSelf().getActivatedChannels(timeIndexLevel)
            channels = weakSelf()._images.metadata.getTimepoint(timeIndexLevel).channelKeys  # List[int]
            # print(f'channels:{channels} type:{type(channels)}')
            
            # abb 20250825, instead of this, get aggregates from attributes parameter
            # aggregates = parseColumns2(
            #     frame.pendingColumns(), name)
            # attributes['_aggregate'] which is like ['size', 'sum', 'mean', 'min', 'max']
            aggregates = attributes['_aggregate'] 
                                    
            if len(channels) == 0 or len(aggregates) == 0:
                # nothin to update
                return gp.GeoDataFrame()

            # logger.warning('202508 _genWrappedFunc() called with:')
            # print(f'method:{method}')  # functions like: Spine.spineRoi, Spine.spineRoiBg
            # print(f'attributes:{attributes}')  # dict including '_aggregate': ['size', 'sum', 'mean', 'min', 'max']
            # print('frame:')
            # print(frame)
            # print(f'frame.pendingColumns():{frame.pendingColumns()}')
            # print(f'aggregates from parseColumns2():{aggregates}')

            shapes: gp.GeoDataFrame = func(frame)
            shapeKey = shapes.columns.symmetric_difference(["t", "z"])[0]
            
            # Create a copy to avoid SettingWithCopyWarning
            shapes = shapes.copy()
            shapes.rename(columns={shapeKey: "shape"}, inplace=True)
            
            # abb channels is always a list
            # channels = list(channels)
            
            # Check if shapes channels and zspread is the same
            if not len(self._recordShapePixels) == 0:
                oldChannels = self._recordShapePixels["channels"]
                oldShapes = self._recordShapePixels["shapes"]
                oldZSpread = self._recordShapePixels["zSpread"]
            else:
                oldChannels = oldShapes = oldZSpread = None

            # Compute the aggregates over the pixels

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
                except (KeyError) as e:  # abb 202508
                    logger.error(e)
                    logger.error("  no old pixels, must calculate new one")
                    logger.error(f"  shapes is: {shapes}")
                    logger.error(f"  channels is: {channels}")
                    logger.error(f"  zSpread is: {zSpread}")
                    pixels = weakSelf().getShapePixels(
                        shapes, channel=channels, zSpread=zSpread)
            
            else:
                # logger.info("202508 wrappedFunc is calling getShapePixels")
                # logger.info(f'!!! shapes is: {type(shapes)}')  # gp.GeoDataFrame
                # print(shapes)
                
                pixels = weakSelf().getShapePixels(
                    shapes, channel=channels, zSpread=zSpread)

                
            # self._recordShapePixels = {"channels": channels, "shapes": shapes, "zSpread": zSpread, "pixels": pixels}
            self._recordShapePixels["pixels"] = pixels

            if isinstance(pixels, pd.Series):
                # one channel was returned
                return pixels.apply(lambda x: pd.Series(
                    # {f"{name}_ch{pixels.name + 1}_{agg}": applyAgg(x, agg) for agg in aggregates}), index=pixels.index)
                    {f"{name}_ch{pixels.name}_{agg}": applyAgg(x, agg) for agg in aggregates}), index=pixels.index)

            # logger.error(f'REMOVE {channels} -> channels = [1]')
            # the channels processed by getShapePixels
            # _channels = list(pixels.columns)
            _channels = channels

            # abb 20250825 expanded below for readability
            # v1
            # return pd.DataFrame({
            #     # f"{name}_ch{channel + 1}_{agg}": pixels[channel].apply(lambda x: getattr(np, agg)(x)) for agg in aggregates for channel in channels
            #     f"{name}_ch{channel}_{agg}": pixels[channel].apply(lambda x: getattr(np, agg)(x)) for agg in aggregates for channel in _channels
            # }, index=pixels.index)

            # v2
            result_dict = {}
            for agg in aggregates:
                for channel in _channels:
                    # Create descriptive column name
                    column_name = f"{name}_ch{channel}_{agg}"
                    
                    # Apply the aggregation function to the channel data
                    aggregated_values = pixels[channel].apply(lambda x: getattr(np, agg)(x))
                    
                    # Store in result dictionary
                    result_dict[column_name] = aggregated_values

            return pd.DataFrame(result_dict, index=pixels.index)

        return wrappedFunc

    def addSchema(self, frame: LazyGeoFrame[Self], channelKeys: List[int]):
        """Add a new channel schema frame to the store.
        Essentially, this adds a new data frame to the store.
        This function is called after append channel
        
        abb this creates all spine intensity analysis columns

        Args:
            frame: spine DF
            channelKeys: List of channel keys to create computed columns for
        """

        # TODO: check if channel schema was already added
        if frame is None:
            return
        
        # logger.warning('!!==!! abb in store.py LazyImagesGeoPandas')

        # Use the provided channel keys directly
        currentChannelKeys = channelKeys

        # Inject computed columns that use the image to calculate roi stats
        # logger.info(f"frame check {frame}")
        for method in frame._schema.__dict__.values():
            if not hasattr(method, "_imageComputed"):
                continue

            attributes: ImageColumnAttributes = method._imageComputed
            # logger.info(f'checking attributes {attributes}')
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
    
    # abb depreciate this
    def _old_getActivatedChannels(self, t):
        return self._images.activatedChannels(t)

    def _old_getInActiveChannels(self, t):
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

    def getImageColumnNames(self, schema_class=None, timepoint: int = 1) -> List[str]:
        """
        Get the list of image column names that would be generated for a given schema.
        
        Args:
            schema_class: The schema class (e.g., Spine, Segment) to get columns for
            timepoint: The timepoint to get channel information from (default: 1)
        
        Returns:
            List[str]: List of image column names that would be generated
        """
        if schema_class is None:
            # If no schema provided, return all possible image columns
            return []
        
        # Get current channels from metadata
        try:
            channels = self._images.metadata.getTimepoint(timepoint).channelKeys
        except (KeyError, AttributeError):
            logger.warning(f"Could not get channel information for timepoint {timepoint}")
            return []
        
        # Get image methods and aggregates from the schema
        image_methods = []
        agg_list = []
        
        # Look for methods with @computeAggregateImage decorator
        for method_name, method in schema_class.__dict__.items():
            if hasattr(method, "_imageComputed"):
                attributes = method._imageComputed
                if "_aggregate" in attributes:
                    image_methods.append(attributes["key"])
                    agg_list = attributes["_aggregate"]
                    # logger.debug(f'Found image method: {attributes["key"]} with aggregates: {attributes["_aggregate"]}')
        
        # Generate column names
        column_names = []
        for method in image_methods:
            for channel in channels:
                for agg in agg_list:
                    column_name = f"{method}_ch{channel}_{agg}"
                    column_names.append(column_name)
        
        return column_names

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

    def getShapePixels(self,
                       shapes: gp.GeoDataFrame,
                       channel: Union[int, List[int]] = 0,  # abb 202508 get rid of default, caller has to know
                       zSpread: int = 0,
                       time=None,
                       z: int = None) -> Union[pd.Series, pd.DataFrame]:
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

def computeAggregateImage(dependencies: Union[List[str],
                                              dict[str, list[str]]] = {},
                                              aggregate: list[str] = [],
                                              **attributes: Unpack[ImageColumnAttributes]):
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
