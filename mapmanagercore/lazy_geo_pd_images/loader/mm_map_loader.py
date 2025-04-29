"""
Rewrite of zarr load.
"""

import os
from pprint import pprint
from typing import Optional, List, Tuple, Union, Iterator

import zarr
import zipfile

import numpy as np
import pandas as pd

import geopandas as gp

# from mapmanagercore.lazy_geo_pd_images.loader.base import ImageLoader
from mapmanagercore.metadata import mmMapMetadata, TimepointMetadata, ChannelMetadata
from mapmanagercore.imageImporter import getImageImporter

from mapmanagercore.logger import logger

# TODO: move shapely code somewhere else
# this file/class is to load/save mmap zarr files
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
import skimage.draw  # only used for skimage.draw.polygon and skimage.draw.line

def shapeIndexes(d: Union[Polygon, LineString]) -> Tuple[np.ndarray, np.ndarray]:
    """ Get the x and y indexes of the pixels in a shape."""

    d = shapely.force_2d(d)

    # TODO: fully support multi-polygon
    if isinstance(d, MultiPolygon):
        d = d.geoms[0]
    if isinstance(d, GeometryCollection):
        d = next(s for s in d.geoms if isinstance(s, Polygon))

    if isinstance(d, Polygon):
        coords = d.exterior.coords.xy
    else:
        coords = d.coords.xy

    x, y = np.array(coords[0]), np.array(coords[1])
    # Since skimage.draw does not support negative coordinates, we need to shift the coordinates.
    minx, miny = int(min(min(x), 0)), int(min(0, min(y)))
    x, y = x - minx, y - miny

    if isinstance(d, Polygon):
        xs, ys = skimage.draw.polygon(x, y)
    else:
        xs, ys = skimage.draw.line(int(x[0]), int(y[0]), int(x[1]), int(y[1]))

    # Shift the coordinates back to the original position.
    xs, ys = xs + minx, ys + miny
    return xs, ys
    
class mmMapLoader():
    """Class to encapsulate an mmMap loader.
    
    Used during runtime to create and manage an mmMap
    as well as to load/manage/save to an mmap file.
    """
    def __init__(self,
                 path: Optional[str] = None):
        """
        Encapsulates an mmap file.

        Args:
            path (str): The path to the .mmap Zarr file.
                If None then create an empty zarr loader
        """
        if path is not None:
            # ensure path is a directory or zip file
            if not os.path.isdir(path) and not path.endswith('.zip'):
                logger.error(f'path must be a directory or zip file {path}')
                raise ValueError(f'path must be a directory or zip file')

        self.path = path
        self._metadata = mmMapMetadata()
        
        # abb depreciate
        # self._imagesSrcs: List[Dict[int, np.ndarray]] = []

        self.group = None
        self._imageChannelDict = {}

        if path is not None:
            self._load()

    def deleteChannel(self, timepoint:int, channel:int):
        """Delete an image channel.
        """
        logger.warning('TODO')

        # 1 metadata

        # 2 images

    def moveChannel(srcTimePoint, srcChannel, 
                    destTimePoint, destChannel):
        logger.info('TODO')

    def _pathExists(self) -> bool:
        _ret = True
        path = self.path
        if path is None:
            _ret = False
        else:
            # ensure path is a directory or zip file
            if not os.path.isdir(path) and not path.endswith('.zip'):
                logger.error(f'path must be a directory or zip file')
                # raise ValueError(f'path must be a directory or zip file')
                _ret = False
        return _ret
    
    def print(self):
        for t,tv in self._imageChannelDict.items():
            print(f't:{t}')
            for c,cv in tv.items():
                print(f'  c:{c} {cv}')

    def importTimepoint(self, path) -> Optional[int]:
        """Import and append a new timepoint from file (can be multiple channels).
        """
        
        # load all data from path
        imageImporter = getImageImporter(path, loadImgData=True)
        if imageImporter is None:
            logger.error(f'did not get image importer for path:{path}')
            return

        # logger.info(f'  _newTimepointKey:{_newTimepointKey}')

        for channelIdx in range(imageImporter.numChannels):
            imgData = imageImporter.getChannelData(channelIdx)
            channelName = imageImporter.channelNames[channelIdx]

            if channelIdx == 0:
                # append empty timepoint metadata
                _newTimepointKey = self.metadata.appendTimepoint(imgData)

                # metadata
                tpmd = self.metadata.getTimepoint(_newTimepointKey)

                # first channel, the one created with appendTimepoint
                _newChannelKey = tpmd.channelKeys[0]
            else:
                _newChannelKey = tpmd.appendChannel(imgData, name=channelName)
            
            # logger.error(f'xxx self:{self} _newTimepointKey:{_newTimepointKey} _newChannelKey:{_newChannelKey} imgData:{imgData.shape}')
            
            # raw data
            self._imageChannelDict[_newTimepointKey] = {}
            self._imageChannelDict[_newTimepointKey][_newChannelKey] = \
                ImageChannel(self, _newTimepointKey, _newChannelKey, imgData=imgData)
                
        return _newTimepointKey
    
    def importChannel(self, path:str,
                       timepoint:int) -> Optional[bool]:
        """Open a file and append channels to an existing timepoint.
        """
        # check that timepoint exists
        if not self.metadata.timepointExists(timepoint):
            logger.error(f'timepoint "{timepoint}" does not exist, expecting one of {self.metadata.timepointKeys}')
            return

        # get image importer from path
        ii = getImageImporter(path, loadImgData=False)  # remember to load pixels with loadData()
        if ii is None:
            logger.error('failed')
            return
        
        # check incoming channel shape matches our shape (at timepoint t)
        _incomingShape = ii.channelShape
        _existingShape = self.timepointShape(timepoint)  # abb this should use metadata ???
        if _incomingShape != _existingShape:
            logger.error(f' img shape mismatch, expecting:{_existingShape} but got {_incomingShape}')
            return
        
        # load actual pixels
        ii.loadData()

        # append all incoming channels to timepoint
        timepointMetadata = self._metadata.getTimepoint(timepoint)
        
        for _incomingChannelIdx in range(ii.numChannels):
            # fetch incoming raw data
            imgData = ii.getChannelData(_incomingChannelIdx)
            channelName = ii.channelNames[_incomingChannelIdx]

            # set metadata
            _newChannelKey = timepointMetadata.appendChannel(imgData, name=channelName)
            
            # logger.warning(f'timepoint:{timepoint} {type(timepoint)}')
            # logger.warning(f'_newChannelKey:{_newChannelKey} {type(_newChannelKey)}')
            # logger.warning(f'self._imageChannelDict.keys() is: {self._imageChannelDict.keys()}')

            # _timepointStr = str(timepoint)

            # set image data (adding a new key)
            _imageChannel = ImageChannel(self, timepoint, _newChannelKey, imgData=imgData)
            self._imageChannelDict[timepoint][_newChannelKey] = _imageChannel

        return True
    
    def timepointShape(self, timepoint:int) -> Optional[Tuple[int, int, int]]:
        """Get the shape of image data at a timepoint.

        All channels have the same shape.

        Returns
        -------
            (z, y, x)
        """
        if not self.metadata.timepointExists(timepoint):
            return
        return self.metadata.getTimepoint(timepoint).shape
    
    @property
    def numTimepoints(self) -> int:
        """Get the number of timepoints (imaging sessions).
        """
        return self.metadata.numTimepoints
    
    def numChannels(self, timepoint:int) -> Optional[int]:
        """Get the number of channels in a timepoint (imaging session).
        """
        return self.metadata.getTimepoint(timepoint).numChannels
    
    @property
    def metadata(self) -> mmMapMetadata:
        """Get the metadata.
        """
        return self._metadata
    
    # baltimore april -> need to use complex core single timepoint
    def getTimepointMetadata(self, t:str) -> TimepointMetadata:
        """Get Timepoint Metadata for a timepoint .
        """
        return self.metadata.getTimepoint(t)
    
    def getChannelMetadata(self, t, c) -> ChannelMetadata:
        """Get ChannelMEtadata for timepoint and channel.
        """
        return self.getTimepointMetadata(t).getChannelMetadata(c)
    
    def getChannelData(self, timepoint:int, channel:int) -> Optional[np.ndarray]:
        """Get the image data for a channel at a timepoint.
        """
        if not self.metadata.timepointExists(timepoint):
            return
        
        # to do, we really need a class to hold image data
        # keys got converted to str
        # if isinstance(timepoint, int):
        #     timepoint = str(timepoint)
        
        _ret = self._imageChannelDict[timepoint][channel].getImageData()
        return _ret

    def getImageChannel(self, t, c) -> "ImageChannel":
        """Get an ImageChannel.
        """
        # TODO: check (t, c) exists and handle error if they don't

        if self._imageChannelDict[t][c] is None:
            self._imageChannelDict[t][c] = ImageChannel(self, t, c)
        return self._imageChannelDict[t][c]
    
    def _getStore(self, path: Optional[str]) -> zarr.DirectoryStore | zarr.ZipStore:
        """Get a zarr store from path.
        """
        if path is None:
            path = self.path
        
        if os.path.isdir(path):
            store = zarr.DirectoryStore(path)
        else:
            store = zarr.ZipStore(self.path, mode="r", compression=zipfile.ZIP_STORED)
        
        return store
    
    def _getChannelPaths(self, path: str) -> Optional[List[str]]:
        """Get channel path from zarr file.

        Returns
        -------
        List of str like ['1/1', '1/2', '2/1]
        """
        fs = self._getStore(path)
        with fs as store:
            group: zarr.hierarchy.Group = zarr.group(store=store)
            try:
                _metadataDict: dict = group.attrs['metadata']
                # _metadata = mmMapMetadata.from_dict(_metadataDict)
                channelPathsList = []
                for timepointKey, timepoint in _metadataDict['timepoints'].items():
                    for channelKeys in timepoint['channels'].keys():
                        channelPathsList.append(f'{timepointKey}/{channelKeys}')
            except (KeyError) as e:
                # no metadata in file
                logger.error(f'while fetching "metadata" -> {e}')
                return
        return channelPathsList
    
    def _zarrExists(self, path: str) -> bool:
        return os.path.isdir(path) or os.path.isfile(path)
    
    def _load(self):
        """Load an .mmap or .mmap.zip from path.
        """
        if os.path.isdir(self.path):
            # store = zarr.DirectoryStore(self.path)
            _zipStore = False
        else:
            # store = zarr.ZipStore(self.path, mode="r")
            _zipStore = True
        
        # logger.info(f'loading {type(store)} fom {self.path}')

        with zarr.ZipStore(self.path, mode="r") \
            if _zipStore else zarr.DirectoryStore(self.path) as store:

            logger.info(f'using {store} to load path:{self.path}')
            
            # zarr.errors.PathNotFoundError: nothing found at path ''
            self.group: zarr.hierarchy.Group = zarr.open(store=store, mode='r')  # / zarr.hierarchy.Group
            # self.group: zarr.hierarchy.Group = zarr.group(store=store)
            """root level of zarr file (on load will contain points, segment, image folders 0,1,2,..., etc)
            """

            # load metadata (use this to create placeholder for t/c)
            try:
                # fails on load zip ???
                # logger.info(f"group.attrs['metadata']:{group.attrs['metadata']}")
                _metadataDict: dict = self.group.attrs['metadata']
                # _metadataDict: dict = store.attrs['metadata']
            except (KeyError, ValueError):
                logger.error('metadata not found in zarr file')
                return
                
            # create metadata from loaded dict
            # dict is nested, need to use dataclasses_json @dataclass_json decorator
            self._metadata = mmMapMetadata.from_dict(_metadataDict)

            # logger.info(f'loaded self._metadata as {type(self._metadata)}')
            # pprint(self._metadata)
        
            # a dict of dict that holds each ImageChannel
            for t in self.metadata.timepointKeys:
                self._imageChannelDict[t] = {}
                timepointMetadata = self.metadata.getTimepoint(t)
                for c in timepointMetadata.channelKeys:
                    # self._imageChannelDict[t][c] = ImageChannel(self, t, c)
                    self._imageChannelDict[t][c] = ImageChannel(self, t, c)
    
    def save(self):
        """Save to an existing mmap file.
        """
        if not self._pathExists():
            return
        self.saveAs(self.path)

    def saveAs(self, path:str) -> Optional[bool]:
        """Save the zarr file to disk.
        """
        if path.endswith('.mmap.zip'):
            logger.error('can not save to .zip')
            return
    
        if path.endswith('.mmap'):
            _zipStore = False
        elif path.endswith('.mmap.zip'):
            _zipStore = True
        else:
            logger.error('path must end with .mmap or .mmap.zip')
            logger.error(f'  got {path}')
            return

        _zarrExists = self._zarrExists(path)
        logger.info(f'saving to _zarrExists:{_zarrExists} path:{path}')

        # with fs as store:
        with zarr.ZipStore(path, mode="w", compression=zipfile.ZIP_STORED) if _zipStore \
            else zarr.DirectoryStore(path) as store:
            
            group: zarr.hierarchy.Group = zarr.group(store=store)

            # save what we have loaded (use metadata)
            # need to check if zarr on disk has items not loaded and delete them (be carful)
            # Remove keys from the zarr file that are not in the current metadata

            # if existing mmap file
            if _zarrExists and not _zipStore:
                _existingMetadata = self._getChannelPaths(path)
                if _existingMetadata is None:
                    # zarr exists but no metadata yet
                    logger.info(f'no metadata in path:{path}')
                else:
                    # logger.info('from path _existingMetadata:')
                    # pprint(_existingMetadata)
                    
                    # existing_channel_keys = {f"{t}/{c}"
                    #                         for t in _existingMetadata.timepointKeys
                    #                         for c in _existingMetadata.getTimepoint(t).channelKeys}
                    existing_channel_keys = set(_existingMetadata)

                    # current runtime
                    expected_channel_keys = {f"{t}/{c}"
                                            for t in self.metadata.timepointKeys
                                            for c in self.metadata.getTimepoint(t).channelKeys}

                    # logger.info('runtime expected_channel_keys:')
                    # pprint(expected_channel_keys)

                    channel_keys_to_remove = existing_channel_keys - expected_channel_keys

                    # logger.info(f'file existing_channel_keys:{existing_channel_keys}')
                    # logger.info(f'runtime expected_channel_keys:{expected_channel_keys}')
                    # logger.info(f'channel_keys_to_remove:{channel_keys_to_remove}')
                
                    for key in channel_keys_to_remove:
                        logger.warning(f"  Removing unused key from zarr file: {key}")
                        del group[key]

            #
            group.attrs['metadata'] = self.metadata.asDict()
            
            # each image timepoint is in the root folder
            # this is compatible with ome-zarr
            # use metadata to traverse
            for t in self.metadata.timepointKeys:
                # logger.info(f't:{t}')
                timepointMetadata = self.metadata.getTimepoint(t)
                channelKeys = timepointMetadata.channelKeys
                for c in channelKeys:
                    # check if already in zarr
                    # do not save again (time and channel keys are immutable)
                    _channelPath = f'{t}/{c}'
                    # logger.info(f'  _channelPath:{_channelPath}')
                    if _channelPath in group:
                        logger.info(f'    {_channelPath} already in file skipping')
                        continue
                    else:
                        #TODO: true save as will have to LOAD ALL DATA (not lazy)
                        if _zipStore:
                            self.getImageChannel(t, c).loadAllImageData()
                            logger.info(f'_zipStore loaded numLoaded:{self.getImageChannel(t, c).numLoaded}')
                        imgData = self.getChannelData(t, c)  # full image data
                        # logger.info(f"    {_channelPath} saving -> {imgData.shape}")
                        # store imgData in zarr file/folder
                        group[_channelPath] = imgData

        # logger.info(f'zarr file saved to {path}')

        return True

    # TODO: max channels is a misnomer. Each timepoint has a number of channels
    # can be 1,2,3, etc
    def _old_maxChannels(self) -> int:
        # return 3
        return 2
    
    # todo: depreciate, called by old MapAnnotation code
    # abb we do not need c
    # all channels in a timepoint have the same shape
    def shape(self, t: int, c:int = 0) -> Tuple[int, int, int]:
        return self.timepointShape(t)
    
    def fetchSlices(self,
                    t:int,
                    channelIdx:int,
                    zRange:List[int]) -> np.ndarray:
        # channelIdx += 1
        _imageChannel = self.getImageChannel(t, channelIdx)
        
        _firstSlice = zRange[0]
        return _imageChannel.getSlice(_firstSlice)  # lazy


    def timePoints(self) -> Iterator[int]:
        """
        Returns an iterator over the time points of the images.

        Returns:
            An iterator that yields the time points of the images.
        """
        return self.metadata.timepointKeys
    
    def channels(self, t:int) -> Iterator[int]:
        return self.metadata.getTimepoint(t).channelKeys
    
    # abb this is WAY TO COMPLEX -->> rewrite
    def getShapePixels(self,
                       shape: gp.GeoDataFrame,
                       zSpread: int = 0,
                    # abb remove dependency on default (channels are now 1 based)
                    # at this level, we don't need to do that, use
                    # mmMap.metadata._possibleChannels
                    #    channel: Union[int, List[int]] = 0,
                       channel: Union[int, List[int]] = 1,
                       time=None,
                       z: int = None):
        """
        Retrieve image slices corresponding to the given shape.

        Args:
            shape (gp.GeoDataFrame): GeoDataFrame containing the shape under the column polygon, along with `z` and time `t`.
            zSpread (int, optional): Number of slices to expand in the z-direction. Defaults to 0.
            channel (int, optional): Channel index. Defaults to 0.
            time (int, optional): Time index. Defaults to None. If provided, the time index will be used instead of the `t` column in the shape.
            z (int, optional): Z index. Defaults to None. If provided, the z index will be used instead of the `z` column in the shape.

        Returns:
            pd.Series: Series containing the image slices corresponding to the shape.
        """
        # logger.info('TODO: FIX')
        # print(f'  shape is:')
        # print(shape)
        # print(f'  zSpread:{zSpread}')
        # print(f'  channel:{channel}')
        # print(f'  time:{time}')
        # print(f'  z:{z}')
        
        results = []
        indexes = []

        if isinstance(shape, list):
            shape = gp.GeoDataFrame(shape, columns=["shape"], geometry="shape")

        if isinstance(shape, pd.Series) or isinstance(shape, gp.GeoSeries):
            shape = shape.to_frame("shape")

        if "t" in shape.index.names:
            # logger.info(f"'t' is in shape.index.names ... why is this important???")
            if "t" not in shape.columns:
                shape.reset_index("t", inplace=True)
            else:
                shape.drop("t", axis=1, inplace=True)

        # time is sometimes None
        # if time is None:
        #     logger.warning(f'time is:{time} channel:{channel}')
        # abb if channel is list, reduce to available in each timepoint

        if time is not None:
            shape["t"] = time

        if "z" not in shape:
            if z is None:
                coords = shape["shape"].get_coordinates(include_z=True)
                shape["z"] = coords["z"].groupby(coords.index).mean()
            else:
                shape["z"] = z

        shape["z"] = shape["z"].astype(int)

        _firstTimepointChannels = channel
        if isinstance(channel, list):
            # abb does this handle different timepoint with different channels?
            for (t, z), group in shape.groupby(by=["t", "z"]):
                # abb only fetch channels that exis
                # channel list is metadata _possibleChannels
                # logger.warning(f'(t,z) is t:{t} z:{z}')
                # abb limit channels to those that exist in timpepoint t
                _channelKeys = self.metadata.getTimepoint(t).channelKeys
                # abb
                _firstTimepointChannels = _channelKeys
                
                images = [self.fetchSlices(
                    #   t, c, (z - zSpread, z + zSpread + 1)) for c in channel]
                      t, c, (z - zSpread, z + zSpread + 1)) for c in _channelKeys]

                for idx, row in group.iterrows():
                    xLim, yLim = images[0].shape
                    xs, ys = shapeIndexes(row["shape"])
                    # Clip the coordinates to the image bounds.
                    inBounds = (xs >= 0) & (xs < xLim) & (
                        ys >= 0) & (ys < yLim)
                    xs = np.clip(xs, 0, xLim - 1)
                    ys = np.clip(ys, 0, yLim - 1)

                    # inject the nan values where the shape is out of bounds.
                    results.append(
                        # [np.where(inBounds, image[xs, ys], np.nan) for image in images])
                        # abj: accounting for pixels being inverted when plotting by switching ys and xs
                        [np.where(inBounds, image[ys, xs], np.nan) for image in images]) 
                    indexes.append(idx)
            
            # abb this is tricky, sometimes `channel` is int, sometimes list ???
            # print(len(results))
            # print(len(indexes))
            # print(len(channel))
            # return pd.DataFrame(results, indexes, columns=channel)
            # logger.warning(f'  -->> _firstTimepointChannels:{_firstTimepointChannels}')
            return pd.DataFrame(results, indexes, columns=_firstTimepointChannels)

        # logger.info(f"shape {shape}")
        for (t, z), group in shape.groupby(by=["t", "z"]):
            # abb
            # _channelKeys = self.metadata.getTimepoint(t).channelKeys

            image = self.fetchSlices(
                t, channel, (z - zSpread, z + zSpread + 1))
                # t, _channelKeys, (z - zSpread, z + zSpread + 1))

            for idx, row in group.iterrows():
                xLim, yLim = image.shape
                xs, ys = shapeIndexes(row["shape"])
                # Clip the coordinates to the image bounds.
                inBounds = (xs >= 0) & (xs < xLim) & (ys >= 0) & (ys < yLim)
                xs = np.clip(xs, 0, xLim - 1)
                ys = np.clip(ys, 0, yLim - 1)

                # inject the nan values where the shape is out of bounds.
                # results.append(np.where(inBounds, image[xs, ys], np.nan))

                # abj: accounting for pixels being inverted when plotting by switching ys and xs
                results.append(np.where(inBounds, image[ys, xs], np.nan)) 
                indexes.append(idx)

        return pd.Series(results, indexes, name=channel)

class ImageChannel():
    """Class to encapsulate a single color channel stack.

    Provides lazy loading of individual images.
    """
    def __init__(self,
                 mapLoader: mmMapLoader,
                 timepoint: str,
                 channel: str,
                 imgData: np.ndarray = None):
        """
        Arguments
        ---------
            mapLoader
                A mmMapLoader encapsulating a mmap zarr file
            
        """
        # root level of zarr file (on load will contain points, segment, image folders 0,1,2,..., etc)
        self.mapLoader:mmMapLoader = mapLoader
        self.timepoint = timepoint
        self.channel = channel

        # shared across all channels in a timepoint
        self.shapeMetadata = self.mapLoader.getTimepointMetadata(timepoint).shapeMetadata
        
        self.channelMetaData = self.mapLoader.getChannelMetadata(timepoint, channel)

        # these seem to be the same, speed wise
        # option 1 is a dict of dict of nparray
        # option2 is a pre-allocated np.array with proper shape
        
        # option 1, a dict of (x,y) slice np.ndarray
        # numSlices = self.shapeMetadata.zPixels
        # self._imageDict = {k:None for k in range(numSlices)}

        # option 2, preallocate np.array
        self._initFromMetadata()

        if imgData is not None:
            # used when importing from file
            self._imgData = imgData
            self._sliceLoaded = [True] * self.shapeMetadata.zPixels

    def _initFromMetadata(self):
        """Initialize from image data.
        
        Used in __init__() and unload all.
        """
        self._imgData = np.zeros(self.shapeMetadata.shape, dtype=self.channelMetaData.dtype)
        self._sliceLoaded = [False] * self.shapeMetadata.zPixels

    def __str__(self):
        _str = f'{self._imgData.shape} num loaded slices: {sum(self._sliceLoaded)}'
        return _str
    
    @property
    def _channelPath(self) -> str:
        """Get the data path into the zarr file.
        """
        return f'{self.timepoint}/{self.channel}'

    @property
    def numSlices(self) -> int:
        return self.shapeMetadata.shape[0]

    @property
    def numLoaded(self) -> int:
        """Get the number of slices loaded into memory.
        """
        return self._sliceLoaded.count(True)
    
    def loadAllImageData(self):
        """Load all image data into memory.
        
        Used when exporting to zar.zip
        """
        self.getVolume(0, self.numSlices - 1)

    def unloadAllImageData(self):
        """Unload all image data from memory.
        """
        self._initFromMetadata()

    def _sliceIsLoaded(self, sliceIdx) -> bool:
        return self._sliceLoaded[sliceIdx]
    
    def getSlice(self, sliceIdx:int) -> np.ndarray:
        """Fetch a single image slice.
        
        Will load from zarr first time.
        """
        # option 1, using dict with one slice per key
        # if self._imageDict[sliceIdx] is None:
        #     # load from zarr
        #     imgData = self.mapLoader.group[self._channelPath][sliceIdx,:,:]  # <zarr.core.Array '/1/1' (70, 1024, 1024) uint16>
        #     self._imageDict[sliceIdx] = imgData
        # return self._imageDict[sliceIdx]
    
        # option 2, pre allocate an nparray
        if not self._sliceIsLoaded(sliceIdx):
            # logger.info(list(self.mapLoader.group.keys()))
            self._imgData[sliceIdx,:,:] = self.mapLoader.group[self._channelPath][sliceIdx,:,:]
            self._sliceLoaded[sliceIdx] = True
        return self._imgData[sliceIdx,:,:]

    def getVolume(self,
                  startSlice:int,
                  stopSlice:Optional[int] = None) -> np.ndarray:
        """Get a volume of slices."""
        # self.mapLoader.group[self._channelPath] is like:
        #   <zarr.core.Array '/1/1' (70, 1024, 1024) uint16>
        
        # option 1
        # if stopSlice is None:
        #     # imgData = self.mapLoader.group[self._channelPath][startSlice,:,:]  
        #     imgData = self.getSlice(startSlice)
        # else:
        #     # always load from zarr
        #     imgData = self.mapLoader.group[self._channelPath][startSlice:stopSlice,:,:]

        # option 2
        if stopSlice is None:
            imgData = self.getSlice(startSlice)
        else:
            # ensure all slices are loaded
            _sliceRange = np.arange(startSlice,stopSlice+1, 1)  # to include last slice
            for _sliceIdx in _sliceRange:
                self.getSlice(_sliceIdx)
            imgData = self._imgData[_sliceRange,:,:]
        
        return imgData
    
    def getImageData(self) -> np.ndarray:
        return self._imgData
        
