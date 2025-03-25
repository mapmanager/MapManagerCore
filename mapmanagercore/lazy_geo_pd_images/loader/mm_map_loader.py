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

from mapmanagercore.lazy_geo_pd_images.loader.base import ImageLoader
from mapmanagercore.metadata.metadata3 import mmMapMetadata, TimepointMetadata, ChannelMetadata
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
    
    Used during runtime to create and manage an mmMap as well as to load/manage/save to an mmmap file.
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
                logger.error(f'path must be a directory or zip file')
                raise ValueError(f'path must be a directory or zip file')

        self.path = path
        self._metadata = mmMapMetadata()
        
        # abb depreciate
        # self._imagesSrcs: List[Dict[int, np.ndarray]] = []

        self.group = None
        self._imageChannelDict = {}

        if path is not None:
            self._loadZarFromPath()

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

    def importTimepoint(self, path) -> Optional[bool]:
        """Import and append a new timepoint from file (can be multiple channels).
        """
        
        # load all data from path
        imageImporter = getImageImporter(path, loadImgData=True)
        if imageImporter is None:
            logger.error(f'did not get image importer for path:{path}')
            return
                
        # append empty timepoint metadata
        _newTimepointKey = self.metadata.appendTimepoint()

        # metadata
        tpmd = self.metadata.getTimepointMetadata(_newTimepointKey)

        # logger.info(f'  _newTimepointKey:{_newTimepointKey}')

        for channelIdx in range(imageImporter.numChannels):
            imgData = imageImporter.getChannelData(channelIdx)
            channelName = imageImporter.channelNames[channelIdx]

            # # metadata
            # tpmd = self.metadata.getTimepointMetadata(_newTimepointKey)
            
            # this is appending channel 3 -->> why?
            _newChannelKey = tpmd.appendChannel(imgData, name=channelName)
            
            # raw data
            self._imageChannelDict[_newTimepointKey] = {}
            self._imageChannelDict[_newTimepointKey][_newChannelKey] = mmMapImageChannel(self, _newTimepointKey, _newChannelKey, imgData=imgData)
                
        return True
    
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
        timepointMetadata = self._metadata.getTimepointMetadata(timepoint)
        
        for _incomingChannelIdx in range(ii.numChannels):
            # fetch incoming raw data
            imgData = ii.getChannelData(_incomingChannelIdx)
            channelName = ii.channelNames[_incomingChannelIdx]

            # set metadata
            _newChannelKey = timepointMetadata.appendChannel(imgData, name=channelName)
            
            # set image data (adding a new key)
            _mmMapImageChannel = mmMapImageChannel(self, timepoint, _newChannelKey, imgData=imgData)
            self._imageChannelDict[timepoint][_newChannelKey] = _mmMapImageChannel

        return True
    
    def timepointShape(self, timepoint:int) -> Optional[Tuple[int, int, int]]:
        """Get the shape of imagedata at a timepoint.

        All channels have the same shape.

        Returns
        -------
            (z, y, x)
        """
        if not self.metadata.timepointExists(timepoint):
            return
        return self.metadata.getTimepointMetadata(timepoint).shape
    
    @property
    def numTimepoints(self) -> int:
        """Get the number of timepoints (imaging sessions).
        """
        return self.metadata.numTimepoints
    
    def numChannels(self, timepoint:int) -> Optional[int]:
        """Get the number of channels in a timepoint (imaging session).
        """
        return self.metadata.getTimepointMetadata(timepoint).numChannels
    
    @property
    def metadata(self) -> mmMapMetadata:
        """Get the metadata.
        """
        return self._metadata
    
    def getTimepointMetadata(self, t:str) -> TimepointMetadata:
        """Get Timepoint Metadata for a timepoint .
        """
        return self.metadata.getTimepointMetadata(t)
    
    def getChannelMetadata(self, t, c) -> ChannelMetadata:
        """Get ChannelMEtadata for timepoint and channel.
        """
        return self.getTimepointMetadata(t).getChannelMetadata(c)
    
    def getChannelData(self, timepoint:int, channel:int) -> Optional[np.ndarray]:
        """Get the image data for a channel at a timepoint.
        """
        if not self.metadata.timepointExists(timepoint):
            return
        _ret = self._imageChannelDict[timepoint][channel].getImageData()
        return _ret
    
    def _loadZarFromPath(self):
        """Load an mmap zarr from path.
        """
        if os.path.isdir(self.path):
            store = zarr.DirectoryStore(self.path)
        else:
            store = zarr.ZipStore(self.path, mode="r")
        
        logger.info(f'loading {type(store)} fom {self.path}')

        group: zarr.hierarchy.Group = zarr.group(store=store)
        self.group = group
        """root level of zarr file (on load will contain points, segment, image folders 0,1,2,..., etc)
        """

        # load metadata (use this to create placeholder for t/c)
        try:
            _metadataDict: dict = group.attrs['metadata']
        except (KeyError, ValueError):
            logger.error('metadata not found in zarr file')
            return
            
        # create metadata from loaded dict
        # dict is nested, need to use dataclasses_json @dataclass_json decorator
        self._metadata = mmMapMetadata.from_dict(_metadataDict)

        # logger.info(f'loaded self._metadata as {type(self._metadata)}')
        # pprint(self._metadata)
    
        # a dict of dict that holds each mmMapImageChannel
        for t in self.metadata.timepointKeys:
            self._imageChannelDict[t] = {}
            timepointMetadata = self.metadata.getTimepointMetadata(t)
            for c in timepointMetadata.channelKeys:
                # self._imageChannelDict[t][c] = mmMapImageChannel(self, t, c)
                self._imageChannelDict[t][c] = None

    def getImageChannel(self, t, c) -> "mmMapImageChannel":
        """Get an mmMapImageChannel.
        """
        if self._imageChannelDict[t][c] is None:
            self._imageChannelDict[t][c] = mmMapImageChannel(self, t, c)
        return self._imageChannelDict[t][c]
    
    def save(self):
        """Save to an existing mmap file.
        """
        if not self._pathExists():
            return
        self.saveAs(self.path)

    def saveAs(self, path:str) -> Optional[bool]:
        """Save the zarr file to disk.
        """
        if path.endswith('.mmap'):
            fs = zarr.DirectoryStore(path)
        elif path.endswith('.mmap.zip'):
            fs = zarr.ZipStore(path, mode="w", compression=zipfile.ZIP_STORED)
        else:
            logger.error('path must end with .mmap or .mmap.zip')
            logger.error(f'  got {path}')
            return
                    
        with fs as store:
            group: zarr.hierarchy.Group = zarr.group(store=store)

            # save what we have loaded (use metadata)
            # need to check if zarr on disk has items not loaded and delete them (be carful)
            # Remove keys from the zarr file that are not in the current metadata

            existing_channel_keys = {f'{t}/{c}'
                                     for t in group.keys()
                                     for c in group[t].keys()}
            expected_channel_keys = {f"{t}/{c}"
                                     for t in self.metadata.timepointKeys
                                     for c in self.metadata.getTimepointMetadata(t).channelKeys}
            channel_keys_to_remove = existing_channel_keys - expected_channel_keys
            # logger.info(f'existing_channel_keys:{existing_channel_keys}')
            # logger.info(f'expected_channel_keys:{expected_channel_keys}')
            # logger.info(f'channel_keys_to_remove:{channel_keys_to_remove}')
            
            for key in channel_keys_to_remove:
                logger.warning(f"Removing unused key from zarr file: {key}")
                del group[key]

            #
            group.attrs['metadata'] = self.metadata.asDict()
            
            # each image timepoint is in the root folder
            # this is compatible with ome-zarr
            # use metadata to traverse
            for t in self.metadata.timepointKeys:
                logger.info(f't:{t}')
                timepointMetadata = self.metadata.getTimepointMetadata(t)
                channelKeys = timepointMetadata.channelKeys
                for c in channelKeys:
                    # check if already in zarr
                    # do not save again (time and channel keys are immutable)
                    _channelPath = f'{t}/{c}'
                    if _channelPath in group:
                        logger.info(f'    {_channelPath} already in file skipping')
                        continue
                    else:
                        #TODO: true save as will have to LOAD ALL DATA (not lazy)
                        imgData = self.getChannelData(t, c)  # full image data
                        logger.info(f"    {_channelPath} saving -> {imgData.shape}")
                        # store imgData in zarr file/folder
                        group[f'{t}/{c}'] = imgData

        logger.info(f'zarr file saved to {path}')

        return True

    #
    # fake inheritance
    #
    
    # TODO: max channels is a misnomer. Each timepoint has a number of channels
    # can be 1,2,3, etc
    def maxChannels(self) -> int:
        # return 3
        return 2
    
    # abb we do not need c
    # all channels in a timepoint have the same shape
    def shape(self, t: int, c:int = 0) -> Tuple[int, int, int]:
        return self.timepointShape(t)
    
    def fetchSlices(self,
                    t:int,
                    channelIdx:int,
                    zRange:List[int]) -> np.ndarray:
        # channelIdx += 1
        mapImageChannel = self.getImageChannel(t, channelIdx)
        
        _firstSlice = zRange[0]
        return mapImageChannel.getSlice(_firstSlice)


    def timePoints(self) -> Iterator[int]:
        """
        Returns an iterator over the time points of the images.

        Returns:
            An iterator that yields the time points of the images.
        """
        return self.metadata.timepointKeys
    
    def channels(self, t:int) -> Iterator[int]:
        return self.metadata.getTimepointMetadata(t).channelKeys
    
    # abb this is WAY TO COMPLEX -->> rewrite
    def getShapePixels(self,
                       shape: gp.GeoDataFrame,
                       zSpread: int = 0,
                       channel: Union[int, List[int]] = 0,
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
        logger.info('TODO: FIX')
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
            if not "t" in shape.columns:
                shape.reset_index("t", inplace=True)
            else:
                shape.drop("t", axis=1, inplace=True)
                
        if time is not None:
            shape["t"] = time

        if not "z" in shape:
            if z is None:
                coords = shape["shape"].get_coordinates(include_z=True)
                shape["z"] = coords["z"].groupby(coords.index).mean()
            else:
                shape["z"] = z

        shape["z"] = shape["z"].astype(int)

        if isinstance(channel, list):
            for (t, z), group in shape.groupby(by=["t", "z"]):
                images = [self.fetchSlices(
                      t, c, (z - zSpread, z + zSpread + 1)) for c in channel]

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
            return pd.DataFrame(results, indexes, columns=channel)

        # logger.info(f"shape {shape}")
        for (t, z), group in shape.groupby(by=["t", "z"]):
            image = self.fetchSlices(
                t, channel, (z - zSpread, z + zSpread + 1))

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

class mmMapImageChannel():
    """Class to encapsulate a single channel stack in an mmap file.

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
        self._imgData = np.zeros(self.shapeMetadata.shape, dtype=self.channelMetaData.dtype)
        self._sliceLoaded = [False] * self.shapeMetadata.zPixels

        if imgData is not None:
            self._imgData = imgData
            self._sliceLoaded = [True] * self.shapeMetadata.zPixels

    def __str__(self):
        _str = f'{self._imgData.shape} num loaded slices: {sum(self._sliceLoaded)}'
        return _str
    
    @property
    def _channelPath(self) -> str:
        """Get the data path into the zarr file.
        """
        return f'{self.timepoint}/{self.channel}'
    
    def getSlice(self, sliceIdx:int) -> np.ndarray:
        """Fetch a single image slice.
        
        Will load from zarr first time.
        """
        # option 1
        # if self._imageDict[sliceIdx] is None:
        #     # load from zarr
        #     imgData = self.mapLoader.group[self._channelPath][sliceIdx,:,:]  # <zarr.core.Array '/1/1' (70, 1024, 1024) uint16>
        #     self._imageDict[sliceIdx] = imgData
        # return self._imageDict[sliceIdx]
    
        # option 2
        if not self._sliceLoaded[sliceIdx]:
            self._imgData[sliceIdx,:,:] = self.mapLoader.group[self._channelPath][sliceIdx,:,:]
            self._sliceLoaded[sliceIdx] = True
        return self._imgData[sliceIdx,:,:]

    def getVolume(self, startSlice:int, stopSlice:Optional[int] = None) -> np.ndarray:
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