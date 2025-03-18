"""
Rewrite of zarr load.
"""

import os
from pprint import pprint
from typing import Optional, List, Dict

import zarr
import zipfile

import numpy as np

from mapmanagercore.lazy_geo_pd_images.loader.base import ImageLoader
from mapmanagercore.metadata.metadata3 import mmMapMetadata, TimepointMetadata
from mapmanagercore.imageImporter import getImageImporter

from mapmanagercore.logger import logger

class ZarrLoader2(ImageLoader):
    def __init__(self, path: Optional[str] = None, lazy: bool = True):
        """
        Initializes a ZarrLoader object.

        Args:
            path (str): The path to the .mmap Zarr file.
                If None then create an empty zarr loader
            lazy (bool, optional): If True, the images will be loaded lazily.
                If False, the images will be loaded eagerly. Defaults to True.
        """
        if path is not None:
            # ensure path is a directory or zip file
            if not os.path.isdir(path) and not path.endswith('.zip'):
                logger.error(f'path must be a directory or zip file')
                raise ValueError(f'path must be a directory or zip file')

        super().__init__()

        self.path = path
        self._metadata = mmMapMetadata()
        self._imagesSrcs: List[Dict[int, np.ndarray]] = []

        self.group = None

        if path is not None:
            self.loadZarFromPath(lazy=lazy)

    def _images(self, t: int, channel: int) -> np.ndarray:
        return self.getChannelData(t, channel)
        # if not self._timepoint_exists(t):
        #     logger.error(f'timepoint {t} does not exist, available timepoints are {self.metadata.timepointIndices}')
        #     return
        # # TODO: write api for this in TimePointMEtadata
        # _channelIndices = self.metadata.getTimepointMetadata(t).channelIndices
        # if channel not in _channelIndices:
        #     logger.error(f'timepoint {t} does not have channel {channel}, available channels are {_channelIndices}')
        #     return

    def appendTimepoint(self, path, verbose=False) -> Optional[bool]:
        """Append a new timepoint from file (can be multiple channels).
        """
        ii = getImageImporter(path, loadImgData=True)
        if ii is None:
            logger.error(f'did not get image importer for path:{path}')
            return
        
        if verbose:
            logger.info(f'ii is:{ii}')

        # empty timepoint metadata
        newTimepointMetadata = TimepointMetadata()

        for channelIdx in range(ii.numChannels):
            imgData = ii.getChannelData(channelIdx)
            channelName = ii.channelNames[channelIdx]

            # metadata
            newTimepointMetadata.appendChannel(imgData, name=channelName)
            
            # image data (one based just like metadata!!!)
            oneChannelDict = {}
            oneChannelDict[channelIdx+1] = imgData
            self._imagesSrcs.append(oneChannelDict)

        self._metadata.appendTimepoint(newTimepointMetadata)

        return True
    
    def appendChannels(self, path:str, timepoint:int) -> Optional[bool]:
        """Open a file and append channels to an existing timepoint.
        """
        # check that timepoint exists
        if not self._timepoint_exists(timepoint):
            logger.error(f'timepoint "{timepoint}" does not exist, expecting one of {range(self.numTimepoints)}')
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
            # fet incoming raw data
            imgData = ii.getChannelData(_incomingChannelIdx)
            channelName = ii.channelNames[_incomingChannelIdx]

            # set metadata
            timepointMetadata.appendChannel(imgData, name=channelName)
            
            # set image data (adding a new key)
            _newChannelKey = self._getNewChannelKey(timepoint)
            logger.info(f'timepoint:{timepoint} _newChannelKey:{_newChannelKey}')
            timepoint -= 1
            self._imagesSrcs[timepoint][_newChannelKey] = imgData  # adding a new key

        return True
    
    def timepointShape(self, timepoint:int):
        """Get the shape of imagedata at a timepoint.

        All channels have the same shape
        """
        if not self._timepoint_exists(timepoint):
            return
        return self.metadata.getTimepointMetadata(timepoint).shape
    
    def _getNewChannelKey(self, timepoint:int) -> Optional[int]:
        """Get next available channel key."""
        if not self._timepoint_exists(timepoint):
            return
        
        timepoint -= 1
        
        _keyList = list(self._imagesSrcs[timepoint].keys())
        _maxKey = max(_keyList)
        _newKey = _maxKey + 1
        # logger.info(_newKey)
        return _newKey
    
    @property
    def numTimepoints(self) -> int:
        """Get the number of timepoints (imaging sessions).
        """
        # return len(self._imagesSrcs)
        return self.metadata.numTimepoints
    
    def numChannels(self, timepoint:int) -> Optional[int]:
        """Get the number of channels in a timepoint (imaging session).
        """
        return self.metadata.getTimepointMetadata(timepoint).numChannels
    
        # if not self._timepoint_exists(timepoint):
        #     return
        # timepoint -= 1
        # return len(self._imagesSrcs[timepoint].keys())
    
    @property
    def metadata(self) -> mmMapMetadata:
        """Get the metadata object.
        """
        return self._metadata
    
    # TODO: use metadata
    def _timepoint_exists(self, t:int):
        # return t < self.numTimepoints
        for aDict in self._imagesSrcs:
            if t in aDict.keys():
                return True
            
        logger.error(f'timepoint {t} does not exist')
        return False
    
    def _printImgSrcs(self):
        # _imagesSrcs is a list of dict (key, nparray)
        logger.info(f'_imagesSrcs is a list with {len(self._imagesSrcs)} timepoints:')
        for idx, aDict in enumerate(self._imagesSrcs):
            logger.info(f'  timepoint idx:{idx} has {len(aDict.keys())} channels')
            for aDictKey in aDict.keys():
                logger.info(f'    channel key:{aDictKey} shape:{aDict[aDictKey].shape}')
    
    def getChannelData(self, timepoint:int, channel:int) -> Optional[np.ndarray]:
        """Get the image data for a channel at a timepoint.
        """
        if not self._timepoint_exists(timepoint):
            return
        
        timepoint -= 1
        if channel not in self._imagesSrcs[timepoint].keys():
            logger.error(f'channel {channel} does not exist, expecting one of {self._imagesSrcs[timepoint].keys()}')
            return
        
        _ret = self._imagesSrcs[timepoint][channel]
        logger.info(f'returning image data shape {_ret.shape}')
        return _ret
    
    # abb not done
    def loadZarFromPath(self, lazy: bool = False):
        """Load an mmap zarr from path.
        """
        if os.path.isdir(self.path):
            store = zarr.DirectoryStore(self.path)
        else:
            store = zarr.ZipStore(self.path, mode="r")
        
        group: zarr.hierarchy.Group = zarr.group(store=store)
        self.group = group  # root level of zarr file (on load will contain points, segment, ...)

        # load metadata
        try:
            _metadataDict: dict = group.attrs['metadata']
        except (KeyError, ValueError):
            logger.error('metadata not found in zarr file')
            return
            
        # create metadata from loaded dict
        # dict is nested, need to use dataclasses_json @dataclass_json decorator
        self._metadata = mmMapMetadata.from_dict(_metadataDict)

        logger.info(f'loaded self._metadata as {type(self._metadata)}')
        # pprint(self._metadata)
    
        for t in self.metadata.getTimepointKeys():
            logger.info(f'  t:{t}')
            # abb refactor, I want timepointMetadata to be `class TimepointMetadata` (not dict)
            timepointMetadata = self.metadata.getTimepointMetadata(t)
            logger.info(f'    timepointMetadata:{type(timepointMetadata)}')
            channelKeys = timepointMetadata.getChannelKeys()
            # channelKeys = timepointMetadata['_metadataList'].keys()  # abb refactor
            logger.info(f'      channelKeys:{channelKeys}')
            for c in channelKeys:
                channelPath = f'{t}/{c}'
                imgData = group[channelPath][:]
                logger.info(f'  channelPath:{channelPath} {imgData.shape}')
                # store imgData in zarr file/folder
                self._imagesSrcs.append({c: imgData})

    def saveAs(self, path:str) -> Optional[bool]:
        """Save the zarr file to disk.
        """
        if path.endswith('.mmap'):
            logger.info('   as a DirectoryStore')
            fs = zarr.DirectoryStore(path)
        elif path.endswith('.mmap.zip'):
            fs = zarr.ZipStore(path, mode="w", compression=zipfile.ZIP_STORED)
            logger.info('   as a ZipStore')
        else:
            logger.error(f'path must end with .mmap or .mmap.zip')
            logger.error(f'  got {path}')
            return
        
        with fs as store:
            group: zarr.hierarchy.Group = zarr.group(store=store)

            group.attrs['metadata'] = self.metadata.asDict()
            
            # each image timepoint/session is in the root folder
            # this is compatible with ome-zarr
            # we use metadata3 to traverse
            for t in self.metadata.getTimepointKeys():
                logger.info(f't:{t}')
                timepointMetadata = self.metadata.getTimepointMetadata(t)
                channelKeys = timepointMetadata.getChannelKeys()
                for c in channelKeys:
                    # check if already in zarr
                    # do not save again (time and channel keys are immutable)
                    _channelPath = f'{t}/{c}'
                    if _channelPath in group:
                        logger.info(f'    already in file skipping {_channelPath}')
                        continue
                    imgData = self.getChannelData(t, c)
                    logger.info(f"    saving {_channelPath} -> {imgData.shape}")
                    # store imgData in zarr file/folder
                    group[f'{t}/{c}'] = imgData

        logger.info(f'zarr file saved to {path}')

        return True
