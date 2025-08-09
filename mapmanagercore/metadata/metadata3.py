"""Metadata stores information about images.

This includes:

 - A list of single timepoint images
 - Each single timepoint image can have multiple (color) channels and contains metadata about the images

 Each `single timepoint image` can be thought of as an `imaging session`.
 The user sits down at the scope and takes an image, we call this a timepoint.
 Alternatively, this could be called an `imaging session`.

Classes:
    mmMapMetadata: A list of `TimepointMetaData` representing a collection of single time point images.
    TimepointMetadata: Metadata for a single timepoint with (potentially) multiple color channels.

Each TimePointMetadata contains:
  - ExperimentMetadata: Metadata for each image acquired.
  - AnalysisParams: Analysis parameters for a single timepoint (also for all timepoints in a mmap).
    - VoxelsMetadata: The metadata for the physical size of a voxel.
    - ShapeMetadata: The shape of the image data in pixels.
    - ChannelMetadata: The metadata for one color channel.
     
Example:

    # make some image data
    imgData = np.random.randint(low=0, high=2**11, size=(20,512,512), dtype=np.uint16)
    
    # create a timepoint and append a channel of image data.
    tpmd = TimepointMetadata()
    tpmd.appendChannel(imgData)

    # append the timepoint to a list of timepoints
    mapmd = mmMapMetadata()
    mapmd.appendTimepoint(tpmd)

    print(mapmd.numTimepoint)  # -> 1
"""

from typing import Tuple, List, Optional, Literal, Self, Any
from copy import copy
from pprint import pprint

import dataclasses
from dataclasses import field, fields
from dataclasses_json import dataclass_json

import numpy as np

from mapmanagercore.metadata._metadata3 import _metadataBase, _metadataList
from mapmanagercore.metadata import AnalysisParams
from mapmanagercore.utils import getAutoContrast  # given img data, get min/max
from mapmanagercore.exceptions import MetadataError
from mapmanagercore.logger import logger

# TODO: move to its own file
@dataclasses.dataclass
class ExperimentMetadata(_metadataBase):
    """For each image acquired, user can specify `Experiment` metadata.
    
    This is designed to capture the experimental conditions of an image such as species, sex, and age.
    
    This spans all color channels.
    """
    Species: str = ''
    Sex: str = ''
    Age: str = ''
    Region: str = ''
    CellType: str = ''
    AcqDate: str = ''
    AcqTime: str = ''
    Condition1: str = ''
    Condition2: str = ''
    Timer1: str = ''
    Timer2: str = ''

@dataclass_json
@dataclasses.dataclass
class ChannelMetadata(_metadataBase):
    """Metadata for one color channel.
    
    When we import,
        We have the full image volume and store (minInt, maxInt, dtype).
    """
        
    minInt: int = 1
    """Min intensity of image data, set on import (see _initFromImgData) then immutable."""
    maxInt: int = 1
    """Max intensity of image data, set on import then immutable."""
    dtype: str = 'Unknown'
    """String representation of dtype"""
    minAutoContrast: int = 0
    """Min auto contrast from all imgData, set on import then imutable."""
    maxAutoContrast: int = 256
    """Max auto contrast from all imgData, set on import then imutable."""

    #
    # the remaining fields can be set by the user.
    minUserContrast: int = 1
    maxUserContrast: int = 1
    
    color : str = "green"  # map -> 'green'
    """Color LUT for the image."""

    name: str = 'Untitled'
    """Name of the channel."""

    # abj: boolean to enable auto recalculation
    channelActivated: bool = True

    def _initFromImgData(self, imgData : np.ndarray):
        """On import of an image.

        This assumes we have the entire image volume.
        """
        self.minInt = int(np.min(imgData))  # immutable
        self.maxInt = int(np.max(imgData))
        
        minContrast, maxContrast = getAutoContrast(imgData)
        # new 20250415
        self.minAutoContrast = minContrast  # immutable
        self.maxAutoContrast = maxContrast

        # set by user
        self.minUserContrast = minContrast
        self.maxUserContrast = maxContrast

        self.dtype = imgData.dtype.name

    def resetAutoContrast(self):
        """Reset to auto contrast (calculated once on import).
        """
        self.minUserContrast = self.minAutoContrast
        self.maxUserContrast = self.maxAutoContrast

    def getUserContrast(self) -> Tuple[int, int]:
        """Get min/max of current user contrast.
        """
        return [self.minUserContrast, self.maxUserContrast]
    
    def setUserContrast(self, theMin, theMax):
        self.minUserContrast = theMin
        self.maxUserContrast = theMax

@dataclass_json
@dataclasses.dataclass
class VoxelMetadata(_metadataBase):
    """The metadata for the physical size of a voxel.

    Args:
        xVoxel: The x-coordinate's voxel size.
        yVoxel: The y-coordinate's voxel size.
        zVoxel: The z-coordinate's voxel size.
    """
    xVoxel: float = 1.0
    yVoxel: float = 1.0
    zVoxel: float = 1.0
    unit: Literal["µm"] = "micrometer"  # abb from µm

@dataclass_json
@dataclasses.dataclass
class ShapeMetadata(_metadataBase):
    """The shape of the image data in pixels.
    
    Set on import and never changes.
    """
    xPixels: int = 1
    yPixels: int = 1
    zPixels: int = 1

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get shape as (z, y, x).
        """
        return (self.zPixels, self.yPixels, self.xPixels)
    
    def _initFromImgData(self, imgData : np.ndarray):
        """Set shape on import.
        
        After import, never changes.
        """
        proposedShape = imgData.shape
        if len(proposedShape)==2:
            # single plane image
            self.xPixels = proposedShape[1]
            self.yPixels = proposedShape[0]
            self.zPixels = 1
        elif len(proposedShape)==3:
            self.zPixels = proposedShape[0]
            self.xPixels = proposedShape[2]
            self.yPixels = proposedShape[1]
        else:
            _err = 'expecting img data of shape len 2 or 3, got {proposedShape}'
            logger.mmlog(_err)
            raise MetadataError(_err)

@dataclass_json
@dataclasses.dataclass
class TimepointMetadata(_metadataList):
    """Metadata for one timepoint.
    
    In our single timepoint analysis, each timepoint is independent of all others.

    Args:
        name: User defined name of the timepoint.

    TODO:
        When we are in a multi-timepoint map (with segments and spine connected)
        we need one global AnalysisParameters (At root of zarr file)
    """

    channels: dict = dataclasses.field(default_factory=dict)
    _key = 'channels'

    name: str = 'Untitled'

    # shapeMetadata: ShapeMetadata = dataclasses.field(default_factory=lambda: ShapeMetadata())
    shapeMetadata: ShapeMetadata = dataclasses.field(default_factory=ShapeMetadata)
    """The shape of the raw image data (once loaded with raw data, this is never changed)."""

    voxelMetadata: VoxelMetadata = dataclasses.field(default_factory=lambda: VoxelMetadata())
    """The physical unit voxel size in each dimension."""

    experimentMetadata : ExperimentMetadata = dataclasses.field(default_factory=lambda: ExperimentMetadata())
    """Experimental metadata corresponding to the conditions an image was aquired."""

    analysisParameters: AnalysisParams = dataclasses.field(default_factory=lambda: AnalysisParams())
    """The analysis parameters for this single timepoint (connected maps use a global version of this)."""
    
    def convertUnits(self, voxelValue, paramKey):
        """ convert from real units to pixel units

        Args:
            voxelValue: voxel value of parameter to be convertedd to pixel value
            paramKey: string name of the analysis parameter
        """
        # we can only convert values to pixel if units is 'um'
        # pull allowPixels bool from our metadata
        # unitStr = 'um'  # TODO write the code to get actual unit value
        allowPixels = True  # False:
        if not allowPixels:
            logger.error(f'Analysis Parameter key "{key}" cannot be converted to pixels (units are "{unitStr}")')
            return
        
        if self.voxelMetadata is not None:
            
            # problem is voxelValue is a single value (physical magnitude) and not in x or y 
            pixelValueX = voxelValue / self.voxelMetadata.xVoxel
            pixelValueY = voxelValue / self.voxelMetadata.yVoxel
        
            # assuming that all dimensions are the same
            if self.voxelMetadata.xVoxel == self.voxelMetadata.yVoxel:
                return pixelValueX

            # Main Problem! for accurate measurements in 2D, a direction must be known
            # however these are general values that are physical units
            
            elif self.voxelMetadata.xVoxel != self.voxelMetadata.yVoxel:
                # p2 = (point1[0] + pixelValueX, point1[1] + pixelValueY)
                # # dx, dy = (point1 - p2)
                # dx, dy = (p2 - point1)

                # # Take the euclidean distance 
                # # pixelDistance = np.sqrt((dx * pixelValueX)**2 + (dy * pixelValueY)**2)
                # pixelDistance = np.sqrt((dx)**2 + (dy)**2)

                # # approximating with geometric mean
                # import math
                # effective_pixel_size = math.sqrt(pixelValueX * pixelValueY)
                # pixel_distance = voxelValue / effective_pixel_size

                # Elliptical Radius
                # pixelDistance = np.sqrt((dx * pixelValueX)**2 + (dy * pixelValueY)**2)
                # pixelDistance = np.sqrt((dx)**2 + (dy)**2)
                pixelDistance = np.sqrt((pixelValueX)**2 + (pixelValueY)**2)
            
                return pixelDistance
            else:
                return pixelValueX
    
        else:
            logger.error(f"No physical voxel units established")

    # DEFUNCT abb 20250519
    # def getValue_pixel(self, key:str) -> int:
    #     """Get an analysis parameter value converted from um -> pixels.
    #     """
    #     # we can only convert values to pixel if units is 'um'
    #     # pull allowPixels bool from our metadata
    #     # unitStr = 'um'  # TODO write the code to get actual unit value
    #     allowPixels = True  # False:
    #     if not allowPixels:
    #         logger.error(f'Analysis Parameter key "{key}" cannot be converted to pixels (units are "{unitStr}")')
    #         return
        
    #     # assuming xVoxel and yVoxel are the same
    #     xVoxel = self.voxelMetadata.xVoxel  # um/pixel
    #     valueInUm = self.analysisParameters.getValue(key)
    #     valueInPixels = valueInUm / xVoxel
    #     return valueInPixels
    
    def __post_init__(self):
        if isinstance(self._metadataList, dict):
            _metadataList = copy(self._metadataList)
            self._metadataList = {}

            for k,v in _metadataList.items():
                # channelMetadata = ChannelMetadata(v)
                channelMetadata = ChannelMetadata.from_dict(v)
                newChannelIndex = self.appendMetadataItem(channelMetadata)
    
    def fromImgData(imgData: np.ndarray) -> Self:
        tpmd = TimepointMetadata()
        tpmd.appendChannel(imgData)
        return tpmd

    def setVoxelMetadata(self, voxelMetadata: VoxelMetadata):
        self.voxelMetadata = voxelMetadata
    
    def appendChannel(self,
                      imgData : np.ndarray,
                      name: Optional[str] = 'Untitled',
                      activateChannel: bool = False
                      ) -> Optional[int]:
        """Given img data, append a new color channel.
            Used when we are importing data.
        
        Parameters:
            imgData: The channel image data to append.
            name: The name of the channel. Default is 'Untitled'.

        Returns:
            Appended channel index on success, otherwise None.

        Notes:
            Will fail if appending channel > 1 and
            proposed channel imgData.shape does not match channel index 1 shape.
        """
        
        proposedShape = imgData.shape
        if self.numChannels == 0:
            # always append channel 0,
            # members like self.xPixels will never change
            self.shapeMetadata._initFromImgData(imgData)

        else:
            # check proposedShape
            if proposedShape != self.shape:
                _err = f'expecting shape {self.shape} but got {proposedShape}'
                # logger.error(_err)
                raise MetadataError(_err)
        
        # abj: when user imports channel they can set whether or not they want it to be "activated"
        # which means to auto compute aggregate columns. Reminder to Pass in that boolean
        # metadataContrast = ChannelMetadata(name=name)
        metadataContrast = ChannelMetadata(name=name, channelActivated=True)
        metadataContrast._initFromImgData(imgData)

        # do the append
        newChannelKey = self.appendMetadataItem(metadataContrast)
        logger.info(f'setting hard coded color using newChannelKey:{newChannelKey}')

        # abj: testing more than 3 indexes
        # colors = ['white', 'red', 'green', 'blue', 'magenta', 'orange']
        # metadataContrast.color = colors[newChannelKey]

        if newChannelKey==1:
            metadataContrast.color = 'red'
        elif newChannelKey == 2:
            metadataContrast.color = 'green'
        elif newChannelKey == 3:
            metadataContrast.color = 'blue'

        return newChannelKey
    
    def deleteChannel(self, channelIdx:int) -> Optional[ChannelMetadata]:
        """Delete a color channel.
        
        Returns:
            On success, deleted ChannelMetadata, otherwise None
        """
        if channelIdx == 0:
            logger.mmlog('cannot delete channel 0')
            return
        
        metadataContrast = self.deleteMetadataItem(channelIdx)
        return metadataContrast
    
    def swapChannels(self, srcChannelIdx, dstChannelIdx) -> bool:
        """Move/swap color channel.
        """
        # abj: not swappining actual images here
        ok = self.swapMetadataItems(srcChannelIdx, dstChannelIdx)
        return ok
    
    def getChannelMetadata(self, channelIdx) -> Optional[ChannelMetadata]:
        """Get metadata for a channel.
        """
        return self.getMetadataItem(channelIdx)

    # def __getitem__(self, index:int) -> ChannelMetadata:
    #     return super().__getitem__(index)
    
    @property
    def numChannels(self) -> int:
        """The number of channels in the timepoint.
        """
        return self.numItems
    
    @property
    def numSlices(self) -> int:
        """The number of image slices in the timepoint.

        All channels are the same
        """
        return self.shape[0]
    
    @property
    def shape(self) -> Tuple[int,int,int]:
        """The shape of the image data in the timepoint.
        
        Notes
        -----
        All channels in a timepoint have the same shape.
        """
        return self.shapeMetadata.shape

    @property
    def shapeDict(self) -> dict:
        """Get shape as a dict.
        """
        return {
            'z': self.shape[0],
            'y': self.shape[1],
            'x': self.shape[2],
        }
        
    def setChannelProperty(self, channelIdx, key, value) -> MetadataError | object:
        """Set channel property.
        """
        if not self.channelExists(channelIdx):
            _err = f'channel {channelIdx} does not exist, expecting one of {self.channelKeys}'
            raise MetadataError(_err)
        
        logger.info(f"setChannelProperty channelIdx {channelIdx} key {key} value {value}")
        
        return self._metadataList[channelIdx].setValue(key, value)

    def getChannelProperty(self, channelIdx, key) -> MetadataError | object:
        """Get channel property.
        """
        if not self.channelExists(channelIdx):
            _err = f'channel {channelIdx} does not exist, expecting one of {self.channelKeys}'
            raise MetadataError(_err)

        return self._metadataList[channelIdx].getValue(key)
    
    # abj
    def getAllChannelProperty(self, key) -> List:
        """ Get a list of all values within Channel dict of a given key

        Example use case: get all activated channels
        """
        # for channel in self.channelKeys:
        allChannelProperty = []
        for channel in self.channelKeys:
            channelProperty = self.getChannelProperty(channel, key)
            allChannelProperty.append(channelProperty)

        return allChannelProperty
    
    def getActivatedChannels(self) -> List:
        """  get all activated channels

        Example use case: get all activated channels
        """
        # for channel in self.channelKeys:
        # self.getAllChannelProperty()
        activatedChannels = []
        for channel in self.channelKeys:
            channelActivated = self.getChannelProperty(channel, "channelActivated")
            # logger.info(f"channelActivated {channelActivated}")
            if channelActivated:
                activatedChannels.append(channel)
                channelActivated = False # reset variable

        # logger.info(f"activatedChannels {activatedChannels}")
        return activatedChannels
    
    # def activateChannel(self, channelIdx, key, value):
    #     self.setChannelProperty(channelIdx, key)

    def getInActiveChannels(self) -> List:
        """  get all inactive channels
        """
        inActiveChannels = []
        for channel in self.channelKeys:
            channelActivated = self.getChannelProperty(channel, "channelActivated")
            if not channelActivated:
                inActiveChannels.append(channel)
                channelActivated = True # reset variable

        logger.info(f"inActiveChannels are: {inActiveChannels}")
        return inActiveChannels
    
    def getChannelNames(self) -> dict:
        """ return dictionary of channel names where 
        key: channel number and value = channel name
        """
        channelNameDict = {}
        for channelIndex in self.channelKeys:
            channelName= self.getChannelProperty(channelIndex, "name")
            channelNameDict[channelIndex] = channelName

        logger.info(f"channelNameDict are: {channelNameDict}")
        return channelNameDict
        
    @property
    def channelKeys(self) -> List[int]:
        """Get the list[int] of channel keys.
        """
        return self.listKeys
    
    def channelExists(self, channelIdx) -> bool:
        return self.keyExists(channelIdx)
        # return channelIdx in self.channelKeys
    
@dataclass_json
@dataclasses.dataclass
class mmMapMetadata(_metadataList):
    """A list of TimepointMetadata.
    
    For single timepoint mmap/zarr,
    represents a list of independent imaging timepoints/sessions.
    """

    # TODO: change this to unlimited number
    _maxChannel = 3

    timepoints: dict = dataclasses.field(default_factory=dict)
    _key = 'timepoints'

    @property
    def possibleChannelKeys(self) -> List[int]:
        return list(range(1, self._maxChannel+1))
    
    def __post_init__(self):
        if isinstance(self._metadataList, dict):
            _metadataList = copy(self._metadataList)
            self._metadataList = {}
            for k,v in _metadataList.items():
                # timepointMetadata = TimepointMetadata(v)
                timepointMetadata = TimepointMetadata.from_dict(v)
                # appending just metadata, not imgData
                self.appendTimepoint(None, timepointMetadata=timepointMetadata)

    def print(self):
        """Print summary of metadata.
        """
        for timepoint in self.timepointKeys:
            print(f't:{timepoint}')
            for channel in self.getTimepoint(timepoint).channelKeys:
                print(f'  c:{channel} {self.getTimepoint(timepoint).shape}')
                # print(f'    minInt {self[timepoint][channel].minInt}')
                # print(f'    maxInt {self[timepoint][channel].maxInt}')

    def getTimepoint(self, index : int) -> MetadataError | TimepointMetadata:
        """Get metadata for a timepoint.
        
        Args:
            index: Timepoint index to get.
        
        Returns:
            Metadata for timepoint at index.

        Raises:
            MetadataError if timepoint does not exist.
        """
        if not self.timepointExists(index):
            _err = f'did not find timepoint {index}, available timepoints are {self.timepointKeys}'
            raise MetadataError(_err)
        
        _metadata = self.getMetadataItem(index)
        return _metadata
    
    def appendTimepoint(self, 
                        imgData : np.ndarray,
                        timepointMetadata:TimepointMetadata = None) -> int:
        """Append a new timepoint.

        Parameters
        ==========
            imgData: imgData to seed the timepoint with (first channel)
        
        Returns
        =======
            New timepoint key
        """

        if timepointMetadata is None:
            timepointMetadata = TimepointMetadata()  # empty
            timepointMetadata.appendChannel(imgData)  # append first channel from data
        
        return self.appendMetadataItem(timepointMetadata)

    def deleteTimepoint(self, index : int) -> MetadataError | TimepointMetadata:
        if not self.timepointExists(index):
            raise MetadataError(f'timepoint {index} does not exist, expecting one of {self.timepointKeys}')
        return self.deleteMetadataItem(index)

    def swapTimepoint(self, srcTimepoint : int, dstTimepoint : int) -> Optional[MetadataError]:
        """Swap position of timepoints.
        
        Raises:
            MetadataError: If src or dst timepoint does not exist
        """
        if not self.timepointExists(srcTimepoint):
            raise MetadataError(f'srcTimepoint {srcTimepoint} does not exist, expecting one of {self.timepointKeys}')
        if not self.timepointExists(dstTimepoint):
            raise MetadataError(f'dstTimepoint {dstTimepoint} does not exist, expecting one of {self.timepointKeys}')

        self.swapMetadataItems(srcTimepoint, dstTimepoint)

    def channelExists(self, timepoint:int, channel:int) -> MetadataError | bool:
        # check timepoint 
        if not self.timepointExists(timepoint):
            raise MetadataError(f'timepoint {timepoint} does not exist, expecting one of {self.timepointKeys}')

        # check channel
        if not self.getTimepoint(timepoint).channelExists(channel):
            raise MetadataError(f'timepoint {timepoint} srcChannelIdx:{channel} does not exist, expecting one of {self.getTimepoint(timepoint).channelKeys}')
  
        return True

    def moveChannel(self,
                    timepoint:int,
                    srcChannelIdx:int,
                    dstChannelIdx:int) -> MetadataError:
        """Move a channel within a timepoint.
        
        Args:
            timepoint: Source timepoint to move from.
            srcChannelIdx: Source channel to move from.
            dstChannelIdx: Destination channel to move to.

        Raises:
            MetadataError: If either timepoint or src or dst channels do not exist
        """

        # check timepoint 
        if not self.timepointExists(timepoint):
            raise MetadataError(f'timepoint {timepoint} does not exist, expecting one of {self.timepointKeys}')

        # check channel
        if not self.getTimepoint(timepoint).channelExists(srcChannelIdx):
            raise MetadataError(f'timepoint {timepoint} srcChannelIdx:{srcChannelIdx} does not exist, expecting one of {self.getTimepoint(timepoint).channelKeys}')
        if not self.getTimepoint(timepoint).channelExists(dstChannelIdx):
            raise MetadataError(f'timepoint {timepoint} dstChannelIdx:{dstChannelIdx} does not exist, expecting one of {self.getTimepoint(timepoint).channelKeys}')
        
        self.getTimepoint(timepoint).swapChannels(srcChannelIdx, dstChannelIdx)
    
    @property
    def numTimepoints(self) -> int:
        """The number of timepoints.
        
        Returns:
            The number of timepoints.
        """
        return self.numItems

    @property
    def timepointKeys(self) -> List[int]:
        """Get the list[int] of timepoint keys.
        """
        return self.listKeys

    def timepointExists(self, t:int) -> bool:
        return self.keyExists(t)
        # return t in self.timepointKeys
    
    # def __getitem__(self, index:int) -> TimepointMetadata:
    #     return super().__getitem__(index)

