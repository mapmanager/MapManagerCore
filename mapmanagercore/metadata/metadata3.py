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

from typing import Tuple, List, Optional, Literal, Self
from copy import copy
from pprint import pprint

import dataclasses
from dataclasses_json import dataclass_json

import numpy as np

from mapmanagercore.metadata._metadata3 import _metadataBase, _metadataList
from mapmanagercore.utils import getAutoContrast  # given img data, get min/max
from mapmanagercore.exceptions import MetadataError
from mapmanagercore.logger import logger

@dataclass_json
@dataclasses.dataclass
class AnalysisParams(_metadataBase):
    """Analysis parameters for a single timepoint (also for all timepoints in a mmap).
    """
    version:float = 0.6
    """Manually increment this when we add to this class."""

    # spines
    brightestPathDistance: int = dataclasses.field(default=10, metadata={'description': 
                                                             'Points along the tracing to find spine connection (anchor).'
                                                             })
    brightestPathChannel: int = dataclasses.field(default=0, metadata={'description': 
                                                             'Image color channel to find brightest connection of spine.'
                                                             })
    brightestPathZSpread: int = dataclasses.field(default=3, metadata={'description': 
                                                             'Number of image slices for max project to find brightest connection of spine.'
                                                             })
    roiExtend: int = dataclasses.field(default=4, metadata={'description': 
                                                             'Number of pixels to extend spine head for spine ROI.'
                                                             })
    roiRadius: int = dataclasses.field(default=4, metadata={'description': 
                                                             'Width of spine ROI.'
                                                             })
    
    # segments
    segmentRadius: int = dataclasses.field(default=4, metadata={'description': 
                                                             'Radius of segment tracing.'
                                                             })
    segmentTracingMaxDistance: int = dataclasses.field(default=90, metadata={'description': 
                                                             'Max distance to trace a brightest path.'
                                                             })
    backgroundRoiGridPoints: int = dataclasses.field(default=5, metadata={'description': 
                                                             'Number of points in grid (nxn) to calculate background ROI.'
                                                             })
    backgroundRoiGridOverlap: float = dataclasses.field(default=0.1, metadata={'description': 
                                                             'Overlap of background grid points.'
                                                             })

    def getDescription(self, attribute_name:str) -> str:
        """Get the 'description' from one attribute metadata.
        """
        return self.__dataclass_fields__[attribute_name].metadata['description']

    def printFields(self):
        """debugging.
        """
        for oneField in dataclasses.fields(self):
            # print(f'oneField:{oneField}')
            
            name = oneField.name
            _type = oneField.type
 
            value = self.getValue(name)
            default = oneField.default
            try:
                metadata = oneField.metadata
            except (KeyError) as e:
                metadata = ''
            print(name, value, _type, default, metadata)
        
        # _fields = list(self.__annotations__)
        # print(_fields)


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
    #
    # the remaining fields can be set by the user.
    minContrast: int = 1
    maxContrast: int = 1
    color : str = "x00FF00"  # map -> 'green'
    """Color LUT for the image."""

    name: str = 'Untitled'
    """Name of the channel."""

    def _initFromImgData(self, imgData : np.ndarray):
        """On import of an image.

        This assumes we have the entire image volume.
        """
        self.minInt = int(np.min(imgData))
        self.maxInt = int(np.max(imgData))
        minContrast, maxContrast = getAutoContrast(imgData)
        self.minContrast = minContrast
        self.maxContrast = maxContrast

        self.dtype = imgData.dtype.name

@dataclass_json
@dataclasses.dataclass
class VoxelMetadata:
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
class ShapeMetadata:
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
                      name: Optional[str] = 'Untitled') -> Optional[int]:
        """Given img data, append a new color channel. Used when we are importing data.
        
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
                
        metadataContrast = ChannelMetadata(name=name)
        metadataContrast._initFromImgData(imgData)
        
        # do the append
        newChannelKey = self.appendMetadataItem(metadataContrast)
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
    def shape(self) -> Tuple[int,int,int]:
        """The shape of the image data in the timepoint.
        
        Notes
        -----
        All channels in a timepoint have the same shape.
        """
        return self.shapeMetadata.shape

    def setChannelProperty(self, channelIdx, key, value) -> MetadataError | object:
        """Set channel property.
        """
        if not self.channelExists(channelIdx):
            _err = f'channel {channelIdx} does not exist, expecting one of {self.channelKeys}'
            raise MetadataError(_err)
        
        return self._metadataList[channelIdx].setValue(key, value)

    def getChannelProperty(self, channelIdx, key) -> MetadataError | object:
        """Get channel property.
        """
        if not self.channelExists(channelIdx):
            _err = f'channel {channelIdx} does not exist, expecting one of {self.channelKeys}'
            raise MetadataError(_err)

        return self._metadataList[channelIdx].getValue(key)
    
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
    
    For single timepoint mmap/zarr, represents a list of independent imaging timepoints/sessions.
    """

    timepoints: dict = dataclasses.field(default_factory=dict)
    _key = 'timepoints'

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

