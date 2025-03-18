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

from mapmanagercore.metadata._metadata import _metadataBase, _metadataList
from mapmanagercore.utils import getAutoContrast  # given img data, get min/max
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
    def shape(self):
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
            logger.mmlog('expecting img data of shape len 2 or 3, got {proposedShape}')
            return
        # self.dtype = imgData.dtype.name

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
            # logger.info('TimepointMetadata -->> converting dict to ChannelMetadata')
            _metadataList = copy(self._metadataList)
            # logger.info(f'original self._metadataList:')
            # pprint(self._metadataList)
            self._metadataList = {}
            for k,v in _metadataList.items():
                # channelMetadata = ChannelMetadata(v)
                channelMetadata = ChannelMetadata.from_dict(v)
                newChannelIndex = self.appendMetadataItem(channelMetadata)

            # logger.info('after:')
            # pprint(self._metadataList)

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
            Will fail if appending channel index > 0 and
            proposed channel imgData.shape does not match channel index 0 shape.
        """
        
        proposedShape = imgData.shape
        if self.numChannels == 0:
            # always append channel 0,
            # members like self.xPixels will never change
            self.shapeMetadata._initFromImgData(imgData)

        else:
            # check proposedShape
            if proposedShape != self.shape:
                logger.mmlog(f'expecting shape {self.shape} but got {proposedShape}')
                return
                
        metadataContrast = ChannelMetadata(name=name)
        metadataContrast._initFromImgData(imgData)
        
        # do the append
        newChannelIndex = self.appendMetadataItem(metadataContrast)
        return newChannelIndex
    
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

    def __getitem__(self, index:int) -> ChannelMetadata:
        return super().__getitem__(index)
    
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

    def setChannelProperty(self, channelIdx, key, value):
        """Set channel property.
        """
        if channelIdx not in self.listIndices:
            return
        return self._metadataList[channelIdx].setValue(key, value)

    def getChannelProperty(self, channelIdx, key):
        """Get channel property.
        """
        if channelIdx not in self.listIndices:
            return
        return self._metadataList[channelIdx].getValue(key)
    
    @property
    def channelIndices(self) -> List[int]:
        """Get the list[int] of channel indices.
        """
        return self.listIndices

    def getChannelKeys(self) -> List[int]:
        """Get the list of channel keys.
        """
        return self.channelIndices
    
@dataclass_json
@dataclasses.dataclass
class mmMapMetadata(_metadataList):
    """A list of TimepointMetadata.
    
    For single timepoint mmap/zarr, represents a list of independent imaging timepoints/sessions.
    """

    def __post_init__(self):
        if isinstance(self._metadataList, dict):
            # logger.info('mmMapMetadata -->> converting dict to TimepointMetadata')
            _metadataList = copy(self._metadataList)
            # logger.info('before:')
            # pprint(self._metadataList)
            self._metadataList = {}
            for k,v in _metadataList.items():
                # timepointMetadata = TimepointMetadata(v)
                timepointMetadata = TimepointMetadata.from_dict(v)
                self.appendTimepoint(timepointMetadata)
            # logger.info('mmMapMetadata after self._metadataList:')
            # pprint(self._metadataList)

    def print(self):
        """Print summary of metadata.
        """
        for timepoint in self.timepointIndices:
            print(f'timepoint {timepoint}')
            for channel in self[timepoint].channelIndices:
                print(f'  channel {channel}')
                print(f'    minInt {self[timepoint][channel].minInt}')
                print(f'    maxInt {self[timepoint][channel].maxInt}')

    def getTimepointMetadata(self, index : int) -> Optional[TimepointMetadata]:
        """Get metadata for a timepoint.
        
        Args:
            index: Timepoint index to get.
        
        See Also:
            __getitem__
            
        Returns:
            Metadata for timepoint at index.
        """
        _metadata = self.getMetadataItem(index)
        if _metadata is None:
            logger.error(f'did not find timepoint {index}, available timepoints are {self.timepointIndices}')
        return _metadata
    
        # _item = self.getMetadataItem(index)
        # logger.info(f'_item:{type(_item)}')
        # return TimepointMetadata.from_dict(_item)

    def insertTimepoint(self, index : int, metadata : TimepointMetadata) -> bool:
        """Insert a new timepoint.

        Args:
            index: The index to insert into.
            metadata: The TimepointMetadata to insert.

        Returns:
            True on success, otherwise False.
        """
        if index not in self.listIndices:
            logger.mmlog(f'src {index} does not exist, available indices are {self.listIndices}')
            return False
        self.insertMetadataItem(index, metadata)
        return True
    
    def appendTimepoint(self, timepointMetadata : TimepointMetadata):
        """Append a new timepoint.

        Args:
            timepointMetadata: The timepoint to append.
        """
        self.appendMetadataItem(timepointMetadata)

    def deleteTimepoint(self, index : int) -> Optional[TimepointMetadata]:
        if index in self.listIndices:
            return self.deleteMetadataItem(index)

    def swapTimepoint(self, srcTimepoint : int, dstTimepoint : int) -> bool:
        """Move/swap timepoints.
        
        Returns:
            True on success, otherwise False
        """
        ok = self.swapMetadataItems(srcTimepoint, dstTimepoint)
        return ok

    def moveChannel(self, srcTimepoint:int, srcChannelIdx:int,
                    dstTimepoint:int, dstChannelIdx:int) -> bool:
        """Move a channel from one timepoint to another.
        
        Args:
            srcTimepoint: Source timepoint to move from.
            srcChannelIdx: Source channel to move from.
            dstTimepoint: Destination timepoint to move to.
            dstChannelIdx: Destination channel to move to.

        Returns:
            True on success, otherwise false.
        """

        # check source timepoint and channel
        if srcTimepoint not in self.timepointIndices:
            return False
        if srcChannelIdx not in self[srcTimepoint].channelIndices:
            return False
        # check dst timepoint and channel
        if dstTimepoint not in self.timepointIndices:
            return False
        # hold off on dst channel check,
        # if within range then insert, otherwise ignore and always append
        # if dstChannel not in self[srcTimepoint].channelIndices:
        #     return False
        
        # remove from source
        _removedChannelMetadata = self[srcTimepoint].deleteChannel(srcChannelIdx)

        # insert into destinations
        _newIndex = self[dstTimepoint].appendMetadataItem(_removedChannelMetadata)  # to do write 

        return True
    
    @property
    def numTimepoints(self) -> int:
        """The number of timepoints.
        
        Returns:
            The number of timepoints.
        """
        return self.numItems

    @property
    def timepointIndices(self) -> List[int]:
        """Get the list[int] of timepoint indices.
        """
        return self.listIndices

    def getTimepointKeys(self) -> List[int]:
        """Get the list of timepoint keys.
        """
        return self.timepointIndices
    
    def __getitem__(self, index:int) -> TimepointMetadata:
        return super().__getitem__(index)

    def asDict(self):
        return dataclasses.asdict(self)