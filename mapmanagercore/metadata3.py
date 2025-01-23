import dataclasses
from typing import Tuple, List, Optional, Literal

import numpy as np

from mapmanagercore.utils import getAutoContrast  # given img data, get min/max
from mapmanagercore.logger import logger

@dataclasses.dataclass
class _metadataBase:
    """Abstract base class for all metadata dataclass.
    """
    def setValue(self, key, value) -> bool:
        """Set a field value.
        
        Returns
        =======
        True if field exists, otherwise False
        """
        try:
            setattr(self, key, value)
            return True
        except (AttributeError) as e:
            logger.warning(e)
            return False
        
    def getValue(self, key) -> Optional[object]:
        """Set a field value.
        
        Returns
        =======
        Field value if field exists, otherwise None

        Notes
        =====
        Returning python None can be misleading, make sure "none" of our fields have default value of None ???
        """
        if not isinstance(key, str):
            logger.mmLog(f'attr must be str, got {type(key)}')
            return
        
        try:
            return getattr(self, key)
        except (AttributeError) as e:
            logger.warning(e)

    def asDict(self) -> dict:
        """Get the dataclass as a python dict.
        """
        return dataclasses.asdict(self)
    
@dataclasses.dataclass
class _metadataList(_metadataBase):
    """Abstract base class for all metadata that holds a list.
    
    This currently includes:
        - TimepointMetadata that has a list of ChannelMetadata (channels)
        - MetadataList that has a list of TimepointMetadata
    """

    metadataList : list = dataclasses.field(default_factory=lambda: [])
    # or? not sure which to use
    # metadataList: List = field(default_factory=list)

    @property
    def listIndices(self) -> List[int]:
        """Get a list of indices from ourmetadataList.
        """
        return list(range(self.numItems))
    
    @property
    def numItems(self) -> int:
        """Get the number of items in the list.
        """
        return len(self.metadataList)
    
    def getMetadataItem(self, index : int) -> Optional[object]:
        """Get metadata for one item in the list.
        
        Returns
        =======
        item if index exists, otherwise None

        See
        ===
        __getitem__(int)
        """
        if index in self.listIndices:
            return self.metadataList[index]
        
    def appendMetadataItem(self, metadata : object) -> int:
        """Append to end of list.
        
        Returns
        =======
        index of new item (0 based).
        """
        self.metadataList.append(metadata)
        return self.numItems - 1
    
    def deleteMetadataItem(self, index : int) -> Optional[object]:
        """Remove from index.

        Returns
        =======
        The item removed, otherwise None
        """
        if index in self.listIndices:
            item = self.metadataList.pop(index)
            return item
    
    def insertMetadataItem(self, index : int, metadata : object) -> Optional[bool]:
        """Insert an item at given index.
        
        Returns
        =======
        True on success, otherwise None
        """
        if index in self.listIndices:
            self.metadataList.insert(index, metadata)
            return True

    def swapMetadataItems(self, srcIndex, dstIndex) -> bool:
        """Swap/move an item in the list.
        
        Returns
        -------
        True on success, otherwise False
        """
        if srcIndex not in self.listIndices:
            logger.mmlog(f'src {srcIndex} does not exist')
            return False
        if dstIndex not in self.listIndices:
            logger.mmlog(f'dst {dstIndex} does not exist')
            return False
        
        # the existing src/dst channels
        srcMetadata = self.getMetadataItem(srcIndex)
        dstMetadata = self.getMetadataItem(dstIndex)
        
        # simple swap !
        self.metadataList[dstIndex] = srcMetadata
        self.metadataList[srcIndex] = dstMetadata

        return True
    
    def setItem(self, index, key, value) -> bool:
        """Set one item (key/value) in list.
        """
        if index in self.listIndices:
            return self.metadataList[index].setValue(key, value)
        else:
            return False
        
    def __getitem__(self, index:int):
        """Prefer to use explicit function getMetadataItem(int)
        """
        return self.metadataList[index]
    
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

@dataclasses.dataclass
class ChannelMetadata(_metadataBase):
    """Metadata for one color channel.
    
    When we import,
        We have the full image volume and store (minInt, maxInt, dtype).
    """
        
    minInt: int = 1
    """Min intensity of image data, set on import then immutable."""
    maxInt: int = 1
    """Max intensity of image data, set on import then immutable."""
    dtype: str = 'Unknown'
    """np """
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

@dataclasses.dataclass
class VoxelMetadata:
    """The metadata for the physical size of a voxel.

    Attributes:
        x (float): The x-coordinate's voxel size.
        y (float): The y-coordinate's voxel size.
        z (float): The z-coordinate's voxel size.
    """
    xVoxel: float = 1.0
    yVoxel: float = 1.0
    zVoxel: float = 1.0
    unit: Literal["µm"] = "micrometer"  # abb from µm

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

@dataclasses.dataclass
class TimepointMetadata(_metadataList):
    """Metadata for one timepoint.
    
    In our single timepoint analysis, each timepoint is independent of all others.

    TODO:
        When we are in a multi-timepoint map (with segments and spine connect)
            we need one global AnalysisParameters (At root of zarr file)
    """

    name: str = 'Untitled'

    shapeMetadata: ShapeMetadata = dataclasses.field(default_factory=lambda: ShapeMetadata())
    """The shape of the raw data (once loaded with raw data, this is never changed)."""

    voxelMetadata: VoxelMetadata = dataclasses.field(default_factory=lambda: VoxelMetadata())
    """The physical unit voxel size in each dimension."""

    experimentMetadata : ExperimentMetadata = dataclasses.field(default_factory=lambda: ExperimentMetadata())
    """Experimental metadata corresponding to the conditions an image was aquired."""

    # TODO convert AnalysisParameter to dataclass (keep the self documentation)
    analysisParameters: AnalysisParams = dataclasses.field(default_factory=lambda: AnalysisParams())
    """The analysis parameters for this single timepoint (connected maps use a global)."""

    def appendChannel(self, imgData : np.ndarray) -> Optional[int]:
        """Given img data, append a new color channel.
        
        Used when we are importing data.
        
        Returns
        -------
        Appended channel index on success, otherwise None

        Notes
        -----
        Will fail if appending channel index > 0 and
            proposed channel shape from imgData does not match channel index 0 shape.
        """
        
        proposedShape = imgData.shape
        if self.numChannels == 0:
            # always append channel 0,
            # members like self.xPixels will never change
            self.shapeMetadata._initFromImgData(imgData)

        else:
            if self.numChannels > 0:
                # check proposedShape
                if proposedShape != self.shape:
                    logger.mmlog(f'expecting shape {self.shape} but got {proposedShape}')
                    return
                
        metadataContrast = ChannelMetadata()
        metadataContrast._initFromImgData(imgData)
        
        # do the append
        newChannelIndex = self.appendMetadataItem(metadataContrast)
        return newChannelIndex
    
    def deleteChannel(self, channelIdx:int) -> Optional[ChannelMetadata]:
        """Delete a color channel.
        """
        if channelIdx == 0:
            logger.mmlog('cannot delete channel 0')
            return
        
        metadataContrast = self.deleteMetadataItem(channelIdx)
        return metadataContrast
    
    def swapChannels(self, srcChannelIdx, dstChannelIdx):
        """Move/swap color channel.
        """
        ok = self.swapMetadataItems(srcChannelIdx, dstChannelIdx)
        return ok
    
    def getChannelMetadata(self, channelIdx) -> Optional[ChannelMetadata]:
        """Get metadata for a channel.
        """
        return self.getMetadataItem(channelIdx)
    
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
        return self.metadataList[channelIdx].setValue(key, value)

    def getChannelProperty(self, channelIdx, key):
        """Get channel property.
        """
        if channelIdx not in self.listIndices:
            return
        return self.metadataList[channelIdx].getValue(key)
    
@dataclasses.dataclass
class mmMapMetadata(_metadataList):
    """A list of TimepointMetadata.
    
    For single timepoint mmap/zarr, represents a list of independent imaging timepoints.
    """

    def getTimepointMetadata(self, index) -> Optional[TimepointMetadata]:
        """Get metadata for a timepoint.
        """
        return self.getMetadataItem(index)

    def insertTimepoint(self, index : int, metadata : TimepointMetadata):
        """Append a new timepoint.
        """
        if index not in self.listIndices:
            logger.mmlog(f'src {index} does not exist, available indices are {self.listIndices}')
            return
        self.insertMetadataItem(index, metadata)

    def appendTimepoint(self, metadata : TimepointMetadata):
        """Append a new timepoint.
        """
        self.appendMetadataItem(metadata)

    def deleteTimepoint(self, index : int) -> Optional[TimepointMetadata]:
        if index in self.listIndices:
            return self.deleteMetadataItem(index)

    def swapTimepoint(self, srcTimepoint, dstTimepoint) -> bool:
        """Move/swap timepoints.
        
        Returns
        =======
        True on success, otherwise False
        """
        ok = self.swapMetadataItems(srcTimepoint, dstTimepoint)
        return ok

    @property
    def numTimepoints(self) -> int:
        """The number of timepoints.
        """
        return self.numItems

    def asDict(self) -> dict:
        return dataclasses.asdict(self)