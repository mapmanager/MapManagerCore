from typing import Optional

import numpy as np
import tifffile  # abb depreciate and use ImageImporter
# import nd2

# import bioio_base
from mapmanagercore.imageImporter import getImageImporter

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata, MetadataContrast
from .base import ImageLoader
from typing import Iterator, List, Union
# from mapmanagercore.utils import getAutoContrast
from mapmanagercore.logger import logger

class MultiImageLoader(ImageLoader):
    """
    abb: Write two sentence docstring decribing the purpose of a MultiImageLoader
    Class for building an MultiImageLoader.
    """

    def __init__(self):
        super().__init__()
        self._imagesSrc = {}
        self._metadata = {}
        self.paths = []  # for logging only

        # abb md3
        from mapmanagercore.metadata.metadata3 import mmMapMetadata
        self._metadata3 = mmMapMetadata()

    def __str__(self):
        return f"Multi image Loader paths: {self.paths}"
    
    def metadata(self, t: int) -> Metadata:
        return self._metadata[t] if t in self._metadata else Metadata()

    # abb imageImporter
    def importNewTimepoint(self, path: str) -> Optional[int]:
        """Import image data into a new timepoint and append all image channels.
        
        Notes:
            Will fail if importer not available or file does not exist.

        Returns:
            New timepoint on success, otherwise None
        """
        
        ii = getImageImporter(path, loadImgData=True)
        if ii is None:
            return
        
        newTimepoint = self.getNewTimepointKey()

        for channelIdx in range(ii.numChannels):
            imgData = ii.getChannelData(channelIdx)
            if imgData is not None:
                self.read(imgData, time=newTimepoint, channel=channelIdx)
                
                # set physical unit (TODO put this into read() ???)
                from mapmanagercore.metadata3 import VoxelMetadata
                physicalPixelSizes = ii.physicalPixelSizes
                tp_metadata3 = self._metadata3.getTimepointMetadata(newTimepoint)
                tp_metadata3.setVoxelMetadata(voxelMetadata=VoxelMetadata(physicalPixelSizes))

        return newTimepoint
    
    # abb imageImporter
    def appendChannels(self, path:str, time:int) -> bool:
        """Append image channels to existing timepoint.
        
        Args:
            path:
                Path to image file like .tif or .nd2
            time:
                Timepoint to append to.

        Returns:
            True on success, otherwise False
            
        Notes:
            Will fail if:
                - Shape of images in path does not match existing channel shape.
                - timepoint (time) does not exist.
            Will warn if:
                - (TODO) Physical units in path does not match existing channel Physical units.
        """

        if not self.timepointExists(time):
            logger.error(f'timepoint {time} does not exist')
            return False

        ii = getImageImporter(path, loadImgData=False)
        if ii is None:
            return

        # check shape
        existingShape = self.shape(time)
        loadedShape = ii.channelShape
        if existingShape != loadedShape:
            logger.error(f'imported shape {loadedShape} does not match existing shape {existingShape}')
            return False
        
        # check physical sizes
        from mapmanagercore.metadata3 import VoxelMetadata
        loadedPhysicalSize = ii.physicalPixelSizes
        tp_metadata3 = self._metadata3.getTimepointMetadata(time)
        existingPhysicalSize = tp_metadata3.voxelMetadata
        logger.warning('TODO: warn user is physical size is different !!!')
        logger.warning(f'  existingPhysicalSize:{existingPhysicalSize}')
        logger.warning(f'  loadedPhysicalSize:{loadedPhysicalSize}')

        #
        # load the image data (ImageImporter is lazy so we can get shae etc.)
        _ = ii.loadData()

        # add imgData channels
        for importChannelIdx in range(ii.numChannels):
            channelImgData = ii.getChannelData(importChannelIdx)  # channel data to import
            newChannelKey = self.getNewChannelKey(time)  # new channel key
            self.read(channelImgData, time=time, channel=newChannelKey)

        return True
    
    # abb imageImporter
    def timepointExists(self, time: int) -> bool:
        """Return True if timepoint exists, otherwise False.
        """
        return time in self._imagesSrc
    
    # abb imageImporter
    def getNewTimepointKey(self) -> int:
        # how do we safely get a new timepoint key? They are just int
        keyList = list(self._imagesSrc.keys())
        if len(keyList) == 0:
            timepoint = 0  # timepoints are 1 based
        else:
            timepoint = max(keyList) + 1
        return timepoint
    
    # abb imageImporter
    def getNewChannelKey(self, time:int) -> int:
        keyList = self.channels(time)
        if len(keyList) == 0:
            channelIdx = 0  # timepoints are 1 based
        else:
            channelIdx = max(keyList) + 1
        return channelIdx

    def read(self, path: Union[str, np.ndarray], time: int = 0, channel: int = 0, name=None):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): Either the path to the image file or a np array.
          time (int): The time index.
          channel (int): The channel index.
        """
        
        # abb if path is np.ndarray (WE NEED UNIT TESTS FOR THIS)
        if name is None:
            # name = path
            name = 'Untitled'

        if isinstance(path, str):
            if path.endswith('.tif'):
                imgData = tifffile.imread(path)
            elif path.endswith('.nd2'):
                imgData = nd2.imread(path)
        else:
            # abb assuming np.ndarray?
            imgData = path

        # abb not needed, handled in ImageImporter()
        # if len(imgData.shape) == 2:
        #     # abb handle 2d, convert 2d image to 3d with one slice
        #     imgData = imgData.reshape((1, imgData.shape[1], imgData.shape[0]))

        if time not in self._imagesSrc:
            # appending first channel to new timepoint
            logger.info(f'appending new timepoint {time} with imgData.shape {imgData.shape}')

            self._imagesSrc[time] = {}
            
            # v1 meta
            _metaData = Metadata()
            # shape of imgData
            _numDims = len(imgData.shape)
            
            _metaData.voxel.x = imgData.shape[2]
            _metaData.voxel.y = imgData.shape[1]
            _metaData.voxel.z = imgData.shape[0]

            # physicaal units (um)
            _metaData.physicalSize.x = 0.15
            _metaData.physicalSize.y = 0.15
            if _numDims == 3:
                _metaData.physicalSize.z = 1

            # abb contrast
            # logger.info(f'abb MetadataContrast channel:{channel} {type(channel)} imgData:{imgData.shape}')
            metadataContrast = MetadataContrast()
            metadataContrast._initFromImgData(imgData)
            _metaData.addColorChannel(metadataContrast)

            self._metadata[time] = _metaData

            # abb md3
            from mapmanagercore.metadata.metadata3 import TimepointMetadata
            timepointMetadata = TimepointMetadata()  # empty, no channels
            timepointMetadata.appendChannel(imgData)
            self._metadata3.appendTimepoint(timepointMetadata)
        else:
            # abb md3, appending a channel to existing timepoint
            logger.info(f'_metadata3: appending new channel to existsting timepoint {time} with imgData.shape:{imgData.shape}')
            timepointMetadata = self._metadata3.getTimepointMetadata(time)
            timepointMetadata.appendChannel(imgData)

        # abb do we really need max channels? We do need current number of channels (during runtime)?
        if channel > self.maxChannels():
            self.setMaxChannels(channel)

        self._imagesSrc[time][channel] = imgData
        self._metadata[time].channelNames[channel] = name
        self.paths.append([time, channel, path])

    # abb TODO: readMetadata, this is never called?
    def readMetadata(self, metadata: Union[Metadata, str], time: int = 0):
        """
        Set the metadata for the given time index.

        Args:
          time (int): The time index.
          metadata (Metadata): The metadata.
        """

        if isinstance(metadata, str):
            with open(metadata, "r") as metadataFile:
                metadata = Metadata.from_json(metadataFile)

        if time in self._metadata:
            oldMetadata = self._metadata[time]
            for channel in oldMetadata.channelNames:
                if channel not in metadata.channelNames:
                    metadata.channelNames[channel] = oldMetadata.channelNames[channel]

        self._metadata[time] = metadata

    # abj 
    def deleteChannel(self, time, channel) -> bool:
        """ delete channel, remove from multImageLoader
        
        Args:
          time (int): The time index.
          channel (int): The channel index.
        """
        if time >= len(self._metadata):
            return False

        a = self._metadata[time].channelNames.pop(channel, None)
        b = self._imagesSrc[time].pop(channel, None)

        logger.info(f"self._imagesSrc {self._imagesSrc}")

        return a != None or b != None

    # abj
    def moveChannel(self, srcTimePoint: int, srcChannel: int, destTimePoint: int, destChannel: int) -> bool:
        """ Same as move channel in zarr.py, but for the use of multiImageLoaders
        """
        if srcTimePoint == destTimePoint and srcChannel == destChannel:
            return False
        
        if srcTimePoint >= len(self._imagesSrc) or destTimePoint >= len(self._imagesSrc):
            return False
        
        if srcChannel not in self._imagesSrc[srcTimePoint]:
            return False
    
        channel = self._imagesSrc[srcTimePoint].pop(srcChannel)
        name = self._metadata[srcTimePoint].channelNames.pop(srcChannel, None)

        if destChannel in self._imagesSrc[destTimePoint]:    
            self._imagesSrc[srcTimePoint][srcChannel] = self._imagesSrc[destTimePoint].pop(destChannel)
            if destChannel in self._metadata[destTimePoint].channelNames:
                self._metadata[srcTimePoint].channelNames[srcChannel] = self._metadata[destTimePoint].channelNames.pop(destChannel)

        self._imagesSrc[destTimePoint][destChannel] = channel
        if name is not None:
            self._metadata[destTimePoint].channelNames[destChannel] = name

    # def abb not used 20241221
    def _old_readAnalysisParams(self, analysisParams: Union[AnalysisParams, str]):
        """
        Set the analysisParams for the given time index.

        Args:
          analysisParams (AnalysisParams): The analysisParams.
        """

        if isinstance(analysisParams, str):
            with open(analysisParams, "r") as analysisParamsFile:
                analysisParams = AnalysisParams(loadJson=analysisParamsFile)

        self._analysisParams = analysisParams

    def timePoints(self) -> Iterator[int]:
        """
        Returns an iterator over the time points of the images.

        Returns:
            An iterator that yields the time points of the images.
        """
        return self._imagesSrc.keys()

    def channels(self, t: int) -> List[int]:
        return list(self._imagesSrc[t].keys())

    def _images(self, t: int, channel: int) -> np.ndarray:
        if t not in self._imagesSrc.keys():
            logger.error(f'abb got bad image key timepoint key {t}, available keys are {self.timePoints()}')
        return self._imagesSrc[t][channel]



