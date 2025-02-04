import numpy as np
import tifffile
import nd2

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata, MetadataContrast
from .base import ImageLoader
from typing import Iterator, List, Union
from mapmanagercore.utils import getAutoContrast
from mapmanagercore.logger import logger

class MultiImageLoader(ImageLoader):
    """
    Class for building an MultiImageLoader.
    """

    def __init__(self):
        super().__init__()
        self._imagesSrc = {}
        self._metadata = {}
        self.paths = []  # for logging only

        # abb md3
        from mapmanagercore.metadata3 import mmMapMetadata
        self._metadata3 = mmMapMetadata()

    def __str__(self):
        return f"Multi image Loader paths: {self.paths}"
    
    def metadata(self, t: int) -> Metadata:
        return self._metadata[t] if t in self._metadata else Metadata()

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
                with nd2.ND2File(path) as myfile:
                    voxel_size = myfile.voxel_size()
                imgData = nd2.imread(path)
                logger.info(f'{imgData.shape} voxel_size:{voxel_size}')
        else:
            # abb assuming np.ndarray?
            imgData = path

        if len(imgData.shape) == 2:
            # abb handle 2d, convert 2d image to 3d with one slice
            logger.warning(f'abb reshaping 2d image shape:{imgData.shape}')
            imgData = imgData.reshape((1, imgData.shape[1], imgData.shape[0]))
            logger.warning(f'  newshape is:{imgData.shape}')

        if time not in self._imagesSrc:
            # appending first channel to new timepoint
            
            self._imagesSrc[time] = {}
            
            # v1 meta
            _metaData = Metadata()
            # shape of imgData
            _numDims = len(imgData.shape)
            
            # abb we need to handle 2d images
            if _numDims == 3:
                _metaData.voxel.x = imgData.shape[2]
                _metaData.voxel.y = imgData.shape[1]
                _metaData.voxel.z = imgData.shape[0]
            elif _numDims == 2:
                _metaData.voxel.x = imgData.shape[1]
                _metaData.voxel.y = imgData.shape[0]
                # _metaData.voxel.z = imgData.shape[0]

            # physicaal units (um)
            _metaData.physicalSize.x = 0.15
            _metaData.physicalSize.y = 0.15
            if _numDims == 3:
                _metaData.physicalSize.z = 1

            # contrast
            # logger.info(f'abb MetadataContrast channel:{channel} {type(channel)} imgData:{imgData.shape}')
            metadataContrast = MetadataContrast()
            metadataContrast._initFromImgData(imgData)
            _metaData.addColorChannel(metadataContrast)

            self._metadata[time] = _metaData

            # abb md3
            from mapmanagercore.metadata3 import TimepointMetadata
            timepointMetadata = TimepointMetadata()  # empty, no channels
            timepointMetadata.appendChannel(imgData)
            self._metadata3.appendTimepoint(timepointMetadata)
        else:
            # abb md3, appending a channel to existing timepoint
            timepointMetadata = self._metadata3.getTimepointMetadata(time)
            timepointMetadata.appendChannel(imgData)

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
        return self._imagesSrc[t][channel]



