import numpy as np
import tifffile

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
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
        # TODO for tif files, do not use imageio, use tifffile
        # note, imageio is silently installed when scikit-image is installed
        # to update, see mapmanagercore.image_importers
        # from imageio import imread
        if name is None:
            name = path

        if isinstance(path, str):
            # imgData = imread(path)
            imgData = tifffile.imread(path)
        else:
            imgData = path
            
        if time not in self._imagesSrc:
            self._imagesSrc[time] = {}
            _metaData = Metadata()
            # shape of imgData
            _metaData.voxel.x = imgData.shape[2]
            _metaData.voxel.y = imgData.shape[1]
            _metaData.voxel.z = imgData.shape[0]

            # physicaal units (um)
            _metaData.physicalSize.x = 0.15
            _metaData.physicalSize.y = 0.15
            _metaData.physicalSize.z = 1

            # contrast
            _metaData.metadataContrast.color = 'TODO: fix this'
            _metaData.metadataContrast.minInt = int(np.min(imgData))
            _metaData.metadataContrast.maxInt = int(np.max(imgData))
            minContrast, maxContrast = getAutoContrast(imgData)
            _metaData.metadataContrast.minContrast = minContrast
            _metaData.metadataContrast.maxContrast = maxContrast
            self._metadata[time] = _metaData

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

    def readNewImages(self, path: Union[str, np.ndarray], time: int = 0, channel: int = 0):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): Either the path to the image file or a np array.
          time (int): The time index.
          channel (int): The channel index.
        """

        currentImages = {} # reformatted current images
        if time not in currentImages:
            logger.info(f"time not in current Images")
            currentImages[time] = []

        if isinstance(path, str):
            # from imageio import imread
            imgData = tifffile.imread(path)
        else:
            imgData = path

        # self._imagesSrcs is current image
        # append to it with new channel
        # self._imagesSrcs[time].append([channel, imgData]
        # Format of self._imagesSrcs: images[time][channel] = image

        # print("self._imagesSrcs", self._imagesSrcs)

        # Reformatting current images so that we can append new one right after
        channelCount = -1
        for time in self._imagesSrcs:
            # print("time: ", time)
            for channelImage in self._imagesSrcs[time]:
                channelCount += 1
                # .append([channel, imgData])
                print("channelCount", channelCount)
                # self._imagesLoaded[time].append([channel, imgData])
                currentImages[time].append([channelCount, channelImage])

                # TODO: create new metaData
        
        # TODO: need to create functionality for when user wants to switch channel numbers
        # TODO: need to check to make sure new image channel has same size as previous  image channel
        # append new images (channel)
        if channel is None:
            newChannel = channelCount + 1
        else:
            newChannel = channel
         
        currentImages[time].append([newChannel, imgData])

        logger.info(f"compare 2 {currentImages}")
        
        # rebuild these images to correct form
        newImages, metaData = self.build(currentImages = currentImages)

        # set these images
        self._imagesSrcs = newImages
        self._metadata = metaData


