# import json
import numpy as np

from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from .base import ImageLoader
from typing import Iterator, Union
from mapmanagercore.utils import getAutoContrast
from mapmanagercore.logger import logger

class MultiImageLoader(ImageLoader):
    """
    Class for building an MultiImageLoader.
    """

    def __init__(self):
        super().__init__()
        # self._images = {}
        self._imagesLoaded = {} # changed name to prevent deletion of images
        self.paths = [] # for logging only
        
    def __str__(self):
        # return f"Multi image Loader paths: {self.paths}"
        retStr = ''
        for _timePoint, _imgList in self._images.items():
            for _channel, imgData in _imgList:
                retStr += f'   tp:"{_timePoint}" ch:{_channel} {imgData.shape} {imgData.dtype}\n'
        return retStr
    
    def read(self, path: Union[str, np.ndarray], time: int = 0, channel: int = 0):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): Either the path to the image file or a np array.
          time (int): The time index.
          channel (int): The channel index.
        """
        # TODO do not use imageio, use bioio
        # note, imageio is silently installed when scikit-image is installed
        # to update, see mapmanagercore.image_importers
        from imageio import imread
        if time not in self._imagesLoaded:
            logger.info(f"time not in MultiImageLoader")
            self._imagesLoaded[time] = []
            # abb
            self._metadata[time] = Metadata()

        if isinstance(path, str):
            imgData = imread(path)
        else:
            imgData = path

        self._imagesLoaded[time].append([channel, imgData])

        # abb
        # logger.info(f'setting metadata time:{time} channel:{channel} imgData:{imgData.shape}')
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

        # self._metadata[time].append([channel, _metaData])
        self._metadata[time] = _metaData
        
        # logger.info(f"compare 1 {self._imagesLoaded}") # abj
        self.paths.append([time, channel, path])

    # abb TODO: this is never called?
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

        self._metadata[time] = metadata

    def build(self, currentImages = None) -> ImageLoader:
        """

        currentImages: Current Image that is already loaded. This is called from _MultiImagerLoader
        """

        # logger.info(f"currentImages compare {currentImages}")
        if currentImages is not None:
            self._imagesLoaded = currentImages
        
        images = {}
        
        # self._imagesLoaded[time].append([channel, imgData])
        # Problem imagesLoaded is in a different format!

        for time, values in self._imagesLoaded.items():
            # if not (time in self._metadata):
            #     raise ValueError(f"Metadata not found for time point {time}")
            # logger.info(f"values, {values} ")
            maxChannel = max(channel for channel, _ in values) + 1
            maxSlice, maxX, maxY = values[0][1].shape
            dimensions = [maxChannel, maxSlice, maxX, maxY]
            images[time] = np.zeros(dimensions, dtype=np.uint16)
            for channel, image in values:
                images[time][channel] = image

        # if there are already images, return data to set within current _MultiImageLoader object
        if currentImages is not None: 
            return images, self._metadata

        # else create new image loader object
        return _MultiImageLoader(images, self._metadata)

# class _MultiImageLoader(ImageLoader):
# abj - inherit from MultiImagerLoader to be able to read and build new channels
class _MultiImageLoader(MultiImageLoader):
    """
    A loader class for loading from imageio supported formats.
    """

    def __init__(self, images: dict[int, np.ndarray], metadata: dict[int, Metadata]):
        """
        Initialize the BaseImage class.

        Args:
          images (np.ndarray): [time, channel, slice].

        """
        super().__init__()
        self._imagesSrcs = images
        self._metadata = metadata
        self._paths = []

    def timePoints(self) -> Iterator[int]:
        """
        Returns an iterator over the time points of the images.

        Returns:
            An iterator that yields the time points of the images.
        """
        return self._imagesSrcs.keys()

    def _images(self, t: int) -> np.ndarray:
        return self._imagesSrcs[t]

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
            imgData = imread(path)
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


