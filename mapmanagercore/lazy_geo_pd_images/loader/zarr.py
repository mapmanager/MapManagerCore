import os
# import json
from typing import Any, Dict, Iterator, List, Union
import numpy as np
import zarr

from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from .base import ImageLoader, Position
from mapmanagercore.logger import logger

class ZarrLoader(ImageLoader):
    """A loader for images stored in a zarr file."""

    def __init__(self, path: Union[str, None] = None, lazy: bool = False):
        """
        Initializes a ZarrLoader object.

        Args:
            path (str): The path to the Zarr file.
            lazy (bool, optional): If True, the images will be loaded lazily.
                If False, the images will be loaded eagerly. Defaults to False.
        """
        super().__init__()

        if path is None:
            self._store = None
            self.group = zarr.group()
            self.group.create_group("images")
        else:
            # abb need to check if local files exist (not https)
            # _isHttps = path.startswith('https')
            # if _isHttps or os.path.isdir(path):
            if os.path.isdir(path):
                self._store = zarr.DirectoryStore(path)
            else:
                self._store = zarr.ZipStore(path, mode="r")
            self.group = zarr.group(store=self._store)

            # abb analysisparams
            try:
                loadedDict = self.group.attrs['analysisParams']
                self._analysisParams = AnalysisParams(loadedDict=loadedDict)
            except (KeyError) as e:
                logger.error(e)
                logger.error(f'available keys are {self.group.attrs.keys()}')
                self._analysisParams = AnalysisParams()

        self._imagesSrcs: List[Dict[int, np.ndarray]] = []
        self._metadata = []
        imagesGroup = self.group["images"]
        for t, group in imagesGroup.groups():
            t = int(t)
            # abb group.attrs["metadata"]) is now a dict
            # self._metadata.append(Metadata(json.loads(group.attrs["metadata"])))
            self._metadata.append(Metadata(group.attrs["metadata"]))
            channels = {}
            for channel, images in group.arrays():
                channel = int(channel)
                channels[channel] = images if lazy else images[:][:]

            self._imagesSrcs.append(channels)

        if path is None:
            # create a default single time point
            self.createTimePoint()

        self.path = path

    def analysisParams(self):
        return self._analysisParams

    def __str__(self):
        return f"Zarr Loader: path: {self.path}"

    def channels(self, t: int) -> List[int]:
        return list(self._imagesSrcs[t].keys())

    def createTimePoint(self) -> bool:
        self._metadata.append(Metadata(name="Unnamed Time Point"))
        self._imagesSrcs.append({})
        return True

    def appendChannelToTimePoint(self, srcTimePoint: int, srcChannel: int, destTimePoint: int) -> bool:
        if srcTimePoint == destTimePoint:
            return False
        
        if srcTimePoint >= len(self._imagesSrcs) or destTimePoint >= len(self._imagesSrcs):
            return False
        
        if srcChannel not in self._imagesSrcs[srcTimePoint]:
            return False
        
        dest = sorted(self._imagesSrcs[destTimePoint].keys())
        destChannel = len(dest) 
        for channel, index in enumerate(dest):
            if channel != index:
                destChannel = index
                break
            
        
        return self.moveChannel(srcTimePoint, srcChannel, destTimePoint, destChannel)
        

    def moveChannel(self, srcTimePoint: int, srcChannel: int, destTimePoint: int, destChannel: int) -> bool:
        if srcTimePoint == destTimePoint and srcChannel == destChannel:
            return False
        
        if srcTimePoint >= len(self._imagesSrcs) or destTimePoint >= len(self._imagesSrcs):
            return False
        
        if srcChannel not in self._imagesSrcs[srcTimePoint]:
            return False
    
        channel = self._imagesSrcs[srcTimePoint].pop(srcChannel)
        name = self._metadata[srcTimePoint].channelNames.pop(srcChannel, None)

        if destChannel in self._imagesSrcs[destTimePoint]:    
            self._imagesSrcs[srcTimePoint][srcChannel] = self._imagesSrcs[destTimePoint].pop(destChannel)
            if destChannel in self._metadata[destTimePoint].channelNames:
                self._metadata[srcTimePoint].channelNames[srcChannel] = self._metadata[destTimePoint].channelNames.pop(destChannel)

        self._imagesSrcs[destTimePoint][destChannel] = channel
        if name is not None:
            self._metadata[destTimePoint].channelNames[destChannel] = name
        return True

    def moveTimePoint(self, srcTimePoint: int, destTimePoint: int, position: Position = Position.OVER) -> bool:
        if srcTimePoint == destTimePoint:
            return False

        if position == Position.OVER:
            swap(self._metadata, srcTimePoint, destTimePoint)
            swap(self._imagesSrcs, srcTimePoint, destTimePoint)
            return True

        self._metadata.insert(destTimePoint, self._metadata.pop(srcTimePoint))
        self._imagesSrcs.insert(
            destTimePoint, self._imagesSrcs.pop(srcTimePoint))

        return True

    def deleteTimePoint(self, timePoint: int) -> bool:
        if timePoint >= len(self._metadata):
            return False

        self._metadata.pop(timePoint)
        self._imagesSrcs.pop(timePoint)
        return True

    def deleteChannel(self, timePoint: int, channel: int) -> bool:
        if timePoint >= len(self._metadata):
            return False

        a = self._metadata[timePoint].channelNames.pop(channel, None)
        b = self._imagesSrcs[timePoint].pop(channel, None)

        return a != None or b != None

    def merge(self, loader: ImageLoader):
        times = sorted(loader.timePoints())
        for time in times:
            metadata = loader.metadata(time)
            if time >= len(self._imagesSrcs):
                self._metadata.append(metadata)
                channels = {}
                for channel in loader.channels(time):
                    channels[channel] = loader._images(time, channel)
                self._imagesSrcs.append(channels)
                continue

            channels = self._imagesSrcs[time]
            destMetadata = self.metadata(time)
            for channel in loader.channels(time):
                channels[channel] = loader._images(time, channel)
                destMetadata.channelNames[channel] = metadata.channelNames[channel]

    def timePoints(self) -> Iterator[int]:
        """
        Returns an iterator over the time points of the images.

        Returns:
            An iterator that yields the time points of the images.
        """
        return range(len(self._imagesSrcs))

    def _images(self, t: int, channel: int) -> np.ndarray:
        images = self._imagesSrcs[t]
        if channel not in images:
            shape = next(iter(images.values())).shape
            return np.zeros(shape, dtype=np.uint8)
        return images[channel]

    def close(self):
        if self._store is not None:
            self._store.close()


def swap(data: List, key1, key2):
    data[key1], data[key2] = data[key2], data[key1]
