from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata
from .base import ImageLoader
from typing import Iterator, List, Union
import numpy as np

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
        from imageio import imread
        if name == None:
            name = path

        if time not in self._imagesSrc:
            self._imagesSrc[time] = {}
            self._metadata[time] = Metadata()

        if isinstance(path, str):
            imgData = imread(path)
        else:
            imgData = path

        if channel > self.maxChannels():
            self.setMaxChannels(channel)

        self._imagesSrc[time][channel] = imgData
        self._metadata[time].channelNames[channel] = name
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

        if time in self._metadata:
            oldMetadata = self._metadata[time]
            for channel in oldMetadata.channelNames:
                if channel not in metadata.channelNames:
                    metadata.channelNames[channel] = oldMetadata.channelNames[channel]

        self._metadata[time] = metadata

    def readAnalysisParams(self, analysisParams: Union[AnalysisParams, str]):
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
