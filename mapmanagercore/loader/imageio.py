import json
import pandas as pd

from mapmanagercore.config import Metadata
from .base import ImageLoader, Loader
from typing import Iterator, Union
import numpy as np


def _createMetaData(imgData : np.ndarray,
                    maxSlices : int = None,
                    numChannels : int = 1) -> Metadata:
    """Get MetaData with a seed image volume.
    
    Parameters
    ----------
    imgData : np.ndarray
        Template image/volume to get size and dtype
    maxSlices : int
        Maximum number of slices in a timeseries/map.
        Used when creating one metadata for a timeseries with
            different number of slices per timepoint.
    numChannels : int
        The number of channels in final map,
            imgData is often just one channel of many

    Notes
    -----
    In final version, this can be metadata for one timepoint (multiple channels)?
    With that, we can remove:
        MetaData.SizeMetadata.t
        
    """

    # from typing import Literal
    from mapmanagercore.config import Metadata, SizeMetadata, VoxelMetadata, MetadataPhysicalSize

    # time = 0  # on create, always the first timepoint
    #numChannels = 1  # on first image, always one channel

    dtype = imgData.dtype
    numSlices, x, y = imgData.shape

    if maxSlices is not None:
        numSlices = maxSlices

    xVoxel = 1  # 0.12
    yVoxel = 1  # 0.12
    zVoxel = 1
    unit = "pixels"  # "µm"

    xPhysical = x * xVoxel
    yPhysical = y * yVoxel
    zPhysical = numSlices * zVoxel

    _metadata = Metadata(
        size=SizeMetadata(x=x,
                        y=y,
                        z=numSlices,
                        # t=time,  # the timepoint of the image
                        c=numChannels),  # number of color channels, will have to update as more are added
        voxel=VoxelMetadata(x=xVoxel,
                            y=yVoxel,
                            z=zVoxel),
        dtype=str(dtype),  # Literal['Uint16'],
        physicalSize=MetadataPhysicalSize(x=xPhysical,
                                y=yPhysical,
                                z=zPhysical,
                                unit=unit)
        )
    
    return _metadata

class MultiImageLoader(Loader):
    """Class for building an MultiImageLoader.
    """

    def __init__(self,
                 lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(),
                 points: Union[str, pd.DataFrame] = pd.DataFrame()):
        super().__init__(lineSegments, points)
        self._images = {}
        self._metadata = {}

    def getZarrPath(self):
        return 'MultiImageLoader no zarr path'
    
    def imread(path: str) -> ImageLoader:
        """
        Load an image using imageio.imread.

        Args:
          path (str): The path to the image.
        """
        from imageio import imread
        return _MultiImageLoader(imread(path))

    # abb
    # def getNumTimepoints(self):
    #    return len(self._images.keys())
     
    def read(self, path : Union[str, np.ndarray], time: int = 0, channel: int = 0):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): Either the path to the image file or a np array.
          time (int): The time index.
          channel (int): The channel index.
        """
        from imageio import imread
        if time not in self._images:
            self._images[time] = []

        if isinstance(path, str):
            imgData = imread(path)
        else:
            imgData = path

        self._images[time].append([channel, imgData])

    def readMetadata(self, metadata: Union[Metadata, str], time: int = 0):
        """
        Set the metadata for the given time index.

        Args:
          time (int): The time index.
          metadata (Metadata): The metadata.
        """

        if isinstance(metadata, str):
            with open(metadata, "r") as metadataFile:
                metadata = json.load(metadataFile)

        self._metadata[time] = metadata

    def images(self) -> ImageLoader:
        images = {}

        for time, values in self._images.items():
            if not (time in self._metadata):
                raise ValueError(f"Metadata not found for time point {time}")

            maxChannel = max(channel for channel, _ in values) + 1
            maxSlice, maxX, maxY = values[0][1].shape
            dimensions = [maxChannel, maxSlice, maxX, maxY]
            images[time] = np.zeros(dimensions, dtype=np.uint16)
            for channel, image in values:
                images[time][channel] = image

        return _MultiImageLoader(images, self._metadata)


class _MultiImageLoader(ImageLoader):
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

    def timePoints(self) -> Iterator[int]:
        return self._imagesSrcs.keys()

    def _images(self, t: int) -> np.ndarray:
        return self._imagesSrcs[t]

    def metadata(self, t: int) -> Metadata:
        return self._metadata[t] if t in self._metadata else Metadata()
