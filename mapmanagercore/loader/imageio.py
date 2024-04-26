import pandas as pd

from mapmanagercore.config import Metadata
from .base import ImageLoader, Loader
from typing import Dict, List, Tuple, Union
import numpy as np
import zarr


class MultiImageLoader(Loader):
    """
    Class for building an MultiImageLoader.
    """

    def __init__(self, lineSegments: Union[str, pd.DataFrame] = pd.DataFrame(), points: Union[str, pd.DataFrame] = pd.DataFrame(), metadata: Union[str, Metadata] = Metadata()):
        super().__init__(lineSegments, points, metadata)
        self._images = []

    def imread(path: str) -> ImageLoader:
        """
        Load an image using imageio.imread.

        Args:
          path (str): The path to the image.
        """
        from imageio import imread
        return _MultiImageLoader(imread(path))

    def read(self, path, time: int = 0, channel: int = 0):
        """
        Load an image from the given path and store it in the images array.

        Args:
          path (str): The path to the image file.
          time (int): The time index.
          channel (int): The channel index.
        """            
        from imageio import imread
        _imgData = imread(path)
        self._images.append([time, channel, _imgData])

        # abb TODO: hold off on this, will try and make a map with one stack meta data
        # as is already done in the code.
        # here I was thinking each timepoint image volume would have its own
        # if len(self._images) == 1:
        #     # make metadata
        #     self._createMetaData(_imgData)
        # else:
        #     # add a color channel
        #     self._metadata['size']['c'] += 1

    # not used
    def _createMetaData(self, imgData : np.array):
        """On first image load, create metadata
       
        Notes
        -----
        This can just represent metadata for one timepoint (multipl channels)?
        With that, we can remove:
            MetaData.SizeMetadata.t
            
        """
    
        # from typing import Literal
        from mapmanagercore.config import Metadata, SizeMetadata, VoxelMetadata, MetadataPhysicalSize

        # time = 0  # on create, always the first timepoint
        numChannels = 1  # on first image, always one channel

        dtype = imgData.dtype
        numSlices, x, y = imgData.shape

        xVoxel = 1  # 0.12
        yVoxel = 1  # 0.12
        zVoxel = 1
        unit = ""  # "µm"

        xPhysical = x * xVoxel
        yPhysical = y * yVoxel
        zPhysical = numSlices * zVoxel

        self._metadata = Metadata(
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
        
    def images(self) -> ImageLoader:
        maxTime = max(time for time, _, _ in self._images) + 1
        maxChannel = max(channel for _, channel, _ in self._images) + 1
        maxSlice, maxX, maxY = self._images[0][2].shape

        # dimensions = [maxChannel, maxSlice, maxX, maxY]
        dimensions = [maxTime, maxChannel, maxSlice, maxX, maxY]
        # images = { t: np.zeros(dimensions, dtype=np.uint16) for t, _, i in  self._images}
        images = np.zeros(dimensions, dtype=np.uint16)

        for time, channel, image in self._images:
            images[time][channel] = image

        return _MultiImageLoader(images)


class _MultiImageLoader(ImageLoader):
    """
    A loader class for loading from imageio supported formats.
    """

    def __init__(self, images: np.ndarray):
        """
        Initialize the BaseImage class.

        Args:
          images (np.ndarray): [time, channel, slice].

        """
        super().__init__()
        self._images = images

    def shape(self) -> Tuple[int, int, int, int, int]:
        return self._images.shape

    def saveTo(self, group: zarr.Group):
        # for time, images in self._images:
            # group.create_dataset(f"t-{time}", data=images, dtype=images.dtype)
        group.create_dataset("images", data=self._images,
                             dtype=self._images.dtype)

    def loadSlice(self, time: int, channel: int, slice: int) -> np.ndarray:
        return self._images[time][channel][slice]

    def fetchSlices(self, time: int, channel: int, sliceRange: Tuple[int, int]) -> np.ndarray:
        return np.max(self._images[time][channel][sliceRange[0]:sliceRange[1]], axis=0)
