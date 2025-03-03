"""Utility class to import image data and metadata from files into MapManager.

MapManager can import a wide range of file types.
Each file type is specified by its file-name extension, for example `.tif`.

To do this, MapManager relies on the [bioio](https://github.com/bioio-devs/bioio) Python package.
Actually, MapManager uses a forked version [mapmanager/bioio](https://github.com/mapmanager/bioio).
This was created so the same code-base can be used in the two main MapManager GUI interfaces including 
the desktop [PyMapManager](https://github.com/mapmanager/PyMapManager) 
and web [WebMapManager](https://github.com/mapmanager/WebMapManager).

The bioio package is developed by 
[The Allen Institute for Cell Science](https://alleninstitute.org/division/cell-science/) to which we are indepted!


Current supported file type extensions are:

 - tif
 - ome.tif
 - nd2 (Nikon)
 - czi (Zeiss)

 We will be expanding this to include other file formats such as `oir` (Olympus).

To get the currently supported file types

```python
import mapmanagercore
from mapmanagercore.imageImporter import acceptedExtensions
acceptedExtensions()
```

How to use:

```python
from mapmanagercore.imageImporter import getImageImporter
ii = getImageImporter(path)
if ii is None:
    print('load failed')
else:
    print(ii.numChannels)
    print(ii.channelShape)
    print(ii.physicalPixelSizes)

    # load the pixel data
    ii.loadData()

    # get one channel pixel data
    channelData = ii.getChannelData(channelIdx=0)
```
"""
import os
from typing import Tuple, List, Optional

import numpy as np

import bioio_base.exceptions
import bioio

from mapmanagercore.metadata3 import TimepointMetadata, VoxelMetadata
from mapmanagercore.logger import logger

def acceptedExtensions() -> List[str]:
    """Get list of accepted extensions from bioio.

    This list will depend on bioio plugins installed with `pip install`.
    """
    _acceptedExtensions = []
    report = bioio.plugins.get_plugins(use_cache=False)
    for item in report.items():
        _acceptedExtensions.append(item[0])
    return _acceptedExtensions

class _ImageImporter():
    """Import image data from file.
    """
    
    def __init__(self,
                 path: str,
                 loadImgData:bool = False):
        """
        Args:
            path: The file path to load from.
            loadImgData: If true then load the img data from path/file. Default is False.

        Raises:
            bioio.exceptions.UnsupportedFileFormatError: No bioio reader could be found that supports the provided image.
            FileNotFoundError: File was not found.

        Notes:
            Use standalone function getImageImporter() during runtime.
        """
        
        # check if file exists (does not work for url like http).
        try:
            with open(path) as _file:
                pass
        except FileNotFoundError as e:
            logger.error(e)
            raise e

        # check we know how to open it using extension
        try:
            plugin = bioio.BioImage.determine_plugin(path)
            # logger.info(f'using plugin:{plugin}')
        except (bioio_base.exceptions.UnsupportedFileFormatError) as e:
            #logger.error(e)
            logger.error(f'did not load path:{path}')
            logger.error(f'  bioio accepted extensions are: {acceptedExtensions()}')
            raise e

        self._path = path

        # load image meatadata (not raw pixels)
        # reader=None, BioImage will determine loader (in future specifically specify things like TiffLoader)
        # This will be needed because bioio loaders like bio-formats includes TiffFile
        # but we want to use the actual TiffFile loader (not bio-format)

        _reader = None  
        self._img: bioio.BioImage = bioio.BioImage(path, reader=_reader) 
        """BioImage that provides metadata info and lazy loading of pixels."""
        
        self._imgDataList: List[np.ndarray] = []
        """List of np.ndarray, one index per color channel. See loadData()"""

        self._timepointMetadata: TimepointMetadata = None
        """TimepointMeta data including info on each color channel."""

        # logger.info(f'created _ImageImporter with loadImgData:{loadImgData} path: {path}')
        # logger.info(f'  numChannels:{self.numChannels} shape:{self.channelShape}')

        if loadImgData:
            self.loadData()
            self.getTimepointMetadata()

        # example API for BioImage
        # logger.info(f'  img:{type(img)}')
        # logger.info(f'  img.dims:{img.dims}')
        # logger.info(f"  img.dims['C']:{img.dims['C']}")
        # logger.info(f'  img.dims.order:{img.dims.order}')
        # logger.info(f'  img.shape:{img.shape}')
        # logger.info(f'  img.dtype:{img.dtype}')
        # logger.info(f'  img.physical_pixel_sizes:{img.physical_pixel_sizes}')
        # logger.info(f'  img.channel_names:{img.channel_names}')

    def __str__(self):
        return (
            '\n_ImageImporter'
            f'  path:{self.path}\n'
            f'  num channels:{self.numChannels}\n'
            f'  channelShape:{self.channelShape}\n'
            f'  physicalPixelSizes:{self.physicalPixelSizes}\n'
        )
    
    @property
    def path(self) -> str:
        """Path to file we loaded from.
        """
        return self._path
    
    @property
    def filename(self) -> str:
        return os.path.split(self.path)[1]
    
    @property
    def numChannels(self) -> int:
        return self._img.dims['C'][0]

    @property
    def channelShape(self) -> Tuple[int, int, int]:
        """Get the channel pixel shape like (z, y, x).

        Notes:
            All channels in a file have the same shape.
        """
        # TCZYX (1, 2, 7, 784, 784)
        shape =  list(self._img.shape)[2:]
        shape = tuple(shape)
        return shape
    
    @property
    def physicalPixelSizes(self) -> Tuple[float, float, float]:
        """Get ZYX voxel size. Usually in um.

        Notes:
            - All channels in a file have the same physical sizes.
            - Some readers set Z=None when Z=1, mapping Z is None to 1
        """
        physicalPixelSizes = self._img.physical_pixel_sizes  # NamedTuple
        zVoxel = 1 if physicalPixelSizes.Z is None else physicalPixelSizes.Z
        yVoxel = physicalPixelSizes.Y
        xVoxel = physicalPixelSizes.X
        return (zVoxel, yVoxel, xVoxel)
    
    @property
    def channelNames(self) -> List[str]:
        names = [str(nameStr) for nameStr in self._img.channel_names]
        return names
    
    def getChannelData(self, channelIdx:int) -> Optional[np.ndarray]:
        """Get image data for one channel.
        """
        if channelIdx > self.numChannels-1:
            logger.error(f'bad channel index {channelIdx}, num channels is {self.numChannels}')
            return
        return self._imgDataList[channelIdx]

    def loadData(self) -> List[np.ndarray]:
        """Load all image data and store each channel in _imgDataList[]
        """
        # The .data and .xarray_data properties will load the whole scene into memory.
        # The .get_image_data function will load the whole scene into memory and then retrieve the specified chunk.
        if len(self._imgDataList) > 0:
            logger.error('already loaded')
            return self._imgDataList
        
        for channelIdx in range(self.numChannels):
            _imgData = self._img.get_image_data("ZYX", T=0, C=channelIdx)  # returns 4D CZYX numpy array
            self._imgDataList.append(_imgData)
            # logger.info(f'  appended channelIdx:{channelIdx} with shape:{_imgData.shape}')

        return self._imgDataList
    
    def getTimepointMetadata(self) -> TimepointMetadata:
        """Get TimepointMetadata from loaded img data.
        
        Notes:
            Need to call loadData() first.
        """
        if len(self._imgDataList) == 0:
            logger.warning(f'num channels is {self.numChannels}, no image data loaded. Did you call loadData()?')
            return
        
        if self._timepointMetadata is None:
            md = TimepointMetadata(name=self.filename)

            zVoxel, yVoxel, xVoxel = self.physicalPixelSizes
            voxelMetadata = VoxelMetadata(zVoxel=zVoxel, yVoxel=yVoxel, xVoxel=xVoxel)
            md.setVoxelMetadata(voxelMetadata)

            for channelIdx in range(self.numChannels):
                userChannelName= self.channelNames[channelIdx]
                md.appendChannel(imgData=self._imgDataList[channelIdx], name=userChannelName)

            self._timepointMetadata = md

        return self._timepointMetadata
    
def getImageImporter(path:str, loadImgData:bool=False) -> Optional[_ImageImporter]:
    """Get an ImageImporter from a path.
    
    Args:
        path: Path to a file
        loadImgData: If True then automatically load image data, otherwise need to call loadData()

    Return:
        ImageImporter (None on failure)

    Notes:
        Will fail (return None) when extension is not supported or file does not exist.
    """
    try:
        ii = _ImageImporter(path, loadImgData)
    except (bioio_base.exceptions.UnsupportedFileFormatError, FileNotFoundError):
        return
    if ii.numChannels == 0:
        logger.error('did not find any channels in file')
        return
    
    return ii
