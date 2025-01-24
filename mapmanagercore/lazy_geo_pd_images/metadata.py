from dataclasses import dataclass, field, asdict
# JSON used by pyodide to transfer metadata to JS
from typing import Dict, Literal  # , List, Optional

import numpy as np

from mapmanagercore.utils import getAutoContrast
from mapmanagercore.logger import logger

@dataclass
class VoxelMetadata:
    """
    Represents the metadata for a voxel size in a 3D space.

    Attributes:
        x (float): The x-coordinate's voxel size.
        y (float): The y-coordinate's voxel size.
        z (float): The z-coordinate's voxel size.
    """
    x: float = 1
    y: float = 1
    z: float = 1

    # def __str__(self):
    #     ret = f'x:{self.x} y:{self.y} z:{self.z}'
    #     return ret

@dataclass
class MetadataPhysicalSize:
    """
    Represents the physical size of the image slices.

    Attributes:
        x (float): The size in the x-direction (width).
        y (float): The size in the y-direction (height).
        unit (Literal["µm"]): The unit of measurement.
    """
    x: float = 1
    y: float = 1
    unit: Literal["µm"] = "micrometer"  # abb from µm

@dataclass
class _metadataBase:
    """Abstract base class for all metadata dataclass.
    """
    def setValue(self, key, value):
        try:
            setattr(self, key, value)
            return True
        except (AttributeError) as e:
            logger.warning(e)
            
    def getValue(self, key):
        try:
            return getattr(self, key)
        except (AttributeError) as e:
            logger.warning(e)

    def asDict(self) -> dict:
        """Might be usefull to travers metadata when we do not know the key:value(s) enclosed.
        """
        return asdict(self)
    
@dataclass
class ExperimentMetadata(_metadataBase):
    """For each image acquired, user can specify `Experiment` metadata.
    
    Notes
    -----
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

@dataclass
class MetadataContrast:
    """
    Metadata for each color channel.
    """
    name: str = 'Undefined'
    """Name of the channel."""
    minInt: int = 1
    maxInt: int = 1
    minContrast: int = 1
    maxContrast: int = 1
    color : str = "x00FF00"  # map -> 'green'
    """Color LUT for the image."""

    def _initFromImgData(self, imgData : np.ndarray):
        self.minInt = int(np.min(imgData))
        self.maxInt = int(np.max(imgData))
        minContrast, maxContrast = getAutoContrast(imgData)
        self.minContrast = minContrast
        self.maxContrast = maxContrast
        
@dataclass
class Metadata():
    name: str = ''
    """Name of the session/timepoint/experiment."""

    experimentMetadata : ExperimentMetadata = field(default_factory=lambda: ExperimentMetadata())

    voxel: VoxelMetadata = field(default_factory=lambda: VoxelMetadata())
    physicalSize: MetadataPhysicalSize = field(default_factory=lambda: MetadataPhysicalSize())

    # do not modify, is used in >20x places
    channelNames: Dict[int, str] = field(default_factory=lambda:{})
    
    metadataContrast : list[MetadataContrast] = field(default_factory=lambda: [])
    """List of metadata for each color channel."""

    def addColorChannel(self, metadataContrast:MetadataContrast = MetadataContrast()):
        self.metadataContrast.append(metadataContrast)

    def asDict(self) -> dict:
        return asdict(self)