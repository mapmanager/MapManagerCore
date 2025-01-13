from dataclasses import dataclass, field
# JSON used by pyodide to transfer metadata to JS
# from dataclasses_json import dataclass_json
from typing import Dict, Literal, List

import numpy as np

from mapmanagercore.logger import logger

# @dataclass_json
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

# @dataclass_json
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

    # def __str__(self):
    #     ret = f'x:{self.x} y:{self.y} unit:{self.unit}'
    #     return ret

# @dataclass_json
@dataclass
class MetadataContrast:
    """
    """
    minInt: int = 1
    maxInt: int = 1
    minContrast: int = 1
    maxContrast: int = 1
    color : str = "0x00FF00"  # map -> 'green'
    
@dataclass
class Metadata:
    name: str = ''
    channelNames: Dict[int, str] = field(default_factory=lambda:{})
    voxel: VoxelMetadata = field(default_factory=lambda: VoxelMetadata())
    physicalSize: MetadataPhysicalSize = field(default_factory=lambda: MetadataPhysicalSize())
    metadataContrast : MetadataContrast = field(default_factory=lambda: MetadataContrast())

